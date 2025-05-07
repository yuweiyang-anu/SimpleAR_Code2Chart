import os
import json
import time
import random
import argparse
import numpy as np
from tqdm import tqdm
from typing import List
import torch
from torchvision.utils import save_image
from vllm import SamplingParams
import torch.multiprocessing as mp

from simpar.model.tokenizer.cosmos_tokenizer.networks import TokenizerConfigs
from simpar.model.tokenizer.cosmos_tokenizer.video_lib import CausalVideoTokenizer as CosmosTokenizer
from simpar.model.builder import vllm_t2i, load_pretrained_model
from simpar.utils import disable_torch_init


@torch.inference_mode()
def generate_one(prompt_dict, model, vq_model, tokenizer, args, save_base_path, codebook_size, latent_size, max_new_tokens):
    save_name = os.path.join(save_base_path, prompt_dict["id"][:-3] + ".png")
    prompt = prompt_dict["prompt"]
    print('=====Current Prompt:=====', prompt)
    format_prompt = "<|t2i|>" + "A detailed digital painting of " + prompt + "<|soi|>"
    input_ids = tokenizer(format_prompt, return_tensors="pt").input_ids.to(args.device)

    uncond_prompt = "<|t2i|>" + "An image of aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion" + "<|soi|>"
    uncond_input_ids = tokenizer(uncond_prompt, return_tensors="pt").input_ids.to(args.device)

    if not args.vllm_serving:
        output_ids = model.generate_visual(
            input_ids,
            negative_prompt_ids=uncond_input_ids,
            cfg_scale=args.cfg_scale,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=max_new_tokens,
            use_cache=True
        )
        index_sample = output_ids[:, input_ids.shape[1]: input_ids.shape[1] + max_new_tokens].clone()

    else:
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=max_new_tokens,
            guidance_scale=args.cfg_scale
        )
        input_dict = {
            "prompt_token_ids": input_ids.squeeze(0),
            "negative_prompt_token_ids": uncond_input_ids.squeeze(0)
        }
        outs = model.generate(input_dict, sampling_params, use_tqdm=False)
        output_id_tensor = torch.tensor(outs[0].outputs[0].token_ids, dtype=input_ids.dtype, device=input_ids.device)
        index_sample = output_id_tensor.clone()

    # VQGAN decoding
    index_sample = index_sample - len(tokenizer)
    index_sample = torch.clamp(index_sample, min=0, max=codebook_size - 1)
    index_sample = index_sample.reshape(-1, latent_size, latent_size).unsqueeze(1)

    samples = vq_model.decode(index_sample)
    samples = samples.squeeze(2)
    save_image(samples, save_name, normalize=True, value_range=(-1, 1))


def run_one_gpu(rank: int, world_size: int, all_prompts: List[dict], args):
    args.device = f"cuda:{rank}"
    disable_torch_init()
    torch.cuda.set_device(rank)

    # Seed for reproducibility
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    tokenizer_config = TokenizerConfigs["DV"].value
    tokenizer_config.update(dict(spatial_compression=16, temporal_compression=8))
    vq_model = CosmosTokenizer(
        checkpoint_enc=f"{args.vq_model_ckpt}/encoder.jit",
        checkpoint_dec=f"{args.vq_model_ckpt}/decoder.jit",
        tokenizer_config=tokenizer_config
    )
    vq_model.eval().requires_grad_(False)

    if not args.vllm_serving:
        tokenizer, model, _, _ = load_pretrained_model(
            args.model_path, attn_implementation="sdpa", device_map=args.device
        )
    else:
        tokenizer, model = vllm_t2i(model_path=args.model_path)

    codebook_size = 64000
    downsample_size = 16
    latent_size = args.image_size // downsample_size
    max_new_tokens = latent_size ** 2

    for main_key, prompts_list in all_prompts.items():
        chunk_size = (len(prompts_list) + world_size - 1) // world_size
        sub_list = prompts_list[rank * chunk_size:(rank + 1) * chunk_size]
        if not sub_list:
            continue
        save_path = os.path.join(args.save_dir, main_key.split("_")[1])
        os.makedirs(save_path, exist_ok=True)

        for prompt_dict in tqdm(sub_list, desc=f"[GPU {rank}] {main_key}", position=rank):
            generate_one(prompt_dict, model, vq_model, tokenizer, args, save_path, codebook_size, latent_size, max_new_tokens)

def collect_py_files_as_prompts(folder_path):
    result = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".py"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            key = filename[:-3]
            if key not in result:
                result[key] = []
            result[key].append({
                "id": filename,
                "prompt": f"```python\n{content}\n```"
            })
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/openseg_blob/v-yuweiyang/SimpleAR/ckpt/SimpleAR-1.5B-RL")
    parser.add_argument("--vq-model-ckpt", type=str, default="/openseg_blob/v-yuweiyang/SimpleAR/ckpt/cosmos")
    parser.add_argument("--code_path", type=str, default="/openseg_blob/v-yuweiyang/chart-foundation-model/benchmark_test/chartmmic/ori_500/")
    parser.add_argument("--save_dir", type=str, default="/openseg_blob/v-yuweiyang/SimpleAR_Code2Chart/results/chart2code_simplear_1.5B_RL_on_chartmmic/")
    parser.add_argument("--vllm_serving", action="store_true")
    parser.add_argument("--image-size", type=int, choices=[256, 512, 768, 1024], default=1024)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=64000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg-scale", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    all_prompts = collect_py_files_as_prompts(args.code_path)

    world_size = torch.cuda.device_count()
    mp.spawn(run_one_gpu, args=(world_size, all_prompts, args), nprocs=world_size)


if __name__ == "__main__":
    main()
