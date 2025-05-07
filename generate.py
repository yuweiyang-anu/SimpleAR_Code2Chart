import os
import time
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torchvision.utils import save_image
from vllm import SamplingParams

from simpar.model.tokenizer.cosmos_tokenizer.networks import TokenizerConfigs
from simpar.model.tokenizer.cosmos_tokenizer.video_lib import CausalVideoTokenizer as CosmosTokenizer
from simpar.model.builder import vllm_t2i, load_pretrained_model
from simpar.utils import disable_torch_init

@torch.inference_mode()
def generate(model, vq_model, tokenizer, prompts, save_dir, args):
    codebook_size = 64000
    downsample_size = 16
    latent_size = args.image_size // downsample_size
    max_new_tokens=latent_size ** 2

    vq_time, llm_time = 0, 0
    for prompt in tqdm(prompts):
        format_prompt = "<|t2i|>" + "A detailed digital painting of " + prompt + "<|soi|>"
        input_ids = tokenizer(format_prompt, return_tensors="pt").input_ids.to(args.device)
        uncond_prompt = "<|t2i|>" + "An image of aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion" + "<|soi|>"
        uncond_input_ids = tokenizer(uncond_prompt, return_tensors="pt").input_ids.to(args.device)

        if not args.vllm_serving: # inference with hf
            t1 = time.time()
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
            sampling_time = time.time() - t1
            llm_time += sampling_time
            index_sample = output_ids[:, input_ids.shape[1]: input_ids.shape[1] + max_new_tokens].clone()
        
        else:
            if args.top_k is None:
                sampling_params = SamplingParams(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=max_new_tokens,
                    guidance_scale=args.cfg_scale
                )
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
            t1 = time.time()
            with torch.inference_mode():
                outs = model.generate(
                    input_dict,
                    sampling_params,
                    use_tqdm=False
                )
            print(outs)
            sampling_time = time.time() - t1
            llm_time += sampling_time

            output_id_tensor = torch.tensor(outs[0].outputs[0].token_ids, dtype=input_ids.dtype, device=input_ids.device)
            index_sample = output_id_tensor.clone()    
        
        # VQGAN decoding
        index_sample = index_sample - len(tokenizer)
        index_sample = torch.clamp(index_sample, min=0, max=codebook_size-1)
        index_sample = index_sample.reshape(-1, latent_size, latent_size).unsqueeze(1)

        t2 = time.time()
        with torch.inference_mode():
            samples = vq_model.decode(index_sample)
        
        vq_time += (time.time() - t2)
        
        samples = samples.squeeze(2)
        save_image(samples, os.path.join(save_dir, f"{prompt[:50]}.png"), normalize=True, value_range=(-1, 1))
    
    print(f"Averaged LLM time: {llm_time / len(prompts)}, averaged VQ time: {vq_time / len(prompts)}")
    return


def main(args):
    # Model
    disable_torch_init()

    # seed everything
    seed = args.seed
    random.seed(seed)              # Set Python random seed
    np.random.seed(seed)           # Set NumPy random seed
    torch.manual_seed(seed)        # Set PyTorch CPU seed
    torch.cuda.manual_seed(seed)   # Set PyTorch CUDA seed
    torch.cuda.manual_seed_all(seed)  # For multi-GPU inference
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Avoid non-deterministic optimizations

    tokenizer_config = TokenizerConfigs["DV"].value
    tokenizer_config.update(dict(spatial_compression=16, temporal_compression=8))
    vq_model = CosmosTokenizer(checkpoint_enc=f"{args.vq_model_ckpt}/encoder.jit", checkpoint_dec=f"{args.vq_model_ckpt}/decoder.jit", tokenizer_config=tokenizer_config)

    vq_model.eval()
    vq_model.requires_grad_(False)

    model_path = os.path.expanduser(args.model_path)
    if not args.vllm_serving:
        tokenizer, model, _, _  = load_pretrained_model(model_path, attn_implementation="sdpa", device_map=args.device)
    else:
        tokenizer, model = vllm_t2i(model_path=model_path)

    generate(model, vq_model, tokenizer, args.prompts, args.save_dir, args)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/openseg_blob/v-yuweiyang/SimpleAR/ckpt/SimpleAR-1.5B-RL") # SimpleAR-1.5B-SFT
    parser.add_argument("--vq-model-ckpt", type=str, default="/openseg_blob/v-yuweiyang/SimpleAR/ckpt/cosmos")
    parser.add_argument("--prompts", nargs="+", default=["Illustrate the growth stages of a carrot from seed to harvest."])
    parser.add_argument("--save_dir", type=str, default="/openseg_blob/v-yuweiyang/SimpleAR_Code2Chart/results/visualization_test")
    parser.add_argument("--vllm_serving", action="store_true")
    parser.add_argument("--image-size", type=int, choices=[256, 512, 768, 1024], default=1024)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=64000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg-scale", type=float, default=6.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)