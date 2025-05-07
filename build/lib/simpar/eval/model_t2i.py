import os
import time
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import transformers
from vllm import SamplingParams

from simpar.model.tokenizer.cosmos_tokenizer.networks import TokenizerConfigs
from simpar.model.tokenizer.cosmos_tokenizer.video_lib import CausalVideoTokenizer as CosmosTokenizer
from simpar.model.builder import load_pretrained_model, vllm_t2i
from simpar.utils import disable_torch_init
from simpar.mm_utils import get_model_name_from_path
from simpar.train.t2i_data import EvalT2IDataset

@dataclass
class DataCollatorForT2IEvalDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, batch):        
        input_ids = [instances[1][: self.tokenizer.model_max_length] for instances in batch]
        uncond_input_ids = [instances[2][: self.tokenizer.model_max_length] for instances in batch]
        if self.tokenizer.pad_token_id is None:
            # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # FIXME: this could only be triggered for llama3 model.
            self.tokenizer.pad_token_id = 0 # This gets the best result. Don't know why.
        input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        uncond_input_ids = self.pad_sequence(uncond_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_masks=input_ids.ne(self.tokenizer.pad_token_id)
        uncond_attention_masks=uncond_input_ids.ne(self.tokenizer.pad_token_id)
        
        try:
            images_list = [instances[0] for instances in batch]
            all_images = torch.stack(images_list)
        except:
            all_images = [instances[0] for instances in batch]
        
        image_paths = [instances[-1] for instances in batch]

        return all_images, input_ids, attention_masks, uncond_input_ids, uncond_attention_masks, image_paths


@torch.inference_mode()
def evaluate(model, vq_model, tokenizer, dataloader, save_dir, args):
    codebook_size = 64000
    downsample_size = 16
    latent_size = args.image_size // downsample_size
    max_new_tokens=latent_size ** 2

    # MLLM sampling
    llm_time = vq_time = 0

    i = 0
    for _, input_ids, attention_masks, uncond_input_ids, uncond_attention_masks, image_paths in tqdm(dataloader):
        input_ids = input_ids.to(args.device)
        attention_masks = attention_masks.to(args.device)
        uncond_input_ids = uncond_input_ids.to(args.device)
        uncond_attention_masks = uncond_attention_masks.to(args.device)

        if not args.vllm_serving: # inference with hf
            input_ids = input_ids.repeat(args.num_images_per_prompt, 1)
            attention_masks = attention_masks.repeat(args.num_images_per_prompt, 1)
            uncond_input_ids = uncond_input_ids.repeat(args.num_images_per_prompt, 1)
            uncond_attention_masks = uncond_attention_masks.repeat(args.num_images_per_prompt, 1)

            t1 = time.time()
            if args.sjd_sampling:
                output_ids = model.generate_visual_sjd(
                    input_ids,
                    tokenizer=tokenizer,
                    negative_prompt_ids=uncond_input_ids,
                    cfg_scale=args.cfg_scale,
                    temperature=args.temperature,
                    max_new_tokens=max_new_tokens,
                    use_cache=True
                )
            
            else:
                output_ids = model.generate_visual(
                    input_ids,
                    attention_mask=attention_masks,
                    negative_prompt_ids=uncond_input_ids,
                    negative_prompt_attention_mask=uncond_attention_masks,
                    cfg_scale=args.cfg_scale,
                    do_sample=True if args.temperature > 0 else False,
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

            output_ids_list = []
            t1 = time.time()
            for _ in range(args.num_images_per_prompt):
                with torch.inference_mode():
                    outs = model.generate(
                        input_dict,
                        sampling_params,
                        use_tqdm=False
                    )
                output_id_tensor = torch.tensor(outs[0].outputs[0].token_ids, dtype=input_ids.dtype, device=input_ids.device)
                output_ids_list.append(output_id_tensor)

            sampling_time = time.time() - t1
            llm_time += sampling_time
            output_ids = torch.stack(output_ids_list, dim=0)
            index_sample = output_ids.clone()    
        
        # VQGAN decoding
        index_sample = index_sample - len(tokenizer)
        index_sample = torch.clamp(index_sample, min=0, max=codebook_size-1)
        index_sample = index_sample.reshape(-1, latent_size, latent_size).unsqueeze(1)

        t2 = time.time()
        with torch.inference_mode():
            samples = vq_model.decode(index_sample)
        
        vq_time += (time.time() - t2)
        captions = tokenizer.batch_decode(input_ids.tolist())
        captions = [captions[0]] * args.num_images_per_prompt
        image_paths = [image_paths[0]] * args.num_images_per_prompt

        stacked_images = []
        for batch_i, (gen_img, caption, img_path) in enumerate(zip(samples, captions, image_paths)):
            if args.benchmark in ["coco", "mjhq"]:
                caption = caption.strip("<|t2i|>").strip("<|soi|>")
                base_img_path = os.path.basename(img_path)
                if not base_img_path.endswith(".png"):
                    base_img_path = base_img_path.split(".")[0] + ".png"
                category = os.path.basename(os.path.dirname(img_path))
                os.makedirs(os.path.join(save_dir, category), exist_ok=True)
                save_image(gen_img.squeeze(1).unsqueeze(0), os.path.join(save_dir, category, base_img_path), nrow=1, normalize=True, value_range=(-1, 1))
            
            elif args.benchmark == "geneval":
                caption = caption.strip("<|t2i|>").strip("<|soi|>")
                save_ind = "%05d" % (i * args.num_chunks + args.chunk_idx)
                save_folder = os.path.join(save_dir, str(save_ind), "samples")
                os.makedirs(save_folder, exist_ok=True)

                save_image(gen_img.squeeze(1).unsqueeze(0), os.path.join(save_folder, "%05d.png" % batch_i), nrow=1, normalize=True, value_range=(-1, 1))
                with open(os.path.join(os.path.dirname(save_folder), "metadata.jsonl"), "w") as f:
                    f.write(json.dumps(img_path))
            
            elif args.benchmark == "dpg":
                caption = caption.strip("<|t2i|>").strip("<|soi|>")
                stacked_images.append(gen_img.squeeze(1))

        if stacked_images:
            save_image_path = os.path.join(save_dir, image_paths[0]["file_name"])
            save_image(stacked_images, save_image_path, nrow=2, normalize=True, value_range=(-1, 1))
    

        i += 1
    
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
    model_name = get_model_name_from_path(model_path)

    llava_model_args = {
        "attn_implementation": "sdpa",
        # "multimodal": False,
    }

    if not args.vllm_serving:
        tokenizer, model, _, _  = load_pretrained_model(model_path, **llava_model_args)
    else:
        tokenizer, model = vllm_t2i(model_path=model_path)

    dataset = EvalT2IDataset(
        image_folder = args.data_dir, data_path = args.ann_path, tokenizer = tokenizer, image_size=args.image_size, benchmark = args.benchmark, num_chunks=args.num_chunks, chunk_idx=args.chunk_idx
    )
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, collate_fn=DataCollatorForT2IEvalDataset(tokenizer))
    save_dir = os.path.join(args.save_dir, args.benchmark, f"cfg{args.cfg_scale}_topp{args.top_p}_topk{args.top_k}_temp{args.temperature}")
    os.makedirs(save_dir, exist_ok=True)

    evaluate(model, vq_model, tokenizer, dataloader, save_dir, args)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./checkpoints/simpar_0.5B_rl")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--vq-model", type=str, default="llamagen")
    parser.add_argument("--vq-model-ckpt", type=str, default="./checkpoints/Cosmos-1.0-Tokenizer-DV8x16x16")
    parser.add_argument("--benchmark", type=str, default="mjhq")
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--ann_path", type=str)
    parser.add_argument("--vllm_serving", action="store_true", default=False)
    parser.add_argument("--sjd_sampling", action="store_true", default=False)
    parser.add_argument("--save_dir", type=str, default="visualize")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, choices=[256, 512, 768, 1024], default=1024)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-images-per-prompt", default=1, type=int)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)