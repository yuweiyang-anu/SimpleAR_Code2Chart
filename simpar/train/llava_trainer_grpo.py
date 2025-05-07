# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
# sys.path.append("/openseg_blob/wxc/SimpleAR")

import logging
import wandb
from typing import Union, Any, Optional
from dataclasses import dataclass, field
from PIL import Image

import torch
import torch.nn as nn
import transformers
import datasets
import transformers
print(transformers.__file__)
from transformers import set_seed
from transformers import AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from accelerate.utils import broadcast_object_list, gather, gather_object, set_seed
import open_clip
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
from open_clip import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
import trl
print(trl.__file__)
import sys
print(sys.path)
print("debug==============")
from trl import GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from trl.trainer.utils import pad
from trl.models import unwrap_model_for_generation

from simpar.model.tokenizer.cosmos_tokenizer.networks import TokenizerConfigs
from simpar.model.tokenizer.cosmos_tokenizer.video_lib import CausalVideoTokenizer as CosmosTokenizer
from simpar.train.t2i_data import GRPOT2IDataset
from simpar.grpo.configs import GRPOConfig
from simpar.grpo.utils.callbacks import get_callbacks
from simpar.grpo.utils.wandb_logging import init_wandb_training
from simpar.grpo.rewards import (
    accuracy_reward,
    code_reward,
    format_reward,
    get_code_format_reward,
    get_cosine_scaled_reward,
    get_repetition_penalty_reward,
    len_reward,
    reasoning_steps_reward,
    tag_count_reward,
    clip_reward,
    aesthetic_reward,
    hps_reward,
)

# 设置 Hugging Face 缓存路径
# os.environ["HF_HOME"] = "/openseg_blob/wxc/huggingface_cache"
import pickle
import torch
import torch.distributed as dist


logger = logging.getLogger(__name__)
def broadcast_pickle_object(obj, src=0, device="cuda:0"):
    """
    将对象 pickle 后，广播出去，再反序列化还原。
    """
    if dist.get_rank() == src:
        # Rank0 序列化对象
        serialized_obj = pickle.dumps(obj)
        tensor = torch.ByteTensor(list(serialized_obj)).to(device)
        size_tensor = torch.tensor([tensor.numel()], device=device)
    else:
        tensor = None
        size_tensor = torch.tensor([0], device=device)
    
    # 广播张量大小
    dist.broadcast(size_tensor, src=src)
    size = size_tensor.item()

    if dist.get_rank() != src:
        # 非源进程根据 size 分配接收缓存
        tensor = torch.empty(size, dtype=torch.uint8, device=device)
    
    # 广播整个 ByteTensor
    dist.broadcast(tensor, src=src)
    
    # 非源进程反序列化对象
    serialized_obj = bytes(tensor.tolist())
    return pickle.loads(serialized_obj)

    


class LLaVAGRPOTrainer(GRPOTrainer):

    def _decode_images(self, completion_ids):
        device = self.accelerator.device
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
        completion_ids = torch.stack(completion_ids, dim=0)
        
        codebook_size = 64000
        latent_size = 1024 // 16
        index_samples = completion_ids - len(self.processing_class)  #为什么减掉151672
        
        index_samples = torch.clamp(index_samples, min=0, max=codebook_size-1)
        index_samples = index_samples.reshape(-1, latent_size, latent_size).unsqueeze(1)

        with torch.inference_mode():
            generated_images = self.vq_model.decode(index_samples).squeeze(2)
        
        # resize to 224 to save memory
        generated_images = torch.nn.functional.interpolate(generated_images, size=(224, 224), mode="bilinear", align_corners=False)
        generated_images = (255 * (generated_images * 0.5 + 0.5)).clamp(0, 255)
        
        mean = torch.tensor(OPENAI_DATASET_MEAN, device=device)
        std = torch.tensor(OPENAI_DATASET_STD, device=device)

        transformed_images = generated_images / 255.0 # B, 3, 224, 224
        transformed_images = (transformed_images - mean[None, :, None, None]) / std[None, :, None, None]

        with torch.inference_mode():
            image_features = self.clip_model.encode_image(transformed_images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        # convert to list for broadcast
        transformed_images = [img.cpu() for img in transformed_images]
        image_features = [feat.cpu() for feat in image_features]
        
        return transformed_images, image_features

    
    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [p for p in prompts]
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        prompt_ids = prompt_ids.to(device)
        prompt_mask = prompt_mask.to(device)
        
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate completions 使用 vLLM 或常规生成
        if self.args.use_vllm:
            # 如果需要，加载主进程的模型权重
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # 通过 gather_object 得到所有进程的 prompts
            all_prompts_text = gather_object(prompts_text)

            if self.accelerator.is_main_process:
                # 去重缓解重复计算，并保证至少有7个 prompt
                ordered_set_of_prompts = list(dict.fromkeys(all_prompts_text))
                if len(ordered_set_of_prompts) < 7:
                    ordered_set_of_prompts += ordered_set_of_prompts[:7 - len(ordered_set_of_prompts)]

                # 调用 vLLM 生成结果
                all_outputs = self.llm.generate(
                    ordered_set_of_prompts, sampling_params=self.sampling_params, use_tqdm=False
                )
                completion_ids = []
                for outputs in all_outputs:
                    for output in outputs.outputs:
                        completion_ids.append(output.token_ids)
                
                decoded_images, decoded_image_embeds = self._decode_images(completion_ids)
            else:
                # 非主进程暂时传递占位数据，等待主进程广播数据
                completion_ids = [None] * len(all_prompts_text)
                decoded_images = [None] * len(all_prompts_text)
                decoded_image_embeds = [None] * len(all_prompts_text)

            ############################################################################
            # 以下部分为 debug 增加信息，帮助定位 broadcast 函数调用时的数据问题
            import pickle
            import torch.distributed

            # --- 调试广播前对 completion_ids 序列化情况
            try:
                serialized_completion_ids = pickle.dumps(completion_ids)
                print(
                    f"[DEBUG] Rank {torch.distributed.get_rank()} - Before broadcast - "
                    f"completion_ids serialized length: {len(serialized_completion_ids)}"
                )
            except Exception as err:
                print(f"[DEBUG] Rank {torch.distributed.get_rank()} - Error while serializing completion_ids: {err}")

            # --- 确保各个进程同步到这里（防止部分进程未到达广播而导致数据不一致）
            torch.distributed.barrier()

            # --- 对 completion_ids 进行广播，并捕获异常
            try:
                completion_ids = broadcast_pickle_object(completion_ids, src=0, device=device)
                print(
                    f"[DEBUG] Rank {torch.distributed.get_rank()} - After broadcast - "
                    f"completion_ids list length: {len(completion_ids)}"
                )
            except Exception as err:
                print(f"[DEBUG] Rank {torch.distributed.get_rank()} - Broadcasting completion_ids failed: {err}")
                raise

            # 对 decoded_images 进行同样的调试与广播
            try:
                serialized_decoded_images = pickle.dumps(decoded_images)
                print(
                    f"[DEBUG] Rank {torch.distributed.get_rank()} - Before broadcast - "
                    f"decoded_images serialized length: {len(serialized_decoded_images)}"
                )
            except Exception as err:
                print(f"[DEBUG] Rank {torch.distributed.get_rank()} - Error while serializing decoded_images: {err}")

            torch.distributed.barrier()

            try:
                decoded_images =  broadcast_pickle_object(decoded_images, src=0, device=device)
                print(
                    f"[DEBUG] Rank {torch.distributed.get_rank()} - After broadcast - "
                    f"decoded_images list length: {len(decoded_images)}"
                )
            except Exception as err:
                print(f"[DEBUG] Rank {torch.distributed.get_rank()} - Broadcasting decoded_images failed: {err}")
                raise

            # 对 decoded_image_embeds 进行同样的调试与广播
            try:
                serialized_decoded_image_embeds = pickle.dumps(decoded_image_embeds)
                print(
                    f"[DEBUG] Rank {torch.distributed.get_rank()} - Before broadcast - "
                    f"decoded_image_embeds serialized length: {len(serialized_decoded_image_embeds)}"
                )
            except Exception as err:
                print(f"[DEBUG] Rank {torch.distributed.get_rank()} - Error while serializing decoded_image_embeds: {err}")

            torch.distributed.barrier()

            try:
                decoded_image_embeds =  broadcast_pickle_object(decoded_image_embeds, src=0, device=device)
                print(
                    f"[DEBUG] Rank {torch.distributed.get_rank()} - After broadcast - "
                    f"decoded_image_embeds list length: {len(decoded_image_embeds)}"
                )
            except Exception as err:
                print(f"[DEBUG] Rank {torch.distributed.get_rank()} - Broadcasting decoded_image_embeds failed: {err}")
                raise
            ############################################################################

            # 每个进程只取对应的 slice
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids = completion_ids[process_slice]
            decoded_images = decoded_images[process_slice]
            decoded_image_embeds = decoded_image_embeds[process_slice]

            # 将生成序列转换为tensor并 pad 后再与 prompt_ids 拼接
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else: #非vllm的都没实现，这里有问题
            # 常规生成路径
            with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(
                    prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                )
            # 根据 prompt 长度拆分生成的 token
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # 以下是对生成内容的后处理、计算 mask、logits 以及评分，简略说明：
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # 拼接 prompt_mask 与 completion_mask 供后续计算使用
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        logits_to_keep = completion_ids.size(1)  # 只计算 completion 部分的 logits
        print(f"self.num_iterations::{self.num_iterations}")
        with torch.inference_mode():
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        # 以下部分是根据 image_embed 以及处理后的 prompt 调用 CLIP 模型生成 feature，
        # 并计算多个 reward 的加权和，后续再分别对各个指标进行统计和记录。 
        completions = []
        for i, (prompt, image_embed) in enumerate(zip(prompts, decoded_image_embeds)):
            prompt_clean = prompt.strip("<|t2i|>").strip("<|soi|>")
            with torch.inference_mode():
                text = self.clip_tokenizer(prompt_clean.strip()).to(device) #hspv-clip 加一些特殊符号以及截断变成token了
                text_feature = self.clip_model.encode_text(text)  #转到空间向量
                text_feature /= text_feature.norm(dim=-1, keepdim=True)

            image_feature = image_embed.unsqueeze(0).to(device)
            completions.append(
                [{
                    "image_feature": image_feature,
                    "text_feature": text_feature,
                }]
            )

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):  
            output_reward_func = reward_func(prompts=prompts, completions=completions) #用image_feature 和text_feature算一个相似度
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Gather rewards：group 内归一化时 completions 可能分布在不同进程
        rewards_per_func = gather(rewards_per_func)

        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        # 计算组内奖励（按 num_generations 分组） 对同一 prompt 下多个生成的候选答案进行归一化
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # 归一化奖励以便计算 advantage 归一化可以使得优势值（advantages）在不同的组之间具有可比性，同时减小不同组 reward 分布的差异对训练的影响，使训练过程更加稳定
        #这里重复是因为reward是(num_prompts * self.num_generations,)维度的
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        #整个 batch 包含多个 prompts ，所以要取出对应的prompt
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # 记录各项指标数据
        mode = "eval" if self.control.should_evaluate else "train"
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)

        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            reward_func_name = reward_func.__name__
            self._metrics[mode][f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }
    


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format', 'format_deepseek', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', tag_count', 'code', 'code_format'.
        cosine_min_value_wrong (`float`):
            Minimum reward for cosine scaling for wrong answers.
        cosine_max_value_wrong (`float`):
            Maximum reward for cosine scaling for wrong answers.
        cosine_min_value_correct (`float`):
            Minimum reward for cosine scaling for correct answers.
        cosine_max_value_correct (`float`):
            Maximum reward for cosine scaling for correct answers.
        cosine_max_len (`int`):
            Maximum length for cosine scaling.
        code_language (`str`):
            Language for code format reward.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "tag_count"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'format_deepseek', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', tag_count', 'code', 'code_format'"
        },
    )
    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={"help": "Minimum reward for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Maximum reward for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={"help": "Minimum reward for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for scaling"},
    )
    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={"help": "Maximum (negative) penalty for for repetition penalty reward"},
    )
    code_language: str = field(
        default="python",
        metadata={
            "help": "Language for code format reward. Based on E2B supported languages https://e2b.dev/docs/code-interpreting/supported-languages",
            "choices": ["python", "javascript", "r", "java", "bash"],
        },
    )

    data_path: str = field(
        default="",
        metadata={"help": "Path to the generated data"},
    )

    vq_model_ckpt: str = field(
        default="/openseg_blob/wxc/SimpleAR/ckpt/cosmos"
    )

    clip_model_ckpt: str = field(
        default="/openseg_blob/wxc/SimpleAR/ckpt/CLIP-ViT-H-14-laion2B-s32B-b79K"
    )
    aest_model_ckpt: str = field(
        default="/openseg_blob/wxc/SimpleAR/ckpt/aesthetic-preditor/HPS_v2_compressed.pt"
    )



def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    ################
    # Load tokenizer
    ################
    print(f"AutoTokenizer==model_args.model_name_or_path::{model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=False)
    
    # Load VQ model
    tokenizer_config = TokenizerConfigs["DV"].value
    tokenizer_config.update(dict(spatial_compression=16, temporal_compression=8))
    vq_model = CosmosTokenizer(checkpoint_enc=f"{script_args.vq_model_ckpt}/encoder.jit", checkpoint_dec=f"{script_args.vq_model_ckpt}/decoder.jit", tokenizer_config=tokenizer_config)
    vq_model.eval()
    vq_model.requires_grad_(False)

    # Load reward model
    print(f"script_args.clip_model_ckpt{script_args.clip_model_ckpt}")
    clip_model, _, clip_preprocess = create_model_and_transforms(
        'ViT-H-14',
        f'{script_args.clip_model_ckpt}/open_clip_pytorch_model.bin',
        precision='amp',
        device="cuda",
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False
    )
    clip_tokenizer = get_tokenizer('ViT-H-14')
    clip_model = clip_model.to("cuda")
    clip_model.eval()

    # Load the dataset
    dataset = GRPOT2IDataset(data_path=script_args.data_path, tokenizer=tokenizer)

    # Get reward functions
    REWARD_FUNCS_REGISTRY = {
        "accuracy": accuracy_reward,
        "format": format_reward,
        "reasoning_steps": reasoning_steps_reward,
        "cosine": get_cosine_scaled_reward(
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward(
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
        "length": len_reward,
        "code": code_reward,
        "code_format": get_code_format_reward(language=script_args.code_language),
        "tag_count": tag_count_reward,
        "clip": clip_reward,
        "aesthetic": aesthetic_reward,
        "hps": hps_reward
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    training_args.model_init_kwargs = model_kwargs
    print("###############################")
    print(training_args)
    print("###############################")
    #############################
    # Initialize the GRPO trainer
    #############################
    trainer = LLaVAGRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        processing_class=tokenizer,
    )

    trainer.vq_model = vq_model

    trainer.clip_preprocess = clip_preprocess
    trainer.clip_tokenizer = clip_tokenizer
    trainer.clip_model = clip_model
    
    # trainer.aesthetic_model = aest_model

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    # metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
