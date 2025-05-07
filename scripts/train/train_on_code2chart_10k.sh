PROMPT_VERSION="qwen_1_5"
LLM_VERSION="/openseg_blob/v-yuweiyang/SimpleAR/ckpt/Qwen2.5-0.5B-Instruct" # for pretrain, use the /Qwen2.5-0.5B-Instruct, SimpleAR-1.5B-RL only use for debug

IFS=',' read -ra ALL_PORTS <<< $METIS_WORKER_0_PORT
export NCCL_SOCKET_IFNAME=eth0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_API_KEY=c6358a6373cf18e79b191e0944e44383f50ba244

# use token for generation, so gen_image_folder is None; <im_start>, <im_end> or <im_patch> is False;
# global batch_size is expanded to 2 * batch_size; gradient_accumulation_steps = 2;
# sdpa: Scaled Dot Product Attention, more faster
ACCELERATE_CPU_AFFINITY=1 \
 torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    simpar/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --gen_data_path /openseg_blob/v-yuweiyang/SimpleAR/datasets/visual_tokens/code2chart_10k_debug_1024_10000_meta.json \
    --gen_image_folder "" \
    --token_dataset True \
    --sample_short True \
    --mm_tunable_parts="mm_language_model" \
    --p_drop_cond 0.1 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name test \
    --output_dir /openseg_blob/v-yuweiyang/SimpleAR/output_ckpt/code2chart_10k/ \
    --num_train_epochs 8 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.0 \
    --lr_scheduler_type "constant" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 32 \
    --report_to wandb \
    --attn_implementation sdpa