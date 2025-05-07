PROMPT_VERSION="qwen_1_5"
LLM_VERSION="/openseg_blob/v-yuweiyang/SimpleAR/ckpt/SimpleAR-1.5B-SFT"
# for sft, use the path of pretrained model, model_path: output_ckpt/xxx, not the Qwen2.5-1.5B-Instruct or SimpleAR-1.5B-SFT
# offical simpleAR do not offer the pretrained model

IFS=',' read -ra ALL_PORTS <<< $METIS_WORKER_0_PORT
export NCCL_SOCKET_IFNAME=eth0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_API_KEY=c6358a6373cf18e79b191e0944e44383f50ba244  # should be changed to your own wandb api_key

# use token for generation, so gen_image_folder is None; <im_start>, <im_end> or <im_patch> is False;
# global batch_size is expanded to 2 * batch_size; gradient_accumulation_steps = 2;
# --sdpa: Scaled Dot Product Attention, more faster
ACCELERATE_CPU_AFFINITY=1 \
 torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    simpar/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --gen_data_path /openseg_blob/v-yuweiyang/SimpleAR/datasets/visual_tokens/code2chart_sft_1024_163189_meta.json \
    --gen_image_folder "" \
    --token_dataset True \
    --sample_short True \
    --mm_tunable_parts="mm_language_model" \
    --p_drop_cond 0.1 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name sft_code2chart \
    --output_dir /openseg_blob/v-yuweiyang/SimpleAR_Code2Chart/output_ckpt/code2chart_160k_for_sft/ \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.0 \
    --lr_scheduler_type "constant" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 5120 \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 32 \
    --report_to wandb \
    --attn_implementation sdpa