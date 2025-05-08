cd /openseg_blob/v-yuweiyang/SimpleAR_Code2Chart
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
accelerate launch --main_process_port 1234  --num_processes 3 --config_file simpar/configs/accelerate_configs/zero3.yaml \
   simpar/train/llava_trainer_grpo.py \
    --config simpar/configs/config_grpo.yaml \
    --data_path /openseg_blob/v-yuweiyang/SimpleAR/datasets/code2chart_converted_160k.json
