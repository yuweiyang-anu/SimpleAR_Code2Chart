torchrun \
 --nnodes=1 --nproc_per_node=4 --master_port 2328 \
 /openseg_blob/v-yuweiyang/SimpleAR/simpar/data/extract_token.py \
    --dataset_type "image" \
    --dataset_name "code2chart_debug_only" \
    --code_path /openseg_blob/v-yuweiyang/SimpleAR/datasets/visual_tokens \
    --gen_data_path /openseg_blob/v-yuweiyang/SimpleAR/datasets/code2chart_converted_10k_debug.json \
    --gen_image_folder "" \
    --gen_resolution 512
