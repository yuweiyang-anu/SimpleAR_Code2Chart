# CKPT_PATH=simpar_0.5B_rl # Overall score (avg. over tasks): 0.59327
CKPT_PATH=simpar_1.5B_rl # Overall score (avg. over tasks): 0.62568
SAVE_FOLDER=$CKPT_PATH
CFG_SCALE=6.0

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m simpar.eval.model_t2i \
    --model-path ./checkpoints/${CKPT_PATH} \
    --save_dir ./visualize/${CKPT_PATH} \
    --ann_path "./eval/geneval/prompts/evaluation_metadata.jsonl" \
    --vq-model "cosmos" \
    --vq-model-ckpt "./checkpoints/Cosmos-1.0-Tokenizer-DV8x16x16" \
    --image-size 1024 \
    --batch-size 1 \
    --temperature 1.0 \
    --top_k 64000 \
    --top_p 1.0 \
    --benchmark "geneval" \
    --num-images-per-prompt 4 \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --cfg-scale $CFG_SCALE &
done
wait