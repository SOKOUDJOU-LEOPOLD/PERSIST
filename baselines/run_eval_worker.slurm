#!/bin/bash
# Generic worker task, like run_persist256_worker.slurm but parameterized for ANY checkpoint /
# dataset / split (not hardcoded to persist-eval-sample). CKPT_PATH/DATASET_ROOT/OUTPUT_DIR/
# SPLIT_ARGS are exported by the submitting script; SPLIT_ARGS is either "--all-episodes" or
# "--split-path <file> --split <name>", passed through unquoted so it expands as multiple args.
#SBATCH --job-name=eval-worker
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

: "${CKPT_PATH:?CKPT_PATH must be exported by the submitting script}"
: "${DATASET_ROOT:?DATASET_ROOT must be exported by the submitting script}"
: "${OUTPUT_DIR:?OUTPUT_DIR must be exported by the submitting script}"
: "${SPLIT_ARGS:?SPLIT_ARGS must be exported by the submitting script}"

export PYTHONUNBUFFERED=1
export TORCH_HOME=/scratch/user/lsokoudj/.cache/torch
export VOXELWM_CACHE_DIR=/scratch/user/lsokoudj/.cache/voxelwm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /scratch/user/lsokoudj/PERSIST/.venv/bin/activate
cd /scratch/user/lsokoudj/PERSIST/baselines

echo "Worker on $(hostname), GPU:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

python eval_oasis_finetuned_fvd_shard.py \
    $SPLIT_ARGS \
    --dataset-root "$DATASET_ROOT" \
    --ckpt-path "$CKPT_PATH" \
    --use-oasis-native-action-layout \
    --frame-lengths 200 400 600 \
    --batch-size "${BATCH_SIZE:-4}" \
    --video-output-dir "$OUTPUT_DIR/videos" \
    --output-dir "$OUTPUT_DIR/episode_feats" \
    --device cuda:0
