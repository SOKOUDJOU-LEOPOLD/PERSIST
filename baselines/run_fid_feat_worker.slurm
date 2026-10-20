#!/bin/bash
# Generic Stage-1 (GPU feature-extraction) worker, same dynamic-claim pattern as
# run_eval_worker.slurm. DATASET_ROOT/GENERATED_VIDEO_DIR/OUTPUT_DIR/SPLIT_ARGS exported by the
# submitting script.
#SBATCH --job-name=fid-feat-worker
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

: "${DATASET_ROOT:?DATASET_ROOT must be exported by the submitting script}"
: "${GENERATED_VIDEO_DIR:?GENERATED_VIDEO_DIR must be exported by the submitting script}"
: "${OUTPUT_DIR:?OUTPUT_DIR must be exported by the submitting script}"
: "${SPLIT_ARGS:?SPLIT_ARGS must be exported by the submitting script}"

export PYTHONUNBUFFERED=1
export TORCH_HOME=/scratch/user/lsokoudj/.cache/torch
export VOXELWM_CACHE_DIR=/scratch/user/lsokoudj/.cache/voxelwm

source /scratch/user/lsokoudj/PERSIST/.venv/bin/activate
cd /scratch/user/lsokoudj/PERSIST/baselines

echo "Worker on $(hostname), GPU:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

python extract_per_frame_fid_features.py \
    $SPLIT_ARGS \
    --dataset-root "$DATASET_ROOT" \
    --generated-video-dir "$GENERATED_VIDEO_DIR" \
    --num-frames 600 \
    --output-dir "$OUTPUT_DIR" \
    --device cuda:0
