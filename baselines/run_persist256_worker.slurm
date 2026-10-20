#!/bin/bash
# Generic worker task -- one array task = one worker process claiming episodes from the shared
# work queue (see eval_oasis_finetuned_fvd_shard.py's module docstring). GPU_TYPE/GPU_COUNT/
# TIME_LIMIT are set by the #SBATCH directives at SUBMIT time via sbatch --gres/--time overrides
# in submit_persist256_eval.sh (one sbatch call per GPU-type pool, this same script reused for
# all three) -- CKPT_PATH/DATASET_ROOT/OUTPUT_DIR are exported by that script.
#SBATCH --job-name=persist256-worker
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=../outputs/finetune_oasis_luanti_hudnav_1000ep/eval_persist256/slurm_worker_%A_%a_%j.log

: "${CKPT_PATH:?CKPT_PATH must be exported by the submitting script}"
: "${DATASET_ROOT:?DATASET_ROOT must be exported by the submitting script}"
: "${OUTPUT_DIR:?OUTPUT_DIR must be exported by the submitting script}"

export PYTHONUNBUFFERED=1
export TORCH_HOME=/scratch/user/lsokoudj/.cache/torch
export VOXELWM_CACHE_DIR=/scratch/user/lsokoudj/.cache/voxelwm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /scratch/user/lsokoudj/PERSIST/.venv/bin/activate
cd /scratch/user/lsokoudj/PERSIST/baselines

echo "Worker on $(hostname), GPU:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

# --batch-size is lower for non-A100 GPUs to reduce OOM risk (set via BATCH_SIZE env, default 4);
# a worker that finds the queue already empty (another pool got there first) exits almost
# immediately, so slightly over-provisioning worker COUNT across pools is cheap.
python eval_oasis_finetuned_fvd_shard.py \
    --all-episodes \
    --dataset-root "$DATASET_ROOT" \
    --ckpt-path "$CKPT_PATH" \
    --use-oasis-native-action-layout \
    --frame-lengths 200 400 600 \
    --batch-size "${BATCH_SIZE:-4}" \
    --video-output-dir "$OUTPUT_DIR/videos" \
    --output-dir "$OUTPUT_DIR/episode_feats" \
    --device cuda:0
