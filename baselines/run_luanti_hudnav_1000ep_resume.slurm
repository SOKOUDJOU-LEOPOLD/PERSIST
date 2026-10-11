#!/bin/bash
#SBATCH --job-name=oasis-luanti-hudnav-1000ep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:4
#SBATCH --nodelist=fc098
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=../outputs/finetune_oasis_luanti_hudnav_1000ep/slurm_%j.log
# Submit from baselines/ (the --output path and all relative paths below assume it):
#   cd baselines && sbatch run_luanti_hudnav_1000ep_resume.slurm
# Resumes from the newest complete step_*.pt in OUT_DIR, so the same file can be resubmitted after
# every 10h timeout. Override with: sbatch --export=ALL,RESUME=../outputs/.../step_XXXXXX.pt ...

REPO=/scratch/user/subarna_tamu.2026/PERSIST_LEOPOLD/PERSIST
OUT_DIR=$REPO/outputs/finetune_oasis_luanti_hudnav_1000ep
EXPECTED_WANDB_ID=x3yhdgnt

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

# Offline: compute nodes log locally with the checkpoint's run id; .scripts/wandb_sync_loop.sh on a
# login node pushes them into the original cloud run with `wandb sync --append`.
export WANDB_MODE=offline
export WANDB_DIR=$OUT_DIR
export TORCH_HOME=/scratch/user/subarna_tamu.2026/.cache/torch
export PYTHONUNBUFFERED=1  # flush print() immediately so the slurm log and the wandb Logs tab stay current
export HF_HOME=/scratch/user/subarna_tamu.2026/.hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source $REPO/.venv/bin/activate
cd $REPO/baselines

# Newest checkpoint, unless it is smaller than the one before it (a save cut off by the time limit).
if [ -z "$RESUME" ]; then
    mapfile -t CKPTS < <(ls -1 $OUT_DIR/step_*.pt | sort)
    RESUME=${CKPTS[-1]}
    if [ ${#CKPTS[@]} -ge 2 ] && [ $(stat -c %s "$RESUME") -lt $(stat -c %s "${CKPTS[-2]}") ]; then
        echo "WARNING: $RESUME looks truncated, falling back to ${CKPTS[-2]}"
        RESUME=${CKPTS[-2]}
    fi
fi
CKPT_WANDB_ID=$(python -c "import torch; print(torch.load('$RESUME', map_location='cpu', weights_only=False, mmap=True).get('wandb_run_id'))")
echo "Resuming from $RESUME (wandb run id: $CKPT_WANDB_ID)"
if [ "$CKPT_WANDB_ID" != "$EXPECTED_WANDB_ID" ]; then
    echo "ERROR: expected wandb run id $EXPECTED_WANDB_ID -- refusing to start a run that won't sync into the original"
    exit 1
fi

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes 4 finetune_oasis.py \
    --split-path ../outputs/oasis_luanti_hudnav_1000ep.json \
    --dataset-root $REPO/data/luanti/extracted/OpenWorldCreativeHud-v0 \
    --output-dir ../outputs/finetune_oasis_luanti_hudnav_1000ep \
    --resume-from $RESUME \
    --use-oasis-native-action-layout \
    --informed-action-init \
    --batch-size 3 \
    --lr-pretrained 1e-6 --lr-new-layer 1e-4 \
    --num-steps 100000 --checkpoint-every 500 --val-every 100 \
    --wandb-run-name oasis-luanti-hudnav-1000ep
