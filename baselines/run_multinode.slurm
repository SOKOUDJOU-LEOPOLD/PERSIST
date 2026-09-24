#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G         # = 32G per GPU task (this cluster rejects --mem-per-gpu)
#SBATCH --mail-type=BEGIN,END,FAIL
# Multi-node, any-GPU-type Oasis fine-tuning: one Slurm task per GPU, placed on whatever nodes have
# free GPUs (A10/A30/A40/A100, mixed). Don't sbatch this directly -- use submit_multinode.sh, which
# sets --ntasks / --exclude / --time / --output and exports the RUN_* settings used below.

set -eo pipefail
REPO=/scratch/user/subarna_tamu.2026/PERSIST_LEOPOLD/PERSIST
EFFECTIVE_BATCH=16   # original runs: 4 GPUs x batch 4 -- kept fixed so any layout trains identically

export TORCH_HOME=/scratch/user/subarna_tamu.2026/.cache/torch
export HF_HOME=/scratch/user/subarna_tamu.2026/.hf
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ib0}

source $REPO/.venv/bin/activate
cd $REPO/baselines

echo "Job $SLURM_JOB_ID: $SLURM_NTASKS tasks on $SLURM_JOB_NODELIST (tasks/node: $SLURM_TASKS_PER_NODE)"
WORLD=$SLURM_NTASKS
if [ $((EFFECTIVE_BATCH % WORLD)) -ne 0 ]; then
    echo "ERROR: $WORLD GPUs can't split an effective batch of $EFFECTIVE_BATCH evenly -- use 4, 8 or 16"
    exit 1
fi

# Per-GPU batch sized for the SMALLEST GPU in the allocation (DDP ranks must all run the same batch).
# Measured with the resume-checkpoint fix: batch 1 ~22 GB, batch 4 fits a 48 GB A40.
GPU_INFO=$(srun --ntasks=$WORLD bash -c \
    'echo "$(nvidia-smi -i ${CUDA_VISIBLE_DEVICES%%,*} --query-gpu=memory.total,name --format=csv,noheader,nounits), $(hostname -s)"')
echo "GPUs (MiB, name, host):"; echo "$GPU_INFO" | sed 's/^/  /'
MIN_GIB=$(echo "$GPU_INFO" | awk -F',' '{print $1}' | sort -n | head -1 | awk '{print int($1/1024)}')
if   [ $MIN_GIB -ge 44 ]; then BATCH=4
elif [ $MIN_GIB -ge 38 ]; then BATCH=3
elif [ $MIN_GIB -ge 31 ]; then BATCH=2
else                           BATCH=1; fi
while [ $((EFFECTIVE_BATCH % (WORLD * BATCH))) -ne 0 ]; do BATCH=$((BATCH - 1)); done
ACCUM=$((EFFECTIVE_BATCH / (WORLD * BATCH)))
echo "Smallest GPU ${MIN_GIB} GiB -> batch $BATCH x accum $ACCUM x $WORLD GPUs = $EFFECTIVE_BATCH per optimizer step"

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

if [ -n "$SMOKE_STEPS" ]; then
    # Throughput / plumbing test: fresh model, no resume, no checkpoints, no wandb, scratch output.
    export WANDB_MODE=disabled
    RUN_ARGS=(--smoke-test-steps $SMOKE_STEPS --val-every $((SMOKE_STEPS / 2))
              --output-dir $REPO/outputs/multinode_smoke/$SLURM_JOB_ID)
else
    export WANDB_MODE=offline   # synced into the original cloud run by .scripts/wandb_sync_loop.sh
    export WANDB_DIR=$RUN_OUT_DIR
    # Newest checkpoint, unless it is smaller than the one before it (a save cut off by the time limit).
    if [ -z "$RESUME" ]; then
        mapfile -t CKPTS < <(ls -1 $RUN_OUT_DIR/step_*.pt | sort)
        RESUME=${CKPTS[-1]}
        if [ ${#CKPTS[@]} -ge 2 ] && [ $(stat -c %s "$RESUME") -lt $(stat -c %s "${CKPTS[-2]}") ]; then
            echo "WARNING: $RESUME looks truncated, falling back to ${CKPTS[-2]}"
            RESUME=${CKPTS[-2]}
        fi
    fi
    CKPT_WANDB_ID=$(python -c "import torch; print(torch.load('$RESUME', map_location='cpu', weights_only=False, mmap=True).get('wandb_run_id'))")
    echo "Resuming from $RESUME (wandb run id: $CKPT_WANDB_ID)"
    if [ "$CKPT_WANDB_ID" != "$RUN_WANDB_ID" ]; then
        echo "ERROR: expected wandb run id $RUN_WANDB_ID -- refusing to start a run that won't sync into the original"
        exit 1
    fi
    RUN_ARGS=(--resume-from $RESUME --output-dir $RUN_OUT_DIR --num-steps 100000
              --checkpoint-every 500 --val-every 100 --wandb-run-name $RUN_WANDB_NAME)
fi

# --gpu-bind=none: every task sees all of the job's GPUs on its node (and picks its own by
# SLURM_LOCALID in slurm_rank_env.sh) -- NCCL's intra-node P2P transport needs to see the peer GPU,
# and fails with "invalid device ordinal" when each task is isolated to its own GPU.
srun --kill-on-bad-exit=1 --cpu-bind=none --gpu-bind=none bash slurm_rank_env.sh \
    --split-path $RUN_SPLIT \
    --dataset-root $RUN_DATASET_ROOT \
    --use-oasis-native-action-layout \
    --informed-action-init \
    --batch-size $BATCH --grad-accumulation-steps $ACCUM \
    --lr-pretrained 1e-6 --lr-new-layer 1e-4 \
    "${RUN_ARGS[@]}"
