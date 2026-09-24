#!/bin/bash
# Submit an Oasis fine-tuning run across any free GPUs (multiple nodes / GPU types) via
# run_multinode.slurm.
#
#   ./submit_multinode.sh <s1nav|hudnav> [any|fast] [N_GPUS] [SMOKE_STEPS]
#
#   any  -- every GPU type except T4 (16 GB, too small). Most GPUs, but a synchronized job runs at
#           the pace of its slowest GPU, so A100s paired with A10s wait on the A10s.
#   fast -- A40/A100 only.
#   N_GPUS must be 4, 8 or 16 (default 8). SMOKE_STEPS runs a short no-resume/no-wandb test.
# Examples:
#   ./submit_multinode.sh s1nav any 4 20     # 20-step smoke test on 4 GPUs
#   ./submit_multinode.sh s1nav any 8        # resume s1nav on 8 GPUs of any type
set -e
cd "$(dirname "$0")"
REPO=/scratch/user/subarna_tamu.2026/PERSIST_LEOPOLD/PERSIST
RUN=${1:?usage: $0 <s1nav|hudnav> [any|fast] [N_GPUS] [SMOKE_STEPS]}
POOL=${2:-any}
N_GPUS=${3:-8}
SMOKE_STEPS=$4

case $RUN in
    s1nav)
        RUN_SPLIT=../outputs/oasis_luanti_1000ep.json
        RUN_DATASET_ROOT=$REPO/data/luanti/extracted/OpenWorldCreative-v0
        RUN_OUT_DIR=$REPO/outputs/finetune_oasis_luanti_1000ep
        RUN_WANDB_ID=03fgmp9b
        RUN_WANDB_NAME=oasis-luanti-1000ep ;;
    hudnav)
        RUN_SPLIT=../outputs/oasis_luanti_hudnav_1000ep.json
        RUN_DATASET_ROOT=$REPO/data/luanti/extracted/OpenWorldCreativeHud-v0
        RUN_OUT_DIR=$REPO/outputs/finetune_oasis_luanti_hudnav_1000ep
        RUN_WANDB_ID=x3yhdgnt
        RUN_WANDB_NAME=oasis-luanti-hudnav-1000ep ;;
    *) echo "unknown run: $RUN"; exit 1 ;;
esac

# Node lists are specific to this cluster's gpu partition (sinfo -p gpu -N -o "%N %G").
T4_NODES=fc003,fc023,fc101,fc102,fc103,fc104
SLOW_NODES=fc099,fc122,fc123,fc124   # A10 / A30 (24 GB)
case $POOL in
    any)  EXCLUDE=$T4_NODES ;;
    fast) EXCLUDE=$T4_NODES,$SLOW_NODES ;;
    *) echo "unknown pool: $POOL (any|fast)"; exit 1 ;;
esac

if [ -n "$SMOKE_STEPS" ]; then
    mkdir -p $REPO/outputs/multinode_smoke
    LOG=$REPO/outputs/multinode_smoke/${RUN}_%j.log
    TIME=00:40:00
    EXTRA_ENV=,SMOKE_STEPS=$SMOKE_STEPS,NCCL_DEBUG=INFO
else
    LOG=$RUN_OUT_DIR/slurm_%j.log
    TIME=10:00:00
    EXTRA_ENV=
fi

sbatch --job-name=oasis-$RUN-mn --ntasks=$N_GPUS --exclude=$EXCLUDE --time=$TIME --output=$LOG \
    --export=ALL,RUN_SPLIT=$RUN_SPLIT,RUN_DATASET_ROOT=$RUN_DATASET_ROOT,RUN_OUT_DIR=$RUN_OUT_DIR,RUN_WANDB_ID=$RUN_WANDB_ID,RUN_WANDB_NAME=$RUN_WANDB_NAME$EXTRA_ENV \
    run_multinode.slurm
