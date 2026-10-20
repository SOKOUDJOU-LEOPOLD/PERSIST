#!/bin/bash
# Single-node, non-Slurm, online-wandb counterpart to train_full29k.slurm -- for csce-chzhang
# (4x H100 NVL, direct internet access, no Slurm scheduler). Same training args/defaults as the
# Slurm version; only the launch/scheduling wrapper differs.
#
# Fine-tune one PERSIST component on datasets/luanti_full29k from the official PERSIST-team weights,
# data-parallel over all local GPUs. Adapted from train_full29k.slurm for a single non-SLURM machine
# (4x H100 NVL, direct internet access) -- same $COMMON/$ONTHEFLY/per-component args, no SLURM/srun/
# multi-node wrapper, no Anaconda libstdc++ preload (this venv isn't Anaconda-based).
#
# Usage:
#   ./dataset_toolkits/manifests/full29k/train_full29k_local.sh voxel
#   ./dataset_toolkits/manifests/full29k/train_full29k_local.sh pixel
#   ./dataset_toolkits/manifests/full29k/train_full29k_local.sh camera
#
# Resubmitting (re-running) with the same component resumes from the newest checkpoint_<step>
# (same --run-name, --resume auto) -- see trainers/__init__.py's check_existing_ckpt.

set -euo pipefail
cd "$(dirname "$0")/../../.."
source .venv/bin/activate
export PYTHONPATH=.
# direct internet access on this machine -- log wandb online, no sync script needed
export WANDB_MODE=${WANDB_MODE:-online}

COMPONENT=$1
GPUS=$(nvidia-smi -L | wc -l)
WORKERS=${WORKERS:-16}

P=checkpoints/pretrained
OUT=${OUT_DIR:-outputs/runs_full29k}  # env overrides (OUT_DIR, DATASET, TOTAL_STEPS, I_VAL, I_SAVE) are for short tests only
mkdir -p "$OUT"
# total_steps is a multiple of i_save so the final step is checkpointed
COMMON="--dataset-path ${DATASET:-datasets/luanti_full29k} --batch-size 2 --total-steps ${TOTAL_STEPS:-20000}
  --lr-warmup-steps 1000 --i-log 100 --i-val ${I_VAL:-500} --i-print 100 --i-save ${I_SAVE:-1000}
  --dataloader-workers $WORKERS --val-dataloader-workers 2
  --full-pass-validation --checkpoint-retention latest_and_best --dataloader-start-method forkserver --no-epoch-boundaries
  --use-wandb --wandb-project persist-luanti-full29k"
ONTHEFLY="--encode-latents-on-the-fly --voxel-encoder-checkpoint $P/voxel-encoder/model.safetensors
  --pixel-vae-checkpoint $P/pixel-vae/model.safetensors"

case "$COMPONENT" in
    voxel)
        SCRIPT=train_scripts/train_diffuser_voxel.py
        ARGS="$COMMON $ONTHEFLY --run-name full29k-voxel-denoiser-xl --output-dir $OUT/voxel_diffuser
          --wandb-run-name voxel-denoiser-xl
          --pretrained-checkpoint $P/voxel-denoiser-xl/model.safetensors --pretrained-checkpoint-model-key denoiser
          --denoiser.patch-size 1 ${GRAD_CKPT:+--denoiser.gradient-checkpointing} diffusion:diffusion-forcing" ;;
    pixel)
        SCRIPT=train_scripts/train_diffuser_pixel.py
        ARGS="$COMMON $ONTHEFLY --run-name full29k-pixel-denoiser --output-dir $OUT/pixel_diffuser
          --wandb-run-name pixel-denoiser
          --voxel-decoder-checkpoint $P/voxel-decoder/model.safetensors --voxel-decoder-batch-size 2
          --pretrained-checkpoint $P/pixel-denoiser/model.safetensors --pretrained-checkpoint-model-key denoiser
          --pretrained-checkpoint-extra $P/voxel-class-embedder/model.safetensors
          --pretrained-checkpoint-extra-model-key voxel_class_embedder
          ${GRAD_CKPT:+--denoiser.gradient-checkpointing} diffusion:diffusion-forcing" ;;
    camera)
        SCRIPT=train_scripts/train_transformer_camera.py
        # the camera script requires i_log % i_val == 0
        ARGS="${COMMON/--i-log 100/--i-log 500} --run-name full29k-camera-model --output-dir $OUT/camera_transformer
          --wandb-run-name camera-model
          --pretrained-checkpoint $P/camera-model/model.safetensors --pretrained-checkpoint-model-key camera
          --model.cam-rotation-output-rep delta_angle --model.cam-fov-output-rep delta" ;;
    *) echo "unknown component $COMPONENT"; exit 1 ;;
esac

echo "=== $COMPONENT: $GPUS GPU(s), $(date) ==="

accelerate launch --mixed_precision bf16 --dynamo_backend no --num_processes "$GPUS" \
    $( [ "$GPUS" -gt 1 ] && echo --multi_gpu ) \
    "$SCRIPT" $(echo "$ARGS")
echo "=== $COMPONENT finished $(date) ==="
