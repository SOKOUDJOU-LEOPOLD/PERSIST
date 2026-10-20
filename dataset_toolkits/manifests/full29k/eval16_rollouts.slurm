#!/bin/bash
# Multi-episode evaluation of voxel-denoiser recipes: 300-frame rollouts on 16 held-out validation episodes
# (eval16_sha256.txt: 8 s1nav + 8 hudnav), real first frame, the model's own camera. Every config uses the
# pretrained pixel denoiser + camera model; only the voxel denoiser changes.
# Usage: sbatch --job-name=eval16-<config> eval16_rollouts.slurm <config>
#   config: pretrained | ft20k | recipe0 | recipe1 | recipe2 | lora-<tag> (merged weights in inference_weights/voxel_<config>.safetensors)
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/user/lsokoudj/PERSIST/outputs/runs_full29k/logs/%x_%j.log

set -uo pipefail
cd /scratch/user/lsokoudj/PERSIST
module load GCC/13.2.0 CUDA/12.6.0
source .venv/bin/activate
export LD_PRELOAD=$EBROOTGCCCORE/lib64/libstdc++.so.6
export PYTHONPATH=.

CONFIG=$1
P=checkpoints/pretrained
R=outputs/runs_full29k
case "$CONFIG" in
    pretrained) VOXEL=$P/voxel-denoiser-xl/model.safetensors ;;
    ft20k)      VOXEL=$R/voxel_diffuser/full29k-voxel-denoiser-xl-uid-b6382043/checkpoints/checkpoint_20000/model.safetensors ;;
    recipe*)    VOXEL=$(ls -d $R/voxel_diffuser/full29k-voxel-denoiser-xl-$CONFIG-uid-*/checkpoints/checkpoint_6000 | head -1)/model.safetensors ;;
    lora*)      VOXEL=$R/inference_weights/voxel_$CONFIG.safetensors ;;  # merged with scripts/merge_lora_checkpoint.py
    *) echo "unknown config $CONFIG"; exit 1 ;;
esac
[ -f "$VOXEL" ] || { echo "missing $VOXEL"; exit 1; }
EPISODES=$(paste -sd, dataset_toolkits/manifests/full29k/eval16_sha256.txt)

echo "=== $CONFIG: voxel denoiser $VOXEL, $(date) ==="
accelerate launch --multi_gpu --num_processes 2 --mixed_precision bf16 --dynamo_backend no -m scripts.run_inference \
    --pipeline-variant XL --dataset-path datasets/luanti_full29k --eval-instances "$EPISODES" --num-frames 300 \
    --include-initial-pixel-frame --include-initial-voxel-frame \
    --seed 0 --mixed-precision bf16 --output-dir $R/eval16/300f/$CONFIG \
    --checkpoints.voxel-encoder $P/voxel-encoder/model.safetensors --checkpoints.voxel-decoder $P/voxel-decoder/model.safetensors \
    --checkpoints.pixel-vae $P/pixel-vae/model.safetensors --checkpoints.camera-model $P/camera-model/model.safetensors \
    --checkpoints.pixel-denoiser $P/pixel-denoiser/model.safetensors --checkpoints.voxel-class-embedder $P/voxel-class-embedder/model.safetensors \
    --checkpoints.voxel-denoiser $VOXEL
echo "=== $CONFIG done (exit $?), $(ls $R/eval16/300f/$CONFIG/*.mp4 2>/dev/null | wc -l)/16 videos, $(date) ==="
