#!/bin/bash
# Rollouts on held-out validation episodes: official pretrained weights vs. the full29k fine-tune (step 20000),
# raw and EMA weights. Same episodes and seed for every config. The VAEs are the official ones in all configs.
# Usage: sbatch rollouts_full29k.slurm
#SBATCH --job-name=rollouts-full29k
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/user/lsokoudj/PERSIST/outputs/runs_full29k/logs/%x_%j.log

set -uo pipefail
cd /scratch/user/lsokoudj/PERSIST
module load GCC/13.2.0 CUDA/12.6.0
source .venv/bin/activate
export LD_PRELOAD=$EBROOTGCCCORE/lib64/libstdc++.so.6
export PYTHONPATH=.

P=checkpoints/pretrained
R=outputs/runs_full29k
V=$R/voxel_diffuser/full29k-voxel-denoiser-xl-uid-b6382043/checkpoints/checkpoint_20000
X=$R/pixel_diffuser/full29k-pixel-denoiser-uid-49074b61/checkpoints/checkpoint_20000
C=$R/camera_transformer/full29k-camera-model-uid-f9de8a12/checkpoints/checkpoint_20000
W=$R/inference_weights
OUT=$R/rollouts
SHORT=00cfab24b970f156e125ad5e8eb8e6e41e8f84d005896662099bf65fc4a03797,0158367167c4da5d5f494406b4cbb3bfd666e257e8e70884e9467aa5c43257b9,02a533d0f5abd79616b46621b2919b1aa301d8765c74921a1efbf927fd3ff657,05dbd97aa315ced77d2ae24585f621fb2525a8f812a47a4a519b1931d59454b6
LONG=00cfab24b970f156e125ad5e8eb8e6e41e8f84d005896662099bf65fc4a03797,02a533d0f5abd79616b46621b2919b1aa301d8765c74921a1efbf927fd3ff657

VAES="--checkpoints.voxel-encoder $P/voxel-encoder/model.safetensors --checkpoints.voxel-decoder $P/voxel-decoder/model.safetensors
  --checkpoints.pixel-vae $P/pixel-vae/model.safetensors"
declare -A CFG
CFG[pretrained]="--checkpoints.voxel-denoiser $P/voxel-denoiser-xl/model.safetensors --checkpoints.camera-model $P/camera-model/model.safetensors
  --checkpoints.pixel-denoiser $P/pixel-denoiser/model.safetensors --checkpoints.voxel-class-embedder $P/voxel-class-embedder/model.safetensors"
CFG[finetuned_raw]="--checkpoints.voxel-denoiser $V/model.safetensors --checkpoints.camera-model $C/model.safetensors
  --checkpoints.pixel-denoiser $X/model.safetensors --checkpoints.voxel-class-embedder $X/model_1.safetensors"
CFG[finetuned_ema]="--checkpoints.voxel-denoiser $W/voxel_denoiser_ema_step20000.safetensors --checkpoints.camera-model $W/camera_camera_ema_step20000.safetensors
  --checkpoints.pixel-denoiser $W/pixel_denoiser_ema_step20000.safetensors --checkpoints.voxel-class-embedder $W/pixel_voxel_class_embedder_ema_step20000.safetensors"

run() {  # $1 config, $2 frames, $3 instances
    echo "=== $1, $2 frames, $(date) ==="
    accelerate launch --multi_gpu --num_processes 2 --mixed_precision bf16 --dynamo_backend no -m scripts.run_inference \
        --pipeline-variant XL --dataset-path datasets/luanti_full29k --eval-instances "$3" --num-frames $2 \
        --seed 0 --mixed-precision bf16 --output-dir $OUT/${2}f/$1 $VAES ${CFG[$1]} || echo "=== FAILED: $1 $2 ==="
}
for cfg in pretrained finetuned_raw finetuned_ema; do run $cfg 16 "$SHORT"; done
for cfg in pretrained finetuned_raw finetuned_ema; do run $cfg 600 "$LONG"; done
echo "=== all rollouts done $(date) ==="
