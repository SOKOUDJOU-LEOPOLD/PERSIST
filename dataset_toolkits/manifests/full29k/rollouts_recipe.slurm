#!/bin/bash
# Recipe test rollouts: final checkpoint of each short voxel-denoiser fine-tune (recipe0/1/2), paired with the
# pretrained pixel denoiser and the ground-truth camera path, real first frame, 600 frames on the hudnav episode
# where pretrained-everything stays stable. Compare with C_pretrained_gt_camera / F_ft_voxel_pre_pixel_gt_camera.
#SBATCH --job-name=rollouts-recipe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/user/lsokoudj/PERSIST/outputs/runs_full29k/logs/%x_%j.log

set -uo pipefail
cd /scratch/user/lsokoudj/PERSIST
module load GCC/13.2.0 CUDA/12.6.0
source .venv/bin/activate
export LD_PRELOAD=$EBROOTGCCCORE/lib64/libstdc++.so.6
export PYTHONPATH=.

P=checkpoints/pretrained
R=outputs/runs_full29k
OUT=$R/rollouts_diag
EP=02a533d0f5abd79616b46621b2919b1aa301d8765c74921a1efbf927fd3ff657
FIXED="--checkpoints.voxel-encoder $P/voxel-encoder/model.safetensors --checkpoints.voxel-decoder $P/voxel-decoder/model.safetensors
  --checkpoints.pixel-vae $P/pixel-vae/model.safetensors --checkpoints.camera-model $P/camera-model/model.safetensors
  --checkpoints.pixel-denoiser $P/pixel-denoiser/model.safetensors --checkpoints.voxel-class-embedder $P/voxel-class-embedder/model.safetensors"

run1() {  # $1 recipe tag, $2 gpu
    local ckpt
    ckpt=$(ls -d $R/voxel_diffuser/full29k-voxel-denoiser-xl-$1-uid-*/checkpoints/checkpoint_6000 2>/dev/null | head -1)
    [ -z "$ckpt" ] && { echo "=== FAILED: no final checkpoint for $1 ==="; return; }
    echo "=== $1 ($ckpt) on GPU $2, $(date) ==="
    CUDA_VISIBLE_DEVICES=$2 python -m scripts.run_inference \
        --pipeline-variant XL --dataset-path datasets/luanti_full29k --eval-instances $EP --num-frames 600 \
        --include-initial-pixel-frame --include-initial-voxel-frame --use-camera-gt \
        --seed 0 --mixed-precision bf16 --output-dir $OUT/600f/R_$1_voxel_pre_pixel_gt_camera \
        $FIXED --checkpoints.voxel-denoiser $ckpt/model.safetensors || echo "=== FAILED: $1 ==="
}
run1 recipe0 0 &
run1 recipe1 1 &
wait
run1 recipe2 0
echo "=== all rollouts done $(date) ==="
