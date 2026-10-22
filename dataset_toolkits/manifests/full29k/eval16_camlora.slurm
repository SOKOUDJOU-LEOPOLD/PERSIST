#!/bin/bash
# Camera-LoRA check: LoRA-10k voxel denoiser + merged LoRA camera (given update count) + pretrained pixel denoiser,
# on eval16 episodes. Same seed and output layout as eval16_rollouts.slurm (config lora10k_camlora<upd>).
# Usage: sbatch eval16_camlora.slurm <upd> <first>-<last>      (episodes = 1-based lines of eval16_sha256.txt)
#SBATCH --job-name=eval16-camlora
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/user/lsokoudj/PERSIST/outputs/runs_full29k/logs/%x_%j.log

set -uo pipefail
cd /scratch/user/lsokoudj/PERSIST
module load GCC/13.2.0 CUDA/12.6.0
source .venv/bin/activate
export LD_PRELOAD=$EBROOTGCCCORE/lib64/libstdc++.so.6
export PYTHONPATH=.

UPD=$1; RANGE=$2
P=checkpoints/pretrained
R=outputs/runs_full29k
W=$R/inference_weights
EPISODES=$(sed -n "${RANGE%-*},${RANGE#*-}p" dataset_toolkits/manifests/full29k/eval16_sha256.txt | paste -sd,)
echo "=== lora10k_camlora$UPD episodes $RANGE on $(hostname), $(date) ==="
python -m scripts.run_inference \
    --pipeline-variant XL --dataset-path datasets/luanti_full29k --eval-instances "$EPISODES" --num-frames 300 \
    --include-initial-pixel-frame --include-initial-voxel-frame \
    --seed 0 --mixed-precision bf16 --output-dir $R/eval16/300f/lora10k_camlora$UPD \
    --checkpoints.voxel-encoder $P/voxel-encoder/model.safetensors --checkpoints.voxel-decoder $P/voxel-decoder/model.safetensors \
    --checkpoints.pixel-vae $P/pixel-vae/model.safetensors --checkpoints.voxel-denoiser $W/voxel_lora-r32_upd10000.safetensors \
    --checkpoints.camera-model $W/camera_lora-r32_upd$UPD.safetensors \
    --checkpoints.pixel-denoiser $P/pixel-denoiser/model.safetensors --checkpoints.voxel-class-embedder $P/voxel-class-embedder/model.safetensors
echo "=== done (exit $?), $(date) ==="
