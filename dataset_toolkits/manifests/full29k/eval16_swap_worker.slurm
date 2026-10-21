#!/bin/bash
# Check 2: keep the LoRA-10k voxel denoiser and swap in the full29k step-20000 EMA camera model and/or pixel denoiser,
# to decide whether those two components need retraining. Same 16 episodes, seed and output layout as eval16_rollouts.slurm.
# Work is split into (config, 4-episode chunk) items; each worker claims items with an atomic mkdir, so workers can be
# submitted to several GPU types at once and whichever start first do the work (the rest exit as soon as nothing is left).
# Usage: sbatch --partition=gpu-a40 --gres=gpu:a40:1 eval16_swap_worker.slurm   (and/or --partition=gpu --gres=gpu:a100:1)
#SBATCH --job-name=eval16-swap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/user/lsokoudj/PERSIST/outputs/runs_full29k/logs/%x_%j.log

set -uo pipefail
cd /scratch/user/lsokoudj/PERSIST
module load GCC/13.2.0 CUDA/12.6.0
source .venv/bin/activate
export LD_PRELOAD=$EBROOTGCCCORE/lib64/libstdc++.so.6
export PYTHONPATH=.

P=checkpoints/pretrained
R=outputs/runs_full29k
W=$R/inference_weights
EPS=dataset_toolkits/manifests/full29k/eval16_sha256.txt
CLAIMS=$R/eval16/claims_swap
mkdir -p $CLAIMS

CAM_PRE="--checkpoints.camera-model $P/camera-model/model.safetensors"
CAM_FT="--checkpoints.camera-model $W/camera_camera_ema_step20000.safetensors"
PIX_PRE="--checkpoints.pixel-denoiser $P/pixel-denoiser/model.safetensors --checkpoints.voxel-class-embedder $P/voxel-class-embedder/model.safetensors"
PIX_FT="--checkpoints.pixel-denoiser $W/pixel_denoiser_ema_step20000.safetensors --checkpoints.voxel-class-embedder $W/pixel_voxel_class_embedder_ema_step20000.safetensors"
declare -A CFG
CFG[lora10k_ftcam]="$CAM_FT $PIX_PRE"
CFG[lora10k_ftpix]="$CAM_PRE $PIX_FT"
CFG[lora10k_ftcampix]="$CAM_FT $PIX_FT"

for CONFIG in lora10k_ftcampix lora10k_ftcam lora10k_ftpix; do
    for RANGE in 1-4 5-8 9-12 13-16; do
        mkdir $CLAIMS/${CONFIG}_$RANGE 2>/dev/null || continue  # someone else has it
        EPISODES=$(sed -n "${RANGE%-*},${RANGE#*-}p" $EPS | paste -sd,)
        echo "=== $CONFIG episodes $RANGE on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader), $(date) ==="
        python -m scripts.run_inference \
            --pipeline-variant XL --dataset-path datasets/luanti_full29k --eval-instances "$EPISODES" --num-frames 300 \
            --include-initial-pixel-frame --include-initial-voxel-frame \
            --seed 0 --mixed-precision bf16 --output-dir $R/eval16/300f/$CONFIG \
            --checkpoints.voxel-encoder $P/voxel-encoder/model.safetensors --checkpoints.voxel-decoder $P/voxel-decoder/model.safetensors \
            --checkpoints.pixel-vae $P/pixel-vae/model.safetensors \
            --checkpoints.voxel-denoiser $W/voxel_lora-r32_upd10000.safetensors ${CFG[$CONFIG]} \
            && touch $CLAIMS/${CONFIG}_$RANGE/done || echo "=== FAILED: $CONFIG $RANGE ==="
    done
done
echo "=== worker done, $(date) ==="
