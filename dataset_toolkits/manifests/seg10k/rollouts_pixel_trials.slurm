#!/bin/bash
# 600-frame rollouts comparing three pixel denoisers on agent-heavy seg10k validation episodes:
#   static61k = the starting point (best static LoRA pixel, step 61k), lora = LoRA r32 trial, full = full fine-tune trial
# Everything else is fixed, so differences come from the pixel denoiser: ground-truth camera (--use-camera-gt), voxel
# denoiser = best static (step 20k), same episodes (rollout_sha256.txt from pick_agent_episodes.py), seed 0.
# The LoRA trial is merged here first (the pixel merge needs a GPU); the full trial's checkpoint loads as is.
# Usage: sbatch [--dependency=afterok:<lora job>:<full job>] rollouts_pixel_trials.slurm [step=4000]
#SBATCH --job-name=seg10k-pixel-trial-rollouts
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/user/lsokoudj/PERSIST/outputs/runs_seg10k/logs/%x_%j.log

set -uo pipefail
cd /scratch/user/lsokoudj/PERSIST
module load GCC/13.2.0 CUDA/12.6.0
source .venv/bin/activate
export LD_PRELOAD=$EBROOTGCCCORE/lib64/libstdc++.so.6
export PYTHONPATH=.

STEP=${1:-4000}
P=checkpoints/pretrained
W=outputs/runs_full29k/inference_weights
R=outputs/runs_seg10k
LORA_DIR=$(ls -d $R/pixel_diffuser/seg10k-pixel-denoiser-lora-r32-trial-uid-*/checkpoints/checkpoint_$STEP | head -1)
FULL_DIR=$(ls -d $R/pixel_diffuser/seg10k-pixel-denoiser-full-trial-uid-*/checkpoints/checkpoint_$STEP | head -1)
OUT=$R/rollouts_pixel_trials/step$STEP
[ -f dataset_toolkits/manifests/seg10k/rollout_sha256.txt ] || python dataset_toolkits/manifests/seg10k/pick_agent_episodes.py 6 || exit 1
EPS=$(paste -sd, dataset_toolkits/manifests/seg10k/rollout_sha256.txt)
mkdir -p $R/inference_weights
LORA_W=$R/inference_weights/pixel_lora-r32-trial_step$STEP.safetensors
[ -f $LORA_W ] || python scripts/merge_lora_checkpoint.py --component pixel --checkpoint-dir $LORA_DIR --rank 32 \
    --out $LORA_W || exit 1

declare -A PIX
PIX[static61k]="--checkpoints.pixel-denoiser $W/pixel_lora-r32_step61000.safetensors --checkpoints.voxel-class-embedder $P/voxel-class-embedder/model.safetensors"
PIX[lora]="--checkpoints.pixel-denoiser $LORA_W --checkpoints.voxel-class-embedder $P/voxel-class-embedder/model.safetensors"
PIX[full]="--checkpoints.pixel-denoiser $FULL_DIR/model.safetensors --checkpoints.voxel-class-embedder $FULL_DIR/model_1.safetensors"

for cfg in static61k lora full; do
    [ $cfg = static61k ] && [ -d $OUT/../static61k ] && continue  # the starting point does not depend on the step
    dest=$OUT/$cfg; [ $cfg = static61k ] && dest=$OUT/../static61k
    echo "=== $cfg, $(date) ==="
    accelerate launch --multi_gpu --num_processes 2 --mixed_precision bf16 --dynamo_backend no -m scripts.run_inference \
        --pipeline-variant XL --dataset-path datasets/seg10k/train --eval-instances "$EPS" --num-frames 600 \
        --use-camera-gt --include-initial-pixel-frame --include-initial-voxel-frame --seed 0 --mixed-precision bf16 \
        --output-dir $dest \
        --checkpoints.voxel-encoder $P/voxel-encoder/model.safetensors --checkpoints.voxel-decoder $P/voxel-decoder/model.safetensors \
        --checkpoints.pixel-vae $P/pixel-vae/model.safetensors --checkpoints.voxel-denoiser $W/voxel_lora-r32_upd10000.safetensors \
        --checkpoints.camera-model $W/camera_lora-r32_step116000.safetensors ${PIX[$cfg]} || echo "=== FAILED: $cfg ==="
done
python dataset_toolkits/manifests/seg10k/compose_pixel_trials.py $OUT
echo "=== all done, $(date) ==="
