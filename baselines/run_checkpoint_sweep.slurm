#!/bin/bash
# Evaluates a FIXED set of 16 episodes (first 16 from persist-eval-sample, by sorted level_id --
# same set for every checkpoint, so results are directly comparable across steps, not confounded
# by different episode composition) at full 600-frame rollout, for one checkpoint per array task.
# Answers: is 600-frame rollout quality (FVD/FID/PSNR) actually improving over training steps,
# or just the easy 1-step teacher-forced val metric training already logs?
#SBATCH --job-name=hudnav-ckpt-sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --array=0-5
#SBATCH --output=../outputs/finetune_oasis_luanti_hudnav_1000ep/checkpoint_sweep/slurm_%A_%a.log

export PYTHONUNBUFFERED=1
export TORCH_HOME=/scratch/user/lsokoudj/.cache/torch
export VOXELWM_CACHE_DIR=/scratch/user/lsokoudj/.cache/voxelwm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /scratch/user/lsokoudj/PERSIST/.venv/bin/activate
cd /scratch/user/lsokoudj/PERSIST/baselines

STEPS=(5000 10000 15000 20000 25000 27500)
STEP=${STEPS[$SLURM_ARRAY_TASK_ID]}
CKPT_PATH=$(printf "../outputs/finetune_oasis_luanti_hudnav_1000ep/step_%06d.pt" "$STEP")
OUT_DIR="../outputs/finetune_oasis_luanti_hudnav_1000ep/checkpoint_sweep"

echo "Evaluating checkpoint step=$STEP ($CKPT_PATH)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

python eval_oasis_finetuned_fvd.py \
    --all-episodes --max-episodes 16 --frame-lengths 200 400 600 \
    --dataset-root /scratch/user/lsokoudj/.cache/huggingface/hub/datasets--PERSIST-team--persist-eval-sample/snapshots/d3a3bee6560304397093ae66523451b9e0d067f7 \
    --ckpt-path "$CKPT_PATH" \
    --use-oasis-native-action-layout \
    --batch-size 8 \
    --output-json "$OUT_DIR/eval_step_${STEP}.json" \
    --device cuda:0
