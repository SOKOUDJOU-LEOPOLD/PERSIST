#!/bin/bash
#SBATCH --job-name=persist256-probe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=../outputs/finetune_oasis_luanti_hudnav_1000ep/eval_persist256_smoke/slurm_probe_%j.log

export PYTHONUNBUFFERED=1
export TORCH_HOME=/scratch/user/lsokoudj/.cache/torch
export VOXELWM_CACHE_DIR=/scratch/user/lsokoudj/.cache/voxelwm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /scratch/user/lsokoudj/PERSIST/.venv/bin/activate
cd /scratch/user/lsokoudj/PERSIST/baselines

read CKPT_PATH CKPT_STEP < <(python resolve_latest_ckpt.py --ckpt-dir ../outputs/finetune_oasis_luanti_hudnav_1000ep)
echo "Using checkpoint: $CKPT_PATH (step $CKPT_STEP)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

python probe_batch_size_memory.py \
    --ckpt-path "$CKPT_PATH" \
    --dataset-root /scratch/user/lsokoudj/.cache/huggingface/hub/datasets--PERSIST-team--persist-eval-sample/snapshots/d3a3bee6560304397093ae66523451b9e0d067f7 \
    --use-oasis-native-action-layout \
    --batch-sizes 4 8 12 16 24 32 48 64 \
    --device cuda:0
