#!/bin/bash
#SBATCH --job-name=fid-per-frame-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=../outputs/finetune_oasis_luanti_hudnav_1000ep/eval_persist256_step40000/slurm_fid_per_frame_smoke_%j.log

export PYTHONUNBUFFERED=1
export TORCH_HOME=/scratch/user/lsokoudj/.cache/torch
export VOXELWM_CACHE_DIR=/scratch/user/lsokoudj/.cache/voxelwm

source /scratch/user/lsokoudj/PERSIST/.venv/bin/activate
cd /scratch/user/lsokoudj/PERSIST/baselines

python compute_per_frame_fid.py \
    --dataset-root /scratch/user/lsokoudj/.cache/huggingface/hub/datasets--PERSIST-team--persist-eval-sample/snapshots/d3a3bee6560304397093ae66523451b9e0d067f7 \
    --all-episodes --max-episodes 4 \
    --generated-video-dir ../outputs/finetune_oasis_luanti_hudnav_1000ep/eval_persist256_step40000/videos \
    --num-frames 600 \
    --output-csv ../outputs/finetune_oasis_luanti_hudnav_1000ep/eval_persist256_step40000/fid_per_frame_smoke.csv \
    --device cuda:0
