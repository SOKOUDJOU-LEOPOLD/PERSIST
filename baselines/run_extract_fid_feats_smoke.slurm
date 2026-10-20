#!/bin/bash
#SBATCH --job-name=fid-feat-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=../outputs/finetune_oasis_luanti_hudnav_1000ep/eval_persist256_step40000/slurm_fid_feat_smoke_%j.log

export PYTHONUNBUFFERED=1
export TORCH_HOME=/scratch/user/lsokoudj/.cache/torch
export VOXELWM_CACHE_DIR=/scratch/user/lsokoudj/.cache/voxelwm

source /scratch/user/lsokoudj/PERSIST/.venv/bin/activate
cd /scratch/user/lsokoudj/PERSIST/baselines

python extract_per_frame_fid_features.py \
    --dataset-root /scratch/user/lsokoudj/.cache/huggingface/hub/datasets--PERSIST-team--persist-eval-sample/snapshots/d3a3bee6560304397093ae66523451b9e0d067f7 \
    --all-episodes \
    --generated-video-dir ../outputs/finetune_oasis_luanti_hudnav_1000ep/eval_persist256_step40000/videos \
    --num-frames 600 \
    --output-dir /tmp/fid_feat_smoke_test \
    --device cuda:0
