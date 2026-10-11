#!/bin/bash
#SBATCH --job-name=oasis-luanti-hudnav-1000ep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --nodelist=fc031
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --mail-user=lsokoudj@tamu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=../outputs/finetune_oasis_luanti_hudnav_1000ep/slurm_%j.log

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

export WANDB_MODE=offline
export WANDB_DIR=/scratch/user/lsokoudj/PERSIST/outputs/finetune_oasis_luanti_hudnav_1000ep
export TORCH_HOME=/scratch/user/lsokoudj/.cache/torch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /scratch/user/lsokoudj/PERSIST/.venv/bin/activate
cd /scratch/user/lsokoudj/PERSIST/baselines
mkdir -p ../outputs/finetune_oasis_luanti_hudnav_1000ep

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes 4 finetune_oasis.py \
    --split-path ../outputs/oasis_luanti_hudnav_1000ep.json \
    --dataset-root /scratch/user/lsokoudj/PERSIST/data/luanti/extracted_hudnav/OpenWorldCreativeHud-v0 \
    --output-dir ../outputs/finetune_oasis_luanti_hudnav_1000ep \
    --use-oasis-native-action-layout \
    --informed-action-init \
    --batch-size 3 \
    --lr-pretrained 1e-6 --lr-new-layer 1e-4 \
    --num-steps 100000 --checkpoint-every 500 --val-every 100 \
    --wandb-run-name oasis-luanti-hudnav-1000ep
