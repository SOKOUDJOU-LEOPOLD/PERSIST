#!/bin/bash
#SBATCH --job-name=oasis-206ep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --nodelist=fc098
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-user=lsokoudj@tamu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=../outputs/finetune_oasis_scaling_206ep/slurm_%j.log

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

export WANDB_MODE=offline
export WANDB_DIR=/scratch/user/lsokoudj/PERSIST/outputs/finetune_oasis_scaling_206ep
export TORCH_HOME=/scratch/user/lsokoudj/.cache/torch

source /scratch/user/lsokoudj/PERSIST/.venv/bin/activate
cd /scratch/user/lsokoudj/PERSIST/baselines
mkdir -p ../outputs/finetune_oasis_scaling_206ep

python finetune_oasis.py \
    --split-path ../outputs/oasis_scaling_206ep.json \
    --output-dir ../outputs/finetune_oasis_scaling_206ep \
    --resume-from ../outputs/finetune_oasis_scaling_206ep/step_009500_newid.pt \
    --use-oasis-native-action-layout \
    --informed-action-init \
    --lr-pretrained 1e-6 --lr-new-layer 1e-4 \
    --num-steps 20000 --checkpoint-every 500 --val-every 100 \
    --wandb-run-name oasis-native-206ep \
    --device cuda:0
