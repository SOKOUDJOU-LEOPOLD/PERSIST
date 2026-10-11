#!/bin/bash
#SBATCH --job-name=grid-hudnav
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=../outputs/finetune_oasis_luanti_hudnav_1000ep/qualitative_grids/slurm_%j.log

source /scratch/user/lsokoudj/PERSIST/.venv/bin/activate
cd /scratch/user/lsokoudj/PERSIST/baselines

python generate_luanti_split_grids.py \
    --ckpt-path ../outputs/finetune_oasis_luanti_hudnav_1000ep/step_027500.pt \
    --dataset-root /scratch/user/lsokoudj/PERSIST/data/luanti/extracted_hudnav/OpenWorldCreativeHud-v0 \
    --split-path ../outputs/oasis_luanti_hudnav_1000ep.json \
    --output-dir ../outputs/finetune_oasis_luanti_hudnav_1000ep/qualitative_grids \
    --splits test --split-labels Test \
    --episodes-per-split 2 --num-frames 150 --ddim-steps 10 --fps 20 --device cuda:0
