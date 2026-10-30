#!/bin/bash
# Long pixel-denoiser fine-tune on seg10k (moving agents): fresh LoRA rank 64 on top of the best static LoRA pixel
# checkpoint (step 61k, merged), normal (unweighted) loss. Chosen over full fine-tuning after the 4k-step trials
# (pixel_trials.sh): same recipe, safer against forgetting the static world.
# 4 A100s, batch 2 x grad-acc 2 x 4 GPUs = 16 per update, 20k steps. Validation and checkpoint every 1k steps
# (I_VAL / I_SAVE count updates, x GRAD_ACC = micro-batch steps), plus a start-of-run validation (baseline).
# NODES=3 runs on 3 A100s instead (12 per update, same steps and wall time) when only 3 are free.
# Usage: [NODES=3] bash dataset_toolkits/manifests/seg10k/pixel_lora_r64.sh
cd /scratch/user/lsokoudj/PERSIST
M=dataset_toolkits/manifests/full29k
R=/scratch/user/lsokoudj/PERSIST/outputs/runs_seg10k
mkdir -p $R/logs
export DATASET=datasets/seg10k/train OUT_DIR=$R WANDB_DIR=$R WANDB_PROJECT=persist-seg10k RUN_PREFIX=seg10k
export INIT_CKPT=outputs/runs_full29k/inference_weights/pixel_lora-r32_step61000.safetensors
export LORA_RANK=64 RUN_SUFFIX=lora-r64
export TOTAL_STEPS=20000 I_VAL=500 I_SAVE=500 RETENTION=all LR=1e-4 WARMUP=1000 GRAD_ACC=2 NOISE_COND=0.1
export VAL_AT_START=1
sbatch --nodes=${NODES:-4} --gres=gpu:a100:1 --cpus-per-task=16 --mem=120G --time=1-06:00:00 --job-name=seg10k-pixel-lora-r64 \
    --output=$R/logs/%x_%j.log $M/train_full29k.slurm pixel
