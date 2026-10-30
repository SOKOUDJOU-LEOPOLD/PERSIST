#!/bin/bash
# Two short pixel-denoiser trials on seg10k (moving agents), both starting from the best static LoRA pixel checkpoint
# (step 61k, merged): LoRA rank 32 vs full fine-tune. Same recipe otherwise, so the comparison is LoRA vs full only.
# 4 A100s each, batch 2 x grad-acc 2 x 4 GPUs = 16 per update, 4k steps (~2.7 h). Checkpoints at 2k and 4k.
# Usage: bash dataset_toolkits/manifests/seg10k/pixel_trials.sh [dependency job id, e.g. the preprocessing finalize job]
cd /scratch/user/lsokoudj/PERSIST
M=dataset_toolkits/manifests/full29k
R=/scratch/user/lsokoudj/PERSIST/outputs/runs_seg10k
mkdir -p $R/logs
export DATASET=datasets/seg10k/train OUT_DIR=$R WANDB_DIR=$R WANDB_PROJECT=persist-seg10k RUN_PREFIX=seg10k
export INIT_CKPT=outputs/runs_full29k/inference_weights/pixel_lora-r32_step61000.safetensors
export TOTAL_STEPS=4000 I_VAL=1000 I_SAVE=2000 RETENTION=all LR=1e-4 WARMUP=500 GRAD_ACC=2 NOISE_COND=0.1
export MAX_RESUBMITS=2
DEP=${1:+--dependency=afterok:$1}
SB="--nodes=4 --gres=gpu:a100:1 --cpus-per-task=16 --mem=120G --time=8:00:00 $DEP"
LORA_RANK=32 RUN_SUFFIX=lora-r32-trial sbatch $SB --job-name=seg10k-pixel-lora-trial \
    --output=$R/logs/%x_%j.log $M/train_full29k.slurm pixel
RUN_SUFFIX=full-trial sbatch $SB --job-name=seg10k-pixel-full-trial \
    --output=$R/logs/%x_%j.log $M/train_full29k.slurm pixel
