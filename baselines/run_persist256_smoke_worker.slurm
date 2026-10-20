#!/bin/bash
#SBATCH --job-name=persist256-smokeworker
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=../outputs/finetune_oasis_luanti_hudnav_1000ep/eval_persist256_smoke/slurm_worker_%j.log

export PYTHONUNBUFFERED=1
export TORCH_HOME=/scratch/user/lsokoudj/.cache/torch
export VOXELWM_CACHE_DIR=/scratch/user/lsokoudj/.cache/voxelwm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /scratch/user/lsokoudj/PERSIST/.venv/bin/activate
cd /scratch/user/lsokoudj/PERSIST/baselines

read CKPT_PATH CKPT_STEP < <(python resolve_latest_ckpt.py --ckpt-dir ../outputs/finetune_oasis_luanti_hudnav_1000ep)
echo "Using checkpoint: $CKPT_PATH (step $CKPT_STEP)"

python eval_oasis_finetuned_fvd_shard.py \
    --all-episodes --max-episodes 4 --frame-lengths 20 40 \
    --dataset-root /scratch/user/lsokoudj/.cache/huggingface/hub/datasets--PERSIST-team--persist-eval-sample/snapshots/d3a3bee6560304397093ae66523451b9e0d067f7 \
    --ckpt-path "$CKPT_PATH" \
    --use-oasis-native-action-layout \
    --batch-size 2 \
    --video-output-dir ../outputs/finetune_oasis_luanti_hudnav_1000ep/eval_persist256_smoke/worker_videos \
    --output-dir ../outputs/finetune_oasis_luanti_hudnav_1000ep/eval_persist256_smoke/episode_feats \
    --device cuda:0

echo "--- episode_feats dir ---"
ls -la ../outputs/finetune_oasis_luanti_hudnav_1000ep/eval_persist256_smoke/episode_feats/

echo "--- test merge on this tiny set ---"
python reap_stale_claims.py --output-dir ../outputs/finetune_oasis_luanti_hudnav_1000ep/eval_persist256_smoke/episode_feats
python merge_oasis_fvd_shards.py \
    --episode-feats-glob '../outputs/finetune_oasis_luanti_hudnav_1000ep/eval_persist256_smoke/episode_feats/*.npz' \
    --frame-lengths 20 40 \
    --checkpoint "$CKPT_PATH" --checkpoint-step "$CKPT_STEP" --split smoke_worker_test \
    --output-json ../outputs/finetune_oasis_luanti_hudnav_1000ep/eval_persist256_smoke/merged_worker_smoke.json
