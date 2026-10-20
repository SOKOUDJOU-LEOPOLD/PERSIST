#!/bin/bash
#SBATCH --job-name=persist256-smokeB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=../outputs/finetune_oasis_luanti_hudnav_1000ep/eval_persist256_smoke/slurm_stageB_%j.log

export PYTHONUNBUFFERED=1
export TORCH_HOME=/scratch/user/lsokoudj/.cache/torch
export VOXELWM_CACHE_DIR=/scratch/user/lsokoudj/.cache/voxelwm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /scratch/user/lsokoudj/PERSIST/.venv/bin/activate
cd /scratch/user/lsokoudj/PERSIST/baselines

read CKPT_PATH CKPT_STEP < <(python resolve_latest_ckpt.py --ckpt-dir ../outputs/finetune_oasis_luanti_hudnav_1000ep)
echo "Using checkpoint: $CKPT_PATH (step $CKPT_STEP)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

# Uses the SAME worker script (eval_oasis_finetuned_fvd_shard.py) and batch-size (4) the real
# A100 pool will run in Stage C -- this validates the actual code path, not just a proxy.
python eval_oasis_finetuned_fvd_shard.py \
    --all-episodes --max-episodes 4 --frame-lengths 200 400 600 \
    --dataset-root /scratch/user/lsokoudj/.cache/huggingface/hub/datasets--PERSIST-team--persist-eval-sample/snapshots/d3a3bee6560304397093ae66523451b9e0d067f7 \
    --ckpt-path "$CKPT_PATH" \
    --use-oasis-native-action-layout \
    --batch-size 4 \
    --video-output-dir ../outputs/finetune_oasis_luanti_hudnav_1000ep/eval_persist256_smoke/videos_stageB \
    --output-dir ../outputs/finetune_oasis_luanti_hudnav_1000ep/eval_persist256_smoke/episode_feats_stageB \
    --device cuda:0

echo "--- disk usage of smoke output ---"
du -sh ../outputs/finetune_oasis_luanti_hudnav_1000ep/eval_persist256_smoke/

echo "--- merge test ---"
python merge_oasis_fvd_shards.py \
    --episode-feats-glob '../outputs/finetune_oasis_luanti_hudnav_1000ep/eval_persist256_smoke/episode_feats_stageB/*.npz' \
    --frame-lengths 200 400 600 \
    --checkpoint "$CKPT_PATH" --checkpoint-step "$CKPT_STEP" --split stageB_test \
    --output-json ../outputs/finetune_oasis_luanti_hudnav_1000ep/eval_persist256_smoke/merged_stageB.json
