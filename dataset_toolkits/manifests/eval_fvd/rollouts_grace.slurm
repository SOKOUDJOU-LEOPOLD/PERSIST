#!/bin/bash
# FVD eval rollouts on Grace: Grace counterpart of rollouts_chzhang.sh (same weights, flags and output names).
# A job array splits the set's validation episodes into SLURM_ARRAY_TASK_COUNT chunks, one 2-A100 node each;
# every task only generates the mp4s still missing in its chunk, so resubmitting the same array resumes.
# Score afterwards with score_fvd_grace.slurm (needs all chunks done).
# Usage: sbatch --array=0-<K-1> rollouts_grace.slurm <persist_eval|hudnav_eval> <w_v0|wo_v0> [batch=1]
#SBATCH --job-name=fvd-rollouts
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/user/lsokoudj/PERSIST/outputs/eval_fvd/logs/%x_%A_%a.log

set -uo pipefail
cd /scratch/user/lsokoudj/PERSIST
module load GCC/13.2.0 CUDA/12.6.0
source .venv/bin/activate
export LD_PRELOAD=$EBROOTGCCCORE/lib64/libstdc++.so.6
export PYTHONPATH=.

SET=${1:?set}; MODE=${2:?mode}; BATCH=${3:-1}
case $MODE in
    w_v0) INCLUDE_FLAGS="--include-initial-voxel-frame --include-initial-pixel-frame" ;;
    wo_v0) INCLUDE_FLAGS="" ;;
    *) echo "unknown mode $MODE"; exit 1 ;;
esac
DATASET=datasets/$SET
P=checkpoints/pretrained
W=outputs/runs_full29k/inference_weights
OUT=outputs/eval_fvd/rollouts/$SET/$MODE
mkdir -p $OUT

# this task's chunk: the validation shas (same filter as rollouts_chzhang.sh), sorted, every K-th from index k
CHUNK=$(python - <<EOF
import os, pandas as pd
m = pd.read_csv("$DATASET/metadata.csv")
shas = sorted(m[(m.validation_set == True) & (m.level_quality_score >= 1.0)].sha256)
mine = shas[$SLURM_ARRAY_TASK_ID::$SLURM_ARRAY_TASK_COUNT]
print(",".join(s for s in mine if not os.path.exists(f"$OUT/XL_{s}.mp4")))
EOF
)
echo "=== $SET / $MODE task $SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT: $(echo $CHUNK | tr , '\n' | grep -c .) to generate, $(date) ==="
[ -z "$CHUNK" ] && exit 0
accelerate launch --multi_gpu --num_processes 2 --mixed_precision bf16 --dynamo_backend no -m scripts.run_inference \
    --pipeline-variant XL --dataset-path $DATASET --eval-instances "$CHUNK" --num-frames 600 \
    $INCLUDE_FLAGS --seed 0 --mixed-precision bf16 --batch-size $BATCH --output-dir $OUT \
    --checkpoints.voxel-encoder $P/voxel-encoder/model.safetensors --checkpoints.voxel-decoder $P/voxel-decoder/model.safetensors \
    --checkpoints.pixel-vae $P/pixel-vae/model.safetensors --checkpoints.voxel-denoiser $W/voxel_lora-r32_upd10000.safetensors \
    --checkpoints.camera-model $W/camera_lora-r32_step116000.safetensors \
    --checkpoints.pixel-denoiser $W/pixel_lora-r32_step61000.safetensors \
    --checkpoints.voxel-class-embedder $P/voxel-class-embedder/model.safetensors
echo "=== exit $?, $(date) ==="
