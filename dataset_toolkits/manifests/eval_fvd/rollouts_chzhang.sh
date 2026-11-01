#!/bin/bash
# FVD eval rollouts on csce-chzhang (2 GPUs): generate `<variant>_<sha256>.mp4` rollouts for every
# validation episode of one eval set, in one mode, then score FVD at 200/400/600 frames.
# Best (lowest-val-loss) LoRA-r32 system: voxel step 20k (upd10000), camera step 116k, pixel step 61k.
#
# Each launch only generates episodes whose mp4 is still missing from the output dir, so a crash or
# a restart resumes rather than regenerating everything. After `MAX_BATCH_TRIES` batched attempts,
# any still-missing episodes are retried one at a time (same GPU pair, single episode) so a crash
# pinpoints the exact offending sha256 in the log instead of hiding inside a batch.
#
# Usage: nohup setsid bash rollouts_chzhang.sh <persist_eval|hudnav_eval> <w_v0|wo_v0> <gpuA,gpuB> [batch] \
#          > /data/lsokoudj/eval/logs/<set>_<mode>.log 2>&1 &
set -uo pipefail

SET="${1:?usage: rollouts_chzhang.sh <persist_eval|hudnav_eval> <w_v0|wo_v0> <gpuA,gpuB> [batch]}"
MODE="${2:?mode required: w_v0 or wo_v0}"
GPUS="${3:?comma-separated GPU pair required, e.g. 0,1}"
BATCH="${4:-1}"
MAX_BATCH_TRIES=3

case "$SET" in
  persist_eval|hudnav_eval) ;;
  *) echo "Unknown set '$SET' (expected persist_eval or hudnav_eval)"; exit 1 ;;
esac
case "$MODE" in
  w_v0) INCLUDE_FLAGS="--include-initial-voxel-frame --include-initial-pixel-frame" ;;
  wo_v0) INCLUDE_FLAGS="" ;;
  *) echo "Unknown mode '$MODE' (expected w_v0 or wo_v0)"; exit 1 ;;
esac

cd ~/PERSIST || exit 1
source .venv/bin/activate
export PYTHONPATH=.
export PATH="$HOME/PERSIST/.venv/bin:$PATH"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPUS"

DATASET="/data/lsokoudj/eval/$SET"
P=checkpoints/pretrained
W=outputs/runs_full29k/inference_weights
OUT="/data/lsokoudj/eval/rollouts/$SET/$MODE"
FVD_OUT="/data/lsokoudj/eval/fvd/$SET/$MODE"
mkdir -p "$OUT" "$FVD_OUT"

CKPT_FLAGS="--checkpoints.voxel-encoder $P/voxel-encoder/model.safetensors \
  --checkpoints.voxel-decoder $P/voxel-decoder/model.safetensors \
  --checkpoints.pixel-vae $P/pixel-vae/model.safetensors \
  --checkpoints.voxel-denoiser $W/voxel_lora-r32_upd10000.safetensors \
  --checkpoints.camera-model $W/camera_lora-r32_step116000.safetensors \
  --checkpoints.pixel-denoiser $W/pixel_lora-r32_step61000.safetensors \
  --checkpoints.voxel-class-embedder $P/voxel-class-embedder/model.safetensors"

all_validation_shas() {
  python3 -c "
import pandas as pd
m = pd.read_csv('$DATASET/metadata.csv')
m = m[(m['validation_set'] == True) & (m['level_quality_score'] >= 1.0)]
print(','.join(m['sha256'].tolist()))
"
}

missing_shas() {
  # $1 = comma-separated candidate shas
  python3 -c "
import os
candidates = '$1'.split(',') if '$1' else []
missing = [s for s in candidates if not os.path.exists(os.path.join('$OUT', f'XL_{s}.mp4'))]
print(','.join(missing))
"
}

run_batch() {
  # $1 = comma-separated shas to roll out
  accelerate launch --multi_gpu --num_processes 2 --mixed_precision bf16 --dynamo_backend no \
    -m scripts.run_inference \
    --pipeline-variant XL --dataset-path "$DATASET" --eval-instances "$1" --num-frames 600 \
    $INCLUDE_FLAGS --seed 0 --mixed-precision bf16 --batch-size "$BATCH" --output-dir "$OUT" \
    $CKPT_FLAGS
}

run_single() {
  # $1 = one sha256, on GPU A only (single process, no accelerate multi_gpu)
  gpu_a="${GPUS%%,*}"
  CUDA_VISIBLE_DEVICES="$gpu_a" python -m scripts.run_inference \
    --pipeline-variant XL --dataset-path "$DATASET" --eval-instances "$1" --num-frames 600 \
    $INCLUDE_FLAGS --seed 0 --mixed-precision bf16 --batch-size 1 --output-dir "$OUT" \
    $CKPT_FLAGS
}

ALL=$(all_validation_shas)
TOTAL=$(python3 -c "print(len('$ALL'.split(',')) if '$ALL' else 0)")
echo "=== $SET / $MODE: $TOTAL validation episode(s), $(date) ==="

try=0
while [ "$try" -lt "$MAX_BATCH_TRIES" ]; do
  MISSING=$(missing_shas "$ALL")
  if [ -z "$MISSING" ]; then
    echo "=== all $TOTAL episodes present after $try batch attempt(s) ==="
    break
  fi
  N_MISSING=$(python3 -c "print(len('$MISSING'.split(',')))")
  try=$((try + 1))
  echo "=== batch attempt $try/$MAX_BATCH_TRIES: $N_MISSING missing, $(date) ==="
  run_batch "$MISSING" || echo "=== batch attempt $try failed (continuing, will retry/isolate) ==="
done

MISSING=$(missing_shas "$ALL")
if [ -n "$MISSING" ]; then
  echo "=== $MAX_BATCH_TRIES batch attempts exhausted; isolating remaining episodes one at a time ==="
  FAILED=""
  IFS=',' read -ra REMAINING <<< "$MISSING"
  for sha in "${REMAINING[@]}"; do
    echo "--- isolating $sha, $(date) ---"
    if ! run_single "$sha"; then
      echo "--- FAILED (isolated): $sha ---"
      FAILED="$FAILED,$sha"
    fi
  done
  if [ -n "$FAILED" ]; then
    echo "=== giving up on: ${FAILED#,} -- report this to the user ==="
  fi
fi

FINAL_MISSING=$(missing_shas "$ALL")
N_DONE=$(python3 -c "print($TOTAL - (len('$FINAL_MISSING'.split(',')) if '$FINAL_MISSING' else 0))")
echo "=== rollouts done: $N_DONE/$TOTAL present, $(date) ==="

echo "=== scoring FVD, $(date) ==="
python -m scripts.score_fvd_videos \
  --dataset-path "$DATASET" --rollout-dir "$OUT" --frame-lengths 200 400 600 \
  --output-dir "$FVD_OUT" --set-name "$SET" --mode-name "$MODE"

echo "=== $SET / $MODE all done, $(date) ==="
