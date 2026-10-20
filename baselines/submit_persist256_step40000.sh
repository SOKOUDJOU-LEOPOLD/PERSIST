#!/bin/bash
# Same PERSIST-256 (persist-eval-sample) evaluation as submit_persist256_eval.sh, but pinned to
# checkpoint step_040000 explicitly (not "whatever's latest") and writing to a SEPARATE,
# checkpoint-specific output dir so the existing step_027500 result stays intact for comparison.
# Reuses the generalized run_eval_worker.slurm/run_eval_merge.slurm (built for the hudnav20k
# eval) with --all-episodes instead of a split file, same as the original persist256 run.
set -euo pipefail
cd "$(dirname "$0")"

CKPT_PATH="/scratch/user/lsokoudj/PERSIST/outputs/finetune_oasis_luanti_hudnav_1000ep/step_040000.pt"
CKPT_STEP=40000
DATASET_ROOT="/scratch/user/lsokoudj/.cache/huggingface/hub/datasets--PERSIST-team--persist-eval-sample/snapshots/d3a3bee6560304397093ae66523451b9e0d067f7"
OUTPUT_DIR="../outputs/finetune_oasis_luanti_hudnav_1000ep/eval_persist256_step40000"
SPLIT_ARGS="--all-episodes"
SPLIT_LABEL="all_256"
RESULT_JSON="eval_fvd_fid_persist256_step40000.json"

mkdir -p "$OUTPUT_DIR/episode_feats" "$OUTPUT_DIR/videos"
export CKPT_PATH CKPT_STEP DATASET_ROOT OUTPUT_DIR SPLIT_ARGS SPLIT_LABEL RESULT_JSON

# Same pool sizing/timing as the original persist256 run (proven this session: A100 batch=8,
# realistic 3h time limits, A40 needs its own partition).
A100_JOB=$(sbatch --parsable --gres=gpu:a100:1 --time=03:00:00 --array=0-31%32 \
    --output="$OUTPUT_DIR/slurm_worker_a100_%A_%a.log" \
    --export=ALL,BATCH_SIZE=8 run_eval_worker.slurm)
echo "Submitted A100 worker pool: $A100_JOB"

A40_JOB=$(sbatch --parsable --partition=gpu-a40 --gres=gpu:a40:1 --time=03:00:00 --array=0-15%16 \
    --output="$OUTPUT_DIR/slurm_worker_a40_%A_%a.log" \
    --export=ALL,BATCH_SIZE=8 run_eval_worker.slurm)
echo "Submitted A40 worker pool: $A40_JOB"

RTX_JOB=$(sbatch --parsable --gres=gpu:rtx:1 --time=03:00:00 --array=0-63%16 \
    --output="$OUTPUT_DIR/slurm_worker_rtx_%A_%a.log" \
    --export=ALL,BATCH_SIZE=2 run_eval_worker.slurm)
echo "Submitted RTX worker pool: $RTX_JOB"

MERGE_JOB=$(sbatch --parsable \
    --dependency=afterany:"$A100_JOB":"$A40_JOB":"$RTX_JOB" \
    --output="$OUTPUT_DIR/slurm_merge_%j.log" \
    run_eval_merge.slurm)
echo "Submitted merge job (runs after all 3 pools finish): $MERGE_JOB"

echo "$A100_JOB $A40_JOB $RTX_JOB $MERGE_JOB" > "$OUTPUT_DIR/last_submitted_job_ids.txt"
