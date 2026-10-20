#!/bin/bash
# Per-frame FID for PERSIST-256 (step_040000's already-generated videos). Stage 1 (GPU feature
# extraction) sharded across the same A100/A40/RTX dynamic work-queue pools used for the main
# eval; Stage 2 (CPU-only Frechet-distance-per-frame) runs after, on many CPU cores.
# Uses account 132688480362 (20,000 SU balance) -- the default account was nearly exhausted by
# today's earlier eval runs.
set -euo pipefail
cd "$(dirname "$0")"

ACCOUNT="132688480362"
DATASET_ROOT="/scratch/user/lsokoudj/.cache/huggingface/hub/datasets--PERSIST-team--persist-eval-sample/snapshots/d3a3bee6560304397093ae66523451b9e0d067f7"
GENERATED_VIDEO_DIR="../outputs/finetune_oasis_luanti_hudnav_1000ep/eval_persist256_step40000/videos"
OUTPUT_DIR="../outputs/finetune_oasis_luanti_hudnav_1000ep/eval_persist256_step40000/per_frame_fid_feats"
SPLIT_ARGS="--all-episodes"
FINAL_CSV="../outputs/finetune_oasis_luanti_hudnav_1000ep/eval_persist256_step40000/fid_per_frame.csv"

mkdir -p "$OUTPUT_DIR"
export DATASET_ROOT GENERATED_VIDEO_DIR OUTPUT_DIR SPLIT_ARGS

# Extraction is ~3.5s/episode (measured) -- these pool sizes give real cross-GPU/cross-node
# parallelism (matching the main eval's pattern) without being wastefully oversized for how
# cheap this stage actually is; time limits are realistic, not padded.
A100_JOB=$(sbatch --parsable --account=$ACCOUNT --gres=gpu:a100:1 --time=00:30:00 --array=0-15%16 \
    --output="$OUTPUT_DIR/slurm_feat_a100_%A_%a.log" run_fid_feat_worker.slurm)
echo "Submitted A100 feature-extraction pool: $A100_JOB"

A40_JOB=$(sbatch --parsable --account=$ACCOUNT --partition=gpu-a40 --gres=gpu:a40:1 --time=00:30:00 --array=0-7%8 \
    --output="$OUTPUT_DIR/slurm_feat_a40_%A_%a.log" run_fid_feat_worker.slurm)
echo "Submitted A40 feature-extraction pool: $A40_JOB"

RTX_JOB=$(sbatch --parsable --account=$ACCOUNT --gres=gpu:rtx:1 --time=00:30:00 --array=0-15%16 \
    --output="$OUTPUT_DIR/slurm_feat_rtx_%A_%a.log" run_fid_feat_worker.slurm)
echo "Submitted RTX feature-extraction pool: $RTX_JOB"

export FEATS_DIR="$OUTPUT_DIR"
export OUTPUT_CSV="$FINAL_CSV"
MERGE_JOB=$(sbatch --parsable --account=$ACCOUNT \
    --dependency=afterany:"$A100_JOB":"$A40_JOB":"$RTX_JOB" \
    --output="$OUTPUT_DIR/slurm_merge_stage2_%j.log" \
    run_fid_merge_stage2.slurm)
echo "Submitted Stage-2 CPU merge job (runs after all 3 pools finish): $MERGE_JOB"

echo "$A100_JOB $A40_JOB $RTX_JOB $MERGE_JOB" > "$OUTPUT_DIR/last_submitted_job_ids.txt"
