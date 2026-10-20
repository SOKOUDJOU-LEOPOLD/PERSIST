#!/bin/bash
# Same dynamic work-queue / multi-GPU-type pool design as submit_persist256_eval.sh, but for
# checkpoint step_040000 on motworld-hudnav-20k's own "eval" split (200 episodes, in-domain --
# unlike persist-eval-sample this is the SAME Luanti-hudnav environment the checkpoint was
# trained on, drawn from the raw dataset's own held-out-episode labeling, distinct from the
# smaller oasis_luanti_hudnav_1000ep.json train_eval/val/test splits used during training).
set -euo pipefail
cd "$(dirname "$0")"

CKPT_PATH="/scratch/user/lsokoudj/PERSIST/outputs/finetune_oasis_luanti_hudnav_1000ep/step_040000.pt"
CKPT_STEP=40000
DATASET_ROOT="/scratch/user/lsokoudj/PERSIST/data/luanti/extracted_hudnav/OpenWorldCreativeHud-v0"
OUTPUT_DIR="../outputs/finetune_oasis_luanti_hudnav_1000ep/eval_hudnav20k_step40000"
SPLIT_ARGS="--split-path ../outputs/motworld_hudnav_20k_eval_split.json --split eval"
SPLIT_LABEL="hudnav20k_eval_200"
RESULT_JSON="eval_fvd_fid_hudnav20k_step40000.json"

mkdir -p "$OUTPUT_DIR/episode_feats" "$OUTPUT_DIR/videos"
export CKPT_PATH CKPT_STEP DATASET_ROOT OUTPUT_DIR SPLIT_ARGS SPLIT_LABEL RESULT_JSON

# Ceilings sized for a 200-episode pool (vs 256 for persist256) -- same reasoning as
# submit_persist256_eval.sh: array size == throttle (no idle backup slots), realistic time
# limits (not padded), A100 gets batch=8 (measured safe+fast this session), A40 same by
# extrapolation, RTX stays conservative at batch=2.
A100_JOB=$(sbatch --parsable --gres=gpu:a100:1 --time=03:00:00 --array=0-24%25 \
    --output="$OUTPUT_DIR/slurm_worker_a100_%A_%a.log" \
    --export=ALL,BATCH_SIZE=8 run_eval_worker.slurm)
echo "Submitted A100 worker pool: $A100_JOB"

A40_JOB=$(sbatch --parsable --partition=gpu-a40 --gres=gpu:a40:1 --time=03:00:00 --array=0-11%12 \
    --output="$OUTPUT_DIR/slurm_worker_a40_%A_%a.log" \
    --export=ALL,BATCH_SIZE=8 run_eval_worker.slurm)
echo "Submitted A40 worker pool: $A40_JOB"

RTX_JOB=$(sbatch --parsable --gres=gpu:rtx:1 --time=03:00:00 --array=0-11%12 \
    --output="$OUTPUT_DIR/slurm_worker_rtx_%A_%a.log" \
    --export=ALL,BATCH_SIZE=2 run_eval_worker.slurm)
echo "Submitted RTX worker pool: $RTX_JOB"

MERGE_JOB=$(sbatch --parsable \
    --dependency=afterany:"$A100_JOB":"$A40_JOB":"$RTX_JOB" \
    --output="$OUTPUT_DIR/slurm_merge_%j.log" \
    run_eval_merge.slurm)
echo "Submitted merge job (runs after all 3 pools finish): $MERGE_JOB"

echo "$A100_JOB $A40_JOB $RTX_JOB $MERGE_JOB" > "$OUTPUT_DIR/last_submitted_job_ids.txt"
