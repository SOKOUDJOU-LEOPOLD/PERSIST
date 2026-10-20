#!/bin/bash
# Resolves the latest finetune_oasis_luanti_hudnav_1000ep checkpoint ONCE (here, on the login
# node), then submits THREE worker pools -- A100, A40, RTX -- all pointed at the SAME shared
# work queue (see eval_oasis_finetuned_fvd_shard.py's module docstring: episodes are claimed
# dynamically, not pre-split, so whichever GPU of whatever type frees up next just claims the
# next unclaimed episode). Fast GPUs naturally finish more episodes than slow ones -- no need to
# predict availability or relative speed in advance. T4 is deliberately excluded (16GB, real risk
# of OOM for this model). A final merge job runs only after all three pools finish
# (--dependency=afterok on all three array job IDs).
#
# See run_persist256_worker.slurm for why the checkpoint must be resolved once and exported, not
# re-resolved per worker.
set -euo pipefail
cd "$(dirname "$0")"

CKPT_DIR="../outputs/finetune_oasis_luanti_hudnav_1000ep"
DATASET_ROOT="/scratch/user/lsokoudj/.cache/huggingface/hub/datasets--PERSIST-team--persist-eval-sample/snapshots/d3a3bee6560304397093ae66523451b9e0d067f7"
OUTPUT_DIR="../outputs/finetune_oasis_luanti_hudnav_1000ep/eval_persist256"

read CKPT_PATH CKPT_STEP < <(/scratch/user/lsokoudj/PERSIST/.venv/bin/python resolve_latest_ckpt.py --ckpt-dir "$CKPT_DIR")
echo "Resolved checkpoint: $CKPT_PATH (step $CKPT_STEP)"

mkdir -p "$OUTPUT_DIR/episode_feats" "$OUTPUT_DIR/videos"

export CKPT_PATH DATASET_ROOT OUTPUT_DIR

# Worker-count ceilings are generous, not exact -- claiming is dynamic, so a worker that finds the
# queue already drained (another pool got there first) exits almost immediately. Sized so the
# three pools' ceilings sum to ~256 (the full episode pool), weighted toward A100 (fastest,
# measured) and RTX given more raw slots to compensate for being ~4x slower per episode
# (measured earlier this session on a comparable rollout). A40 availability has fluctuated
# (seen fully allocated at times), so its pool is modest. Concurrency (the %N throttle) is
# deliberately lower than the ceiling for every pool -- this is a shared, contended cluster.
#
# IMPORTANT -- SU budget: this cluster's admission check charges purely by requested
# CPU-hours (cpus-per-task x array-size x time-limit), IDENTICAL regardless of GPU type, and
# checks the array's full worst-case cost upfront against the account balance (learned the hard
# way: a 128-task/24h A100 array was rejected outright, wanting 24,576 SUs, when the account only
# had ~16k). Array size is kept equal to the throttle (no idle "backup" slots, which don't help
# much anyway under dynamic claiming -- a worker just keeps looping until the queue drains or its
# own time limit hits) and time limits are realistic (from Stage B's real measured ~7-8
# min/episode at batch=4, expected faster at batch=8), not padded-for-safety, specifically so the
# worst-case reservation stays small relative to the actual balance.
A100_ARRAY="0-31%32"
A40_ARRAY="0-15%16"
RTX_ARRAY="0-63%16"

# A100 batch-size=8 is measured, not guessed: probe_batch_size_memory.py swept 4/8/12/16 on a
# real A100 and found memory is nowhere near the limit (12.67GB used at batch=16, on a 40GB
# card) but per-episode throughput plateaus right around batch=8 (47.3s/episode at batch=4 vs
# 32.5s at batch=8, then flat through batch=16) -- the GPU is compute-saturated there, so going
# bigger burns more memory for no further speed. A40 reuses the same value by extrapolation
# (same generation, more memory headroom, not independently measured) rather than 4.
A100_JOB=$(sbatch --parsable --gres=gpu:a100:1 --time=03:00:00 --array=$A100_ARRAY \
    --export=ALL,BATCH_SIZE=8 run_persist256_worker.slurm)
echo "Submitted A100 worker pool: $A100_JOB (array $A100_ARRAY)"

# A40 nodes live in a SEPARATE partition ("gpu-a40", g[101-115]) from the generic "gpu" partition
# (a100/rtx/t4 nodes) that run_persist256_worker.slurm's #SBATCH defaults target -- --gres alone
# isn't enough, --partition must be overridden too, or SLURM reports "Requested node
# configuration is not available" (learned the hard way: no node in "gpu" has an a40 GRES).
A40_JOB=$(sbatch --parsable --partition=gpu-a40 --gres=gpu:a40:1 --time=03:00:00 --array=$A40_ARRAY \
    --export=ALL,BATCH_SIZE=8 run_persist256_worker.slurm)
echo "Submitted A40 worker pool: $A40_JOB (array $A40_ARRAY)"

RTX_JOB=$(sbatch --parsable --gres=gpu:rtx:1 --time=03:00:00 --array=$RTX_ARRAY \
    --export=ALL,BATCH_SIZE=2 run_persist256_worker.slurm)
echo "Submitted RTX worker pool: $RTX_JOB (array $RTX_ARRAY)"

# afterany (not afterok): each worker completes episodes independently and writes one atomic
# .npz per episode, so a single failed/preempted/OOM'd task among ~256 doesn't corrupt anything
# -- it just means fewer completed episodes. afterok would block the merge FOREVER if even one
# task across all 3 pools fails, which across this many heterogeneous-GPU tasks is a real risk
# worth avoiding rather than requiring a fully clean run.
export CKPT_STEP
MERGE_JOB_ID=$(sbatch --parsable \
    --dependency=afterany:"$A100_JOB":"$A40_JOB":"$RTX_JOB" \
    run_persist256_merge.slurm)
echo "Submitted merge job (runs after all 3 pools finish, regardless of individual task failures): $MERGE_JOB_ID"

echo "$A100_JOB $A40_JOB $RTX_JOB $MERGE_JOB_ID" > "$OUTPUT_DIR/last_submitted_job_ids.txt"
