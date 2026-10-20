#!/bin/bash
#SBATCH --job-name=persist256-merge
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=../outputs/finetune_oasis_luanti_hudnav_1000ep/eval_persist256/slurm_merge_%j.log

# CPU-only (scipy sqrtm on pooled feature covariance) -- submitted with
# --dependency=afterany:<3 worker pool job IDs> by submit_persist256_eval.sh, so this runs once
# all 3 GPU-type pools finish, regardless of whether every individual worker task succeeded.
: "${CKPT_PATH:?CKPT_PATH must be exported by the submitting script}"
: "${CKPT_STEP:?CKPT_STEP must be exported by the submitting script}"
: "${OUTPUT_DIR:?OUTPUT_DIR must be exported by the submitting script}"

export PYTHONUNBUFFERED=1
source /scratch/user/lsokoudj/PERSIST/.venv/bin/activate
cd /scratch/user/lsokoudj/PERSIST/baselines

# Clears any claim left behind by a worker that died mid-episode (OOM/preemption/walltime) --
# without this, that episode's claim marker would make it look permanently "in progress" to any
# future retry run, even though nothing is actually working on it. Does not affect completed
# episodes.
python reap_stale_claims.py --output-dir "$OUTPUT_DIR/episode_feats"

python merge_oasis_fvd_shards.py \
    --episode-feats-glob "$OUTPUT_DIR/episode_feats/*.npz" \
    --frame-lengths 200 400 600 \
    --checkpoint "$CKPT_PATH" \
    --checkpoint-step "$CKPT_STEP" \
    --split all_256 \
    --output-json "$OUTPUT_DIR/eval_fvd_fid_persist256.json"
