#!/bin/bash
# Generic merge step, like run_persist256_merge.slurm but parameterized for any output dir/label.
#SBATCH --job-name=eval-merge
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

: "${CKPT_PATH:?CKPT_PATH must be exported by the submitting script}"
: "${CKPT_STEP:?CKPT_STEP must be exported by the submitting script}"
: "${OUTPUT_DIR:?OUTPUT_DIR must be exported by the submitting script}"
: "${SPLIT_LABEL:?SPLIT_LABEL must be exported by the submitting script}"
: "${RESULT_JSON:?RESULT_JSON must be exported by the submitting script}"

export PYTHONUNBUFFERED=1
source /scratch/user/lsokoudj/PERSIST/.venv/bin/activate
cd /scratch/user/lsokoudj/PERSIST/baselines

python reap_stale_claims.py --output-dir "$OUTPUT_DIR/episode_feats"

python merge_oasis_fvd_shards.py \
    --episode-feats-glob "$OUTPUT_DIR/episode_feats/*.npz" \
    --frame-lengths 200 400 600 \
    --checkpoint "$CKPT_PATH" \
    --checkpoint-step "$CKPT_STEP" \
    --split "$SPLIT_LABEL" \
    --output-json "$OUTPUT_DIR/$RESULT_JSON"
