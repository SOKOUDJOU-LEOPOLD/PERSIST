#!/bin/bash
# Stage 2 (CPU-only, multiprocessing across cores -- see merge_per_frame_fid.py's docstring for
# why this can't use GPUs). FEATS_DIR/NUM_FRAMES/OUTPUT_CSV exported by the submitting script.
#SBATCH --job-name=fid-merge-stage2
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=00:30:00

: "${FEATS_DIR:?FEATS_DIR must be exported by the submitting script}"
: "${OUTPUT_CSV:?OUTPUT_CSV must be exported by the submitting script}"

export PYTHONUNBUFFERED=1
source /scratch/user/lsokoudj/PERSIST/.venv/bin/activate
cd /scratch/user/lsokoudj/PERSIST/baselines

python merge_per_frame_fid.py \
    --feats-dir "$FEATS_DIR" \
    --num-frames 600 \
    --num-workers 48 \
    --output-csv "$OUTPUT_CSV"
