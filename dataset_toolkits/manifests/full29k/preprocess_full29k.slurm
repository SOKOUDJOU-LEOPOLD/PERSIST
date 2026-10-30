#!/bin/bash
# One shard of a process_mt_data.py stage over datasets/luanti_full29k, as a Slurm array task.
# Usage: sbatch --array=0-19 preprocess_full29k.slurm check   # stage 1: validity checks (preprocessed_<rank>.csv)
#        sbatch --array=0-19 preprocess_full29k.slurm voxelize # stage 2: voxel classes vs. the seeded 2138-class dict
# Between stages run: python dataset_toolkits/build_metadata.py --output_dir datasets/luanti_full29k
#SBATCH --job-name=prep-full29k
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/user/lsokoudj/PERSIST/datasets/luanti_full29k/logs/prep_%x_%A_%a.log

set -euo pipefail
cd /scratch/user/lsokoudj/PERSIST
source .venv/bin/activate
export PYTHONPATH=.

STAGE=$1
DS=${DS:-datasets/luanti_full29k}  # other datasets: DS=... and sbatch --output=<DS>/logs/...
RANK=$SLURM_ARRAY_TASK_ID
WORLD=$SLURM_ARRAY_TASK_COUNT

case "$STAGE" in
    check)    FLAGS="--check_conditions_only --overwrite" ;;
    voxelize) FLAGS="" ;;
    *) echo "unknown stage $STAGE"; exit 1 ;;
esac

echo "=== $STAGE rank $RANK/$WORLD starting $(date) ==="
python dataset_toolkits/process_mt_data.py --output_dir $DS --rank $RANK --world_size $WORLD \
    --max_workers $SLURM_CPUS_PER_TASK $FLAGS
echo "=== $STAGE rank $RANK/$WORLD done $(date) ==="
