#!/bin/bash
# Extract every "train"-split episode of one Luanti archive into datasets/luanti_full29k/raw/.
# "eval" and "unused" episodes are excluded and never written to disk.
# Usage: sbatch extract_full29k.slurm s1nav|hudnav
#SBATCH --job-name=extract-full29k
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/user/lsokoudj/PERSIST/datasets/luanti_full29k/logs/extract_%x_%j.log

set -euo pipefail
module load GCCcore/13.3.0 zstd/1.5.6

SRC=$1
case "$SRC" in
    s1nav)  ARCHIVE=motworld-s1nav-10k.tar.zst ;;
    hudnav) ARCHIVE=motworld-hudnav-20k.tar.zst ;;
    *) echo "unknown source $SRC"; exit 1 ;;
esac

ROOT=/scratch/user/lsokoudj/PERSIST
M=$ROOT/dataset_toolkits/manifests/full29k
OUT=$ROOT/datasets/luanti_full29k/raw

echo "=== $SRC extraction starting $(date) ==="
zstd -dc "$ROOT/data/luanti/$ARCHIVE" | tar -xf - -C "$OUT" --exclude-from="$M/${SRC}_exclude.txt"
echo "=== $SRC extraction done $(date) ==="
echo "EXTRACTION_DONE_EXIT_$?"
