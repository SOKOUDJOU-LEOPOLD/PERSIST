#!/bin/bash
# Post-voxelization steps for datasets/luanti_full29k, chained by Slurm dependencies.
# Usage: sbatch postprocess_full29k.slurm merge     # merge voxelized_<rank>.csv records into metadata.csv
#        sbatch postprocess_full29k.slurm finalize  # merge again, assign the 400-episode val split, cropped voxels
#SBATCH --job-name=post-full29k
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/user/lsokoudj/PERSIST/datasets/luanti_full29k/logs/post_%x_%j.log

set -euo pipefail
cd /scratch/user/lsokoudj/PERSIST
source .venv/bin/activate
export PYTHONPATH=.
DS=${DS:-datasets/luanti_full29k}  # other datasets: DS=..., VAL=... and sbatch --output=<DS>/logs/...
M=dataset_toolkits/manifests/full29k
VAL=${VAL:-$M/val_sha256.txt}

summary() {
    python - <<EOF
import pandas as pd
m = pd.read_csv("$DS/metadata.csv")
m["source"] = m.local_path.str.contains("Hud").map({True: "hudnav", False: "s1nav"})
m["rejected_classdict"] = m.preprocessed & (m.level_quality_score == 0)
m["save_failed"] = m.preprocessed & ~m.voxelized & (m.level_quality_score == 10)
print(m.groupby("source")[["voxelized", "rejected_classdict", "save_failed", "validation_set"]].sum())
EOF
}

echo "=== $1 starting $(date) ==="
python dataset_toolkits/build_metadata.py --output_dir $DS
summary
if [ "$1" = "finalize" ]; then
    python dataset_toolkits/assign_validation_split.py --dataset_dir $DS \
        --val_sha256_file $VAL --replace_rejected
    python dataset_toolkits/process_mt_data.py --output_dir $DS --compute_cropped_voxels \
        --max_workers $SLURM_CPUS_PER_TASK
    summary
fi
echo "=== $1 done $(date) ==="
