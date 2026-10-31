#!/bin/bash
# Build datasets/hudnav_eval: the 200 held-out "eval" episodes of motworld-hudnav-20k (none are in luanti_full29k),
# extracted from the local archive and preprocessed like luanti_full29k (seeded 2138-class dict, check, voxelize),
# with every surviving episode marked as validation so run_inference / FVD scoring pick them all up.
# 200 episodes are few enough to run every stage in one job.
# Usage: sbatch dataset_toolkits/manifests/eval_fvd/prepare_hudnav_eval.slurm
#SBATCH --job-name=prep-hudnav-eval
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/user/lsokoudj/PERSIST/datasets/hudnav_eval/logs/prep_%j.log

set -euo pipefail
cd /scratch/user/lsokoudj/PERSIST
module load GCCcore/13.3.0 zstd/1.5.6
source .venv/bin/activate
export PYTHONPATH=.
DS=datasets/hudnav_eval
SPLITS=data/luanti/motworld-hudnav-20k.splits.json
mkdir -p $DS/raw

echo "=== extract $(date) ==="
python - <<EOF
import json
s = json.load(open("$SPLITS"))
names = [e["name"] for e in s["episodes"] if e["split"] == "eval"]
open("$DS/eval_dirs.txt", "w").write("".join(f"{s['environment']}/{n}/\n" for n in names))
print(len(names), "eval episodes")
EOF
zstd -dc data/luanti/motworld-hudnav-20k.tar.zst | tar -xf - -C $DS/raw --files-from=$DS/eval_dirs.txt
echo "extracted $(ls $DS/raw/OpenWorldCreativeHud-v0 | wc -l) episode dirs"

cp dataset_toolkits/default_mt_voxel_classdict.json $DS/mt_voxel_classdict.json
W=$SLURM_CPUS_PER_TASK
echo "=== check $(date) ===";    python dataset_toolkits/build_metadata.py --output_dir $DS
python dataset_toolkits/process_mt_data.py --output_dir $DS --max_workers $W --check_conditions_only --overwrite
echo "=== voxelize $(date) ==="; python dataset_toolkits/build_metadata.py --output_dir $DS
python dataset_toolkits/process_mt_data.py --output_dir $DS --max_workers $W
python dataset_toolkits/build_metadata.py --output_dir $DS

echo "=== validation split $(date) ==="
python - <<EOF
import pandas as pd
m = pd.read_csv("$DS/metadata.csv")
v = m[m.voxelized]
open("$DS/val_sha256.txt", "w").write("\n".join(v.sha256) + "\n")
print(f"{len(v)} of {len(m)} episodes voxelized; rejected: {sorted(m[~m.voxelized].local_path.str.rsplit('/', n=1).str[1])}")
EOF
python dataset_toolkits/assign_validation_split.py --dataset_dir $DS --val_sha256_file $DS/val_sha256.txt
echo "=== done $(date) ==="
