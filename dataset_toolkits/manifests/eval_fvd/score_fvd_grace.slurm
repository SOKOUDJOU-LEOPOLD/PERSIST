#!/bin/bash
# FVD at 200/400/600 frames for one set and mode of rollouts_grace.slurm.
# Usage: sbatch [--dependency=afterok:<array job id>] score_fvd_grace.slurm <persist_eval|hudnav_eval> <w_v0|wo_v0>
#SBATCH --job-name=fvd-score
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/user/lsokoudj/PERSIST/outputs/eval_fvd/logs/%x_%j.log

set -uo pipefail
cd /scratch/user/lsokoudj/PERSIST
module load GCC/13.2.0 CUDA/12.6.0
source .venv/bin/activate
export LD_PRELOAD=$EBROOTGCCCORE/lib64/libstdc++.so.6
export PYTHONPATH=.

SET=${1:?set}; MODE=${2:?mode}
python -m scripts.score_fvd_videos --dataset-path datasets/$SET --rollout-dir outputs/eval_fvd/rollouts/$SET/$MODE \
    --frame-lengths 200 400 600 --output-dir outputs/eval_fvd/fvd/$SET/$MODE --set-name $SET --mode-name $MODE
