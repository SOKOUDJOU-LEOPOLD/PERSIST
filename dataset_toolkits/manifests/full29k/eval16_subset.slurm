#!/bin/bash
# Same rollouts as eval16_rollouts.slurm, for a subset of (config, episode) pairs on ONE GPU, so an evaluation
# can be spread over whatever single GPUs are free (e.g. A40s). Output paths match eval16_rollouts.slurm.
# Usage: sbatch --partition=gpu-a40 --gres=gpu:a40:1 eval16_subset.slurm "<config>:<first>-<last> [<config>:<first>-<last> ...]"
#   episodes are 1-based line numbers in eval16_sha256.txt
#SBATCH --job-name=eval16-subset
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/user/lsokoudj/PERSIST/outputs/runs_full29k/logs/%x_%j.log

set -uo pipefail
cd /scratch/user/lsokoudj/PERSIST
module load GCC/13.2.0 CUDA/12.6.0
source .venv/bin/activate
export LD_PRELOAD=$EBROOTGCCCORE/lib64/libstdc++.so.6
export PYTHONPATH=.

P=checkpoints/pretrained
R=outputs/runs_full29k
EPS=dataset_toolkits/manifests/full29k/eval16_sha256.txt
for item in $*; do  # each argument (or space-separated word) is one <config>:<first>-<last> item
    CONFIG=${item%%:*}; RANGE=${item##*:}
    VOXEL=$R/inference_weights/voxel_$CONFIG.safetensors
    [ -f "$VOXEL" ] || { echo "missing $VOXEL"; continue; }
    EPISODES=$(sed -n "${RANGE%-*},${RANGE#*-}p" $EPS | paste -sd,)
    echo "=== $CONFIG episodes $RANGE, $(date) ==="
    python -m scripts.run_inference \
        --pipeline-variant XL --dataset-path datasets/luanti_full29k --eval-instances "$EPISODES" --num-frames 300 \
        --include-initial-pixel-frame --include-initial-voxel-frame \
        --seed 0 --mixed-precision bf16 --output-dir $R/eval16/300f/$CONFIG \
        --checkpoints.voxel-encoder $P/voxel-encoder/model.safetensors --checkpoints.voxel-decoder $P/voxel-decoder/model.safetensors \
        --checkpoints.pixel-vae $P/pixel-vae/model.safetensors --checkpoints.camera-model $P/camera-model/model.safetensors \
        --checkpoints.pixel-denoiser $P/pixel-denoiser/model.safetensors --checkpoints.voxel-class-embedder $P/voxel-class-embedder/model.safetensors \
        --checkpoints.voxel-denoiser $VOXEL || echo "=== FAILED: $CONFIG $RANGE ==="
done
echo "=== subset done, $(date) ==="
