#!/bin/bash
# Fine-tune one PERSIST component on datasets/luanti_full29k from the official PERSIST-team weights,
# data-parallel over all GPUs of all allocated nodes. Resubmitting with the same component resumes
# from the newest checkpoint_<step> (same --run-name, --resume auto).
# Usage (GPU layout is chosen at submit time):
#   sbatch --nodes=8 --gres=gpu:a100:1 --job-name=train-voxel  train_full29k.slurm voxel
#   sbatch --nodes=2 --gres=gpu:a40:3  --job-name=train-pixel  train_full29k.slurm pixel
#   sbatch --nodes=1 --gres=gpu:a100:2 --job-name=train-camera train_full29k.slurm camera
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=180G
#SBATCH --time=4-00:00:00
#SBATCH --output=/scratch/user/lsokoudj/PERSIST/outputs/runs_full29k/logs/%x_%j.log

set -euo pipefail
cd /scratch/user/lsokoudj/PERSIST
module load GCC/13.2.0 CUDA/12.6.0
source .venv/bin/activate
# the Anaconda-based python ships an older libstdc++ than the GCC 13 that JIT-builds nvdiffrast
export LD_PRELOAD=$EBROOTGCCCORE/lib64/libstdc++.so.6
export PYTHONPATH=.
export LOGLEVEL=INFO  # torch.distributed.elastic logs rendezvous progress
# compute nodes have no internet: log wandb offline, sync_wandb_full29k.sh on the login node uploads it
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${WANDB_DIR:-/scratch/user/lsokoudj/PERSIST/outputs/runs_full29k}

COMPONENT=$1
GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
NUM_PROCESSES=$((SLURM_NNODES * GPUS_PER_NODE))
MAIN_IP=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
WORKERS=$((SLURM_CPUS_PER_TASK / GPUS_PER_NODE - 2))

P=checkpoints/pretrained
OUT=${OUT_DIR:-outputs/runs_full29k}  # env overrides (OUT_DIR, DATASET, TOTAL_STEPS, I_VAL, I_SAVE) are for short tests only
# total_steps is a multiple of i_save so the final step is checkpointed
COMMON="--dataset-path ${DATASET:-datasets/luanti_full29k} --batch-size 2 --total-steps ${TOTAL_STEPS:-20000}
  --lr-warmup-steps 1000 --i-log 100 --i-val ${I_VAL:-500} --i-print 100 --i-save ${I_SAVE:-1000}
  --dataloader-workers $WORKERS --val-dataloader-workers 2
  --full-pass-validation --checkpoint-retention latest_and_best --dataloader-start-method forkserver --no-epoch-boundaries
  --use-wandb --wandb-project persist-luanti-full29k"
ONTHEFLY="--encode-latents-on-the-fly --voxel-encoder-checkpoint $P/voxel-encoder/model.safetensors
  --pixel-vae-checkpoint $P/pixel-vae/model.safetensors"

case "$COMPONENT" in
    voxel)
        SCRIPT=train_scripts/train_diffuser_voxel.py
        ARGS="$COMMON $ONTHEFLY --run-name full29k-voxel-denoiser-xl --output-dir $OUT/voxel_diffuser
          --wandb-run-name voxel-denoiser-xl
          --pretrained-checkpoint $P/voxel-denoiser-xl/model.safetensors --pretrained-checkpoint-model-key denoiser
          --denoiser.patch-size 1 --denoiser.gradient-checkpointing diffusion:diffusion-forcing" ;;
    pixel)
        SCRIPT=train_scripts/train_diffuser_pixel.py
        ARGS="$COMMON $ONTHEFLY --run-name full29k-pixel-denoiser --output-dir $OUT/pixel_diffuser
          --wandb-run-name pixel-denoiser
          --voxel-decoder-checkpoint $P/voxel-decoder/model.safetensors --voxel-decoder-batch-size 2
          --pretrained-checkpoint $P/pixel-denoiser/model.safetensors --pretrained-checkpoint-model-key denoiser
          --pretrained-checkpoint-extra $P/voxel-class-embedder/model.safetensors
          --pretrained-checkpoint-extra-model-key voxel_class_embedder
          --denoiser.gradient-checkpointing diffusion:diffusion-forcing" ;;
    camera)
        SCRIPT=train_scripts/train_transformer_camera.py
        # the camera script requires i_log % i_val == 0
        ARGS="${COMMON/--i-log 100/--i-log 500} --run-name full29k-camera-model --output-dir $OUT/camera_transformer
          --wandb-run-name camera-model
          --pretrained-checkpoint $P/camera-model/model.safetensors --pretrained-checkpoint-model-key camera
          --model.cam-rotation-output-rep delta_angle --model.cam-fov-output-rep delta" ;;
    *) echo "unknown component $COMPONENT"; exit 1 ;;
esac

echo "=== $COMPONENT: $SLURM_NNODES node(s) x $GPUS_PER_NODE GPU(s) = $NUM_PROCESSES processes, main $MAIN_IP, $(date) ==="

# Preflight: NCCL and dataloader workers need /dev/shm, which other users' leftovers can fill on shared nodes.
# If any node is short, resubmit this same job with those nodes excluded (up to 3 times) instead of failing later.
MIN_SHM_GB=${MIN_SHM_GB:-16}
BAD_NODES=$(srun --ntasks-per-node=1 bash -c \
    'avail=$(df --output=avail -BG /dev/shm | tail -1 | tr -dc 0-9); [ "$avail" -lt '$MIN_SHM_GB' ] && hostname' || true)
BAD_NODES=$(echo $BAD_NODES | tr ' ' ',')
if [ -n "$BAD_NODES" ]; then
    ATTEMPT=${PREFLIGHT_ATTEMPT:-0}
    EXCLUDE=$(echo "${PREFLIGHT_EXCLUDE:+$PREFLIGHT_EXCLUDE,}$BAD_NODES" | tr ',' '\n' | sort -u | paste -sd,)
    echo "Preflight: /dev/shm under ${MIN_SHM_GB} GB on $BAD_NODES (attempt $ATTEMPT)"
    if [ "$ATTEMPT" -lt 3 ]; then
        PREFLIGHT_ATTEMPT=$((ATTEMPT + 1)) PREFLIGHT_EXCLUDE=$EXCLUDE sbatch --exclude="$EXCLUDE" \
            --nodes=$SLURM_NNODES --gres=$(scontrol show job $SLURM_JOB_ID | grep -oP 'TresPerNode=\K\S+' | sed 's,gres/,,g') \
            --job-name=$SLURM_JOB_NAME --cpus-per-task=$SLURM_CPUS_PER_TASK --mem=${SLURM_MEM_PER_NODE}M \
            --output="$(scontrol show job $SLURM_JOB_ID | grep -oP 'StdOut=\K\S+' | sed "s/$SLURM_JOB_ID/%j/")" \
            "$0" "$@"
        echo "Resubmitted excluding $EXCLUDE"
    fi
    exit 1
fi

srun --kill-on-bad-exit=1 bash -c "accelerate launch --mixed_precision bf16 --dynamo_backend no \
    --num_machines $SLURM_NNODES --num_processes $NUM_PROCESSES --machine_rank \$SLURM_NODEID \
    --main_process_ip $MAIN_IP --main_process_port $((20000 + SLURM_JOB_ID % 20000)) $( [ $NUM_PROCESSES -gt 1 ] && echo --multi_gpu ) \
    $SCRIPT $(echo $ARGS)"
echo "=== $COMPONENT finished $(date) ==="
