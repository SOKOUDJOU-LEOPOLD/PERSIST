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
#SBATCH --signal=B:USR1@900
#SBATCH --output=/scratch/user/lsokoudj/PERSIST/outputs/runs_full29k/logs/%x_%j.log

set -uo pipefail
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
# recipe overrides (defaults = the original full29k run): LR, WARMUP, GRAD_ACC, NOISE_COND (diffusers only),
# RUN_SUFFIX (appended to --run-name, so an experiment gets its own run dir and wandb run),
# LORA_RANK (>0: LoRA fine-tune, diffusers only), RETENTION (latest_and_best | all: keep every checkpoint_<step>)
SUFFIX=${RUN_SUFFIX:+-$RUN_SUFFIX}
DIFFUSION="diffusion:diffusion-forcing ${NOISE_COND:+--diffusion.noise-cond $NOISE_COND}"
LORA="${LORA_RANK:+--lora-rank $LORA_RANK}"
# total_steps is a multiple of i_save so the final step is checkpointed
COMMON="--dataset-path ${DATASET:-datasets/luanti_full29k} --batch-size 2 --total-steps ${TOTAL_STEPS:-20000}
  --learning-rate ${LR:-1e-4} --lr-warmup-steps ${WARMUP:-1000} --grad-accumulation-steps ${GRAD_ACC:-1} --i-log 100 --i-val ${I_VAL:-500} --i-print 100 --i-save ${I_SAVE:-1000}
  --dataloader-workers $WORKERS --val-dataloader-workers 2
  --full-pass-validation --checkpoint-retention ${RETENTION:-latest_and_best} --dataloader-start-method forkserver --no-epoch-boundaries
  --use-wandb --wandb-project persist-luanti-full29k"
ONTHEFLY="--encode-latents-on-the-fly --voxel-encoder-checkpoint $P/voxel-encoder/model.safetensors
  --pixel-vae-checkpoint $P/pixel-vae/model.safetensors"

case "$COMPONENT" in
    voxel)
        SCRIPT=train_scripts/train_diffuser_voxel.py
        ARGS="$COMMON $ONTHEFLY $LORA --run-name full29k-voxel-denoiser-xl$SUFFIX --output-dir $OUT/voxel_diffuser
          --wandb-run-name voxel-denoiser-xl
          --pretrained-checkpoint $P/voxel-denoiser-xl/model.safetensors --pretrained-checkpoint-model-key denoiser
          --denoiser.patch-size 1 --denoiser.gradient-checkpointing $DIFFUSION" ;;
    pixel)
        SCRIPT=train_scripts/train_diffuser_pixel.py
        ARGS="$COMMON $ONTHEFLY $LORA --run-name full29k-pixel-denoiser$SUFFIX --output-dir $OUT/pixel_diffuser
          --wandb-run-name pixel-denoiser
          --voxel-decoder-checkpoint $P/voxel-decoder/model.safetensors --voxel-decoder-batch-size 2
          --pretrained-checkpoint $P/pixel-denoiser/model.safetensors --pretrained-checkpoint-model-key denoiser
          --pretrained-checkpoint-extra $P/voxel-class-embedder/model.safetensors
          --pretrained-checkpoint-extra-model-key voxel_class_embedder
          --denoiser.gradient-checkpointing $DIFFUSION" ;;
    camera)
        SCRIPT=train_scripts/train_transformer_camera.py
        # the camera script requires i_log % i_val == 0
        ARGS="${COMMON/--i-log 100/--i-log 500} --run-name full29k-camera-model$SUFFIX --output-dir $OUT/camera_transformer
          --wandb-run-name camera-model
          --pretrained-checkpoint $P/camera-model/model.safetensors --pretrained-checkpoint-model-key camera
          --model.cam-rotation-output-rep delta_angle --model.cam-fov-output-rep delta" ;;
    *) echo "unknown component $COMPONENT"; exit 1 ;;
esac

echo "=== $COMPONENT: $SLURM_NNODES node(s) x $GPUS_PER_NODE GPU(s) = $NUM_PROCESSES processes, main $MAIN_IP, $(date) ==="

# ---- Self-healing: an unfinished run resubmits itself with the same layout and resumes from its newest
# checkpoint (--resume auto). Triggers: training exits early (crash / killed hung step), the time-limit warning
# (USR1, see --signal), or a bad /dev/shm node at preflight. The new job depends on this one (afterany), so the
# two never train at once. At most MAX_RESUBMITS automatic resubmits per chain.
SELF=/scratch/user/lsokoudj/PERSIST/dataset_toolkits/manifests/full29k/train_full29k.slurm
MAX_RESUBMITS=${MAX_RESUBMITS:-5}
LOGFILE=$(scontrol show job $SLURM_JOB_ID | grep -oP 'StdOut=\K\S+')
RESUBMITTED=0
resubmit() {  # $1 = reason, $2 = extra nodes to exclude (comma list, optional)
    [ "$RESUBMITTED" = 1 ] && return
    local n=${RESUBMIT_COUNT:-0}
    if [ "$n" -ge "$MAX_RESUBMITS" ]; then
        echo "=== NOT resubmitting ($1): reached MAX_RESUBMITS=$MAX_RESUBMITS ==="; return
    fi
    local exclude
    exclude=$(echo "${EXCLUDE_NODES:+$EXCLUDE_NODES,}${2:-}" | tr ',' '\n' | grep -v '^$' | sort -u | paste -sd,)
    RESUBMIT_COUNT=$((n + 1)) EXCLUDE_NODES=$exclude sbatch --dependency=afterany:$SLURM_JOB_ID \
        --nodes=$SLURM_NNODES --gres=$(scontrol show job $SLURM_JOB_ID | grep -oP 'TresPerNode=\K\S+' | sed 's,gres/,,g') \
        --job-name=$SLURM_JOB_NAME --cpus-per-task=$SLURM_CPUS_PER_TASK --mem=${SLURM_MEM_PER_NODE}M \
        --time=$(scontrol show job $SLURM_JOB_ID | grep -oP 'TimeLimit=\K\S+') ${exclude:+--exclude=$exclude} \
        --output="$(echo $LOGFILE | sed "s/$SLURM_JOB_ID/%j/")" "$SELF" "$COMPONENT"
    RESUBMITTED=1
    echo "=== Resubmitted ($1), attempt $((n + 1))/$MAX_RESUBMITS${exclude:+, excluding $exclude} ==="
}
trap 'echo "=== time limit approaching ==="; resubmit "time limit"' USR1

# Preflight: NCCL and dataloader workers need /dev/shm, which other users' leftovers can fill on shared nodes.
MIN_SHM_GB=${MIN_SHM_GB:-16}
BAD_NODES=$(srun --ntasks-per-node=1 bash -c \
    'avail=$(df --output=avail -BG /dev/shm | tail -1 | tr -dc 0-9); [ "$avail" -lt '$MIN_SHM_GB' ] && hostname' || true)
BAD_NODES=$(echo $BAD_NODES | tr ' ' ',')
if [ -n "$BAD_NODES" ]; then
    echo "Preflight: /dev/shm under ${MIN_SHM_GB} GB on $BAD_NODES"
    resubmit "bad /dev/shm" "$BAD_NODES"
    exit 1
fi

# Hang watchdog: no log output for HANG_MINUTES (training prints every step; a full validation pass or a
# checkpoint save takes a few minutes) -> cancel the training step, which then counts as an early exit below.
HANG_MINUTES=${HANG_MINUTES:-30}
( last=-1; idle=0
  while sleep 60; do
      size=$(stat -c %s "$LOGFILE" 2>/dev/null || echo 0)
      if [ "$size" = "$last" ]; then idle=$((idle + 1)); else idle=0; fi; last=$size
      if [ $idle -ge $HANG_MINUTES ]; then
          echo "=== WATCHDOG: no progress for $HANG_MINUTES min, cancelling training step ==="
          scancel --signal=KILL "$SLURM_JOB_ID.$(squeue -h -s -j $SLURM_JOB_ID -o %i | grep -oP '\.\K[0-9]+$' | tail -1)"; break
      fi
  done ) &
WATCHDOG=$!

# run in the background and wait, so the USR1 trap can fire while training runs
srun --kill-on-bad-exit=1 bash -c "accelerate launch --mixed_precision bf16 --dynamo_backend no \
    --num_machines $SLURM_NNODES --num_processes $NUM_PROCESSES --machine_rank \$SLURM_NODEID \
    --main_process_ip $MAIN_IP --main_process_port $((20000 + SLURM_JOB_ID % 20000)) $( [ $NUM_PROCESSES -gt 1 ] && echo --multi_gpu ) \
    $SCRIPT $(echo $ARGS)" &
SRUN=$!
RC=0
while true; do  # wait returns early when the USR1 trap fires; keep waiting for training itself
    wait $SRUN && RC=0 || RC=$?
    kill -0 $SRUN 2>/dev/null || break
done
{ kill $WATCHDOG; wait $WATCHDOG; } 2>/dev/null || true

RUN_NAME=$(echo "$ARGS" | grep -oP -- "--run-name \K\S+")
FINAL_CKPT=$(ls -d $OUT/*/$RUN_NAME-uid-*/checkpoints/checkpoint_${TOTAL_STEPS:-20000} 2>/dev/null | head -1 || true)
if [ -n "$FINAL_CKPT" ]; then
    echo "=== $COMPONENT finished all ${TOTAL_STEPS:-20000} steps ($FINAL_CKPT) $(date) ==="
else
    echo "=== $COMPONENT stopped early (exit $RC) $(date) ==="
    resubmit "stopped early, exit $RC"
fi
