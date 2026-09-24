#!/bin/bash
# Per-GPU task wrapper for run_multinode.slurm: `srun` starts one copy of this per GPU (on any mix of
# nodes / GPU types), and it turns Slurm's task variables into the env:// variables that
# torch.distributed / Accelerator() read -- so finetune_oasis.py itself needs no launcher code.
# MASTER_ADDR / MASTER_PORT are exported by run_multinode.slurm before srun.
export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NTASKS
export LOCAL_WORLD_SIZE=$SLURM_NTASKS_PER_NODE

# With --gpus-per-task=1 Slurm sets CUDA_VISIBLE_DEVICES to just this task's GPU, which torch then
# sees as cuda:0. If the task instead sees several GPUs, index by the task's slot on the node.
# (Counted from CUDA_VISIBLE_DEVICES, not nvidia-smi -- nvidia-smi ignores that variable.)
IFS=',' read -ra VISIBLE <<< "$CUDA_VISIBLE_DEVICES"
if [ ${#VISIBLE[@]} -eq 1 ]; then
    export LOCAL_RANK=0
else
    export LOCAL_RANK=$SLURM_LOCALID
fi

echo "[rank $RANK/$WORLD_SIZE] host=$(hostname -s) local_rank=$LOCAL_RANK" \
     "gpu=$(nvidia-smi -i ${VISIBLE[$LOCAL_RANK]} --query-gpu=name --format=csv,noheader 2>/dev/null)"
exec python finetune_oasis.py "$@"
