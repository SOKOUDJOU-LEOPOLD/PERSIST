#!/bin/bash
# Is batch 4 faster than batch 1 for FVD rollouts, with the same quality? Same 8 hudnav-eval episodes, w_v0, 150
# frames, 2 A100s: batch 1 (4 pipe calls per GPU) vs batch 4 (1 call per GPU). Logs wall time and peak GPU memory
# per setting, then mean LPIPS / PSNR against ground truth per setting (samples differ only in their random noise).
# Usage: sbatch batch_test_grace.slurm
#SBATCH --job-name=fvd-batch-test
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/user/lsokoudj/PERSIST/outputs/eval_fvd/logs/%x_%j.log

set -uo pipefail
cd /scratch/user/lsokoudj/PERSIST
module load GCC/13.2.0 CUDA/12.6.0
source .venv/bin/activate
export LD_PRELOAD=$EBROOTGCCCORE/lib64/libstdc++.so.6
export PYTHONPATH=.

DATASET=datasets/hudnav_eval
P=checkpoints/pretrained
W=outputs/runs_full29k/inference_weights
T=outputs/eval_fvd/batch_test
FRAMES=150
EPS=$(python -c "
import pandas as pd
m = pd.read_csv('$DATASET/metadata.csv')
print(','.join(sorted(m[m.validation_set].sha256)[:8]))")

for B in ${BATCHES:-1 4}; do  # BATCHES="4" reruns only batch 4 (batch 1 videos kept)
    mkdir -p $T/b$B
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -l 5 > $T/b$B/mem.csv &
    SMI=$!
    START=$(date +%s)
    accelerate launch --multi_gpu --num_processes 2 --mixed_precision bf16 --dynamo_backend no -m scripts.run_inference \
        --pipeline-variant XL --dataset-path $DATASET --eval-instances "$EPS" --num-frames $FRAMES \
        --include-initial-voxel-frame --include-initial-pixel-frame --seed 0 --mixed-precision bf16 --batch-size $B \
        --output-dir $T/b$B \
        --checkpoints.voxel-encoder $P/voxel-encoder/model.safetensors --checkpoints.voxel-decoder $P/voxel-decoder/model.safetensors \
        --checkpoints.pixel-vae $P/pixel-vae/model.safetensors --checkpoints.voxel-denoiser $W/voxel_lora-r32_upd10000.safetensors \
        --checkpoints.camera-model $W/camera_lora-r32_step116000.safetensors \
        --checkpoints.pixel-denoiser $W/pixel_lora-r32_step61000.safetensors \
        --checkpoints.voxel-class-embedder $P/voxel-class-embedder/model.safetensors || echo "=== FAILED: batch $B ==="
    kill $SMI
    echo "=== batch $B: $(( $(date +%s) - START )) s for 8 episodes x $FRAMES frames (incl. model load);" \
         "peak GPU memory $(cut -d, -f2 $T/b$B/mem.csv | sort -n | tail -1) MiB, $(ls $T/b$B/*.mp4 | wc -l) videos ==="
done

python - <<EOF
import os, lpips, pandas as pd, torch, torch.nn.functional as F, torchvision.io as io
torch.set_grad_enabled(False)
net = lpips.LPIPS(net="alex", verbose=False).cuda()
m = pd.read_csv("$DATASET/metadata.csv").set_index("sha256")
rows = []
for sha in "$EPS".split(","):
    gt = io.read_video(f"$DATASET/{m.loc[sha, 'local_path']}/rgb.mp4", pts_unit="sec")[0][:$FRAMES]
    for b in (1, 2, 4):  # every batch size tested so far that has this episode's video
        if not os.path.exists(f"$T/b{b}/XL_{sha}.mp4"):
            continue
        v = io.read_video(f"$T/b{b}/XL_{sha}.mp4", pts_unit="sec")[0][:$FRAMES]
        t = min(len(v), len(gt))
        f = lambda x: F.interpolate(x[:t].permute(0, 3, 1, 2).float().cuda() / 127.5 - 1, size=(180, 320), mode="bilinear", antialias=True)
        lp = torch.cat([net(f(v)[i:i + 32], f(gt)[i:i + 32]).flatten() for i in range(0, t, 32)]).mean().item()
        mse = ((v[:t].float() - gt[:t].float()) ** 2).mean().item()
        rows.append(dict(sha=sha[:10], batch=b, frames=t, lpips=lp, psnr=10 * torch.log10(torch.tensor(255 ** 2 / mse)).item()))
d = pd.DataFrame(rows)
print(d.pivot(index="sha", columns="batch", values=["lpips", "psnr"]).round(3).to_string())
print(d.groupby("batch")[["lpips", "psnr"]].mean().round(3).to_string())
EOF
echo "=== done, $(date) ==="
