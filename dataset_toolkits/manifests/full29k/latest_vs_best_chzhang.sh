#!/bin/bash
# Latest vs best LoRA systems on csce-chzhang (2 GPUs): 600-frame rollouts of 6 hudnav episodes
# (3 seen in training, 3 held out), for side-by-side grids GT | latest | best.
#   latest = voxel step 70k, camera step 140k, pixel step 70k
#   best   = lowest val loss: voxel step 20k (first run, = voxel_lora-r32_upd10000), camera step 116k, pixel step 61k
# The pixel merge needs a GPU (its rasterizer builds a CUDA context), so it runs here; the voxel/camera merged
# weights are copied from Grace. Class embedder = pretrained (LoRA leaves it frozen).
# Usage: nohup bash latest_vs_best_chzhang.sh <gpu for latest> <gpu for best> > <log> 2>&1 &
cd ~/PERSIST; source .venv/bin/activate; export PYTHONPATH=.
P=checkpoints/pretrained; R=outputs/runs_full29k; W=$R/inference_weights
X=$R/pixel_diffuser/full29k-pixel-denoiser-lora-r32-uid-cbd05ad7/checkpoints
OUT=$R/rollouts_latest_vs_best/600f
SEEN=a7da4e62e503ecaf1035e5f3a7532f5d9e450517c57639242f720b76e370a33d,d286a14ed1175edc72e5ae1c7c15079b84aca0cf523b78c082bbf04b43e90dde,9140d7331ea96229a17a601d77ac117309bd6a5dbb7bac5877378516bed90670
HELDOUT=4cf288ffb5292d74e281866abc77dd8355441e1e8a4007dcd1bd810b6803fd0d,be95c1788a501cfe400c591a6959caffbe0983f97f3b28dcbaa51c5a3787bb23,b396722d695478e5d61c213f322070f5489d4f16f6107d741d13688679be8873

for s in 70000:checkpoint_70000 61000:best; do
    [ -f $W/pixel_lora-r32_step${s%%:*}.safetensors ] || CUDA_VISIBLE_DEVICES=$1 python scripts/merge_lora_checkpoint.py \
        --component pixel --checkpoint-dir $X/${s#*:} --rank 32 --out $W/pixel_lora-r32_step${s%%:*}.safetensors || exit 1
done

run() {  # <gpu> <config> <voxel> <camera> <pixel> <episodes>
    echo "=== $2 on GPU $1: ${6:0:12}..., $(date) ==="
    CUDA_VISIBLE_DEVICES=$1 python -m scripts.run_inference \
        --pipeline-variant XL --dataset-path datasets/luanti_full29k --eval-instances "$6" --num-frames 600 \
        --include-initial-pixel-frame --include-initial-voxel-frame --seed 0 --mixed-precision bf16 --output-dir $OUT/$2 \
        --checkpoints.voxel-encoder $P/voxel-encoder/model.safetensors --checkpoints.voxel-decoder $P/voxel-decoder/model.safetensors \
        --checkpoints.pixel-vae $P/pixel-vae/model.safetensors --checkpoints.voxel-denoiser $W/$3.safetensors \
        --checkpoints.camera-model $W/$4.safetensors --checkpoints.pixel-denoiser $W/$5.safetensors \
        --checkpoints.voxel-class-embedder $P/voxel-class-embedder/model.safetensors \
        || echo "=== FAILED: $2 ${6:0:12} ==="
    echo "=== $2 ${6:0:12} done, $(date) ==="
}
LATEST="voxel_lora-r32_step70000 camera_lora-r32_step140000 pixel_lora-r32_step70000"
BEST="voxel_lora-r32_upd10000 camera_lora-r32_step116000 pixel_lora-r32_step61000"
run $1 latest $LATEST $SEEN &
run $1 latest $LATEST $HELDOUT &
run $2 best $BEST $SEEN &
run $2 best $BEST $HELDOUT &
wait
echo "=== all done, $(date) ==="
