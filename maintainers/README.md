### Publishing to HuggingFace (maintainers)

`dataset_toolkits/hf_upload.py` packs and pushes the artifacts. It needs a write token
(`huggingface-cli login`) for a user with write access to the target org:

```bash
# Sample eval dataset: copies the minimal files for N validation episodes and uploads a dataset repo
python -m dataset_toolkits.hf_upload upload-eval-dataset \
  --src datasets/dynamic48_L8 --repo-id PERSIST-team/persist-eval-sample

# Weights: one model repo per checkpoint, each as model.safetensors
# (voxel denoiser is variant-specific: -s and -xl repos; all others shared)
python -m dataset_toolkits.hf_upload upload-weights \
  --namespace PERSIST-team \
  --voxel-encoder        ckpts/voxel_vae/model.safetensors \
  --voxel-decoder        ckpts/voxel_vae/model_1.safetensors \
  --pixel-vae            ckpts/pixel_vae/model.safetensors \
  --camera-model         ckpts/camera_model/model.safetensors \
  --voxel-denoiser-s     ckpts/voxel_dit_s/model.safetensors \
  --voxel-denoiser-xl    ckpts/voxel_dit_xl/model.safetensors \
  --pixel-denoiser       ckpts/pixel_dit/model.safetensors \
  --voxel-class-embedder ckpts/pixel_dit/model_1.safetensors
```