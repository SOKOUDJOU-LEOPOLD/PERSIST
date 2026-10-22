"""Merge a LoRA fine-tuning checkpoint (voxel denoiser or camera model) into plain weights for the inference pipeline.

Rebuilds the model, injects LoRA with the training rank/alpha, loads the checkpoint's model.safetensors
(base + adapters; or the EMA adapters from ema.safetensors with --ema), merges, and saves a state dict with the
original parameter names: --checkpoints.voxel-denoiser / --checkpoints.camera-model in scripts/run_inference.py
load it unchanged.

Example:
    python scripts/merge_lora_checkpoint.py --checkpoint-dir <run>/checkpoints/checkpoint_6000 --rank 32 \
        --out outputs/runs_full29k/inference_weights/voxel_lora_6000.safetensors [--ema] [--component camera]
"""
from dataclasses import dataclass
from typing import Literal

import torch
import tyro
from safetensors.torch import load_file, save_file

from models import camera_transformer, dit_voxel
from utils.lora import inject_lora, merge_lora


@dataclass
class Args:
    checkpoint_dir: str
    rank: int
    out: str
    alpha: float | None = None
    component: Literal["voxel", "camera"] = "voxel"
    patch_size: int = 1
    """voxel only: 1 = XL voxel denoiser (as trained), 2 = S"""
    ema: bool = False
    """merge the EMA adapters (ema.safetensors) instead of the raw ones"""


def build_model(args: Args):
    if args.component == "voxel":
        cfg = dit_voxel.VoxelDitDenoiserArgs()
        cfg.patch_size = args.patch_size
        return getattr(dit_voxel, cfg.name)(**vars(cfg)), "denoiser."
    # same output representations as the camera fine-tune in train_full29k.slurm
    cfg = camera_transformer.CameraTransformerArgs(cam_rotation_output_rep="delta_angle", cam_fov_output_rep="delta")
    return getattr(camera_transformer, cfg.name)(**vars(cfg)), "camera."


def main(args: Args):
    model, ema_prefix = build_model(args)
    inject_lora(model, args.rank, args.alpha)
    state = {k.removeprefix("_orig_mod."): v for k, v in load_file(f"{args.checkpoint_dir}/model.safetensors").items()}
    model.load_state_dict(state, strict=True)
    if args.ema:
        ema = {k.removeprefix(ema_prefix): v for k, v in load_file(f"{args.checkpoint_dir}/ema.safetensors").items()}
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        assert set(ema) == trainable, f"EMA keys != LoRA parameters ({len(ema)} vs {len(trainable)})"
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in ema:
                    p.copy_(ema[n])
    n = merge_lora(model)
    out = {k: v.contiguous() for k, v in model.state_dict().items()}
    reference = build_model(args)[0].state_dict()
    assert out.keys() == reference.keys(), f"merged keys differ from a plain {args.component} model"
    save_file(out, args.out)
    print(f"merged {n} LoRA layers ({'EMA' if args.ema else 'raw'} adapters) -> {args.out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
