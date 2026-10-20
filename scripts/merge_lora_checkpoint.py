"""Merge a LoRA fine-tuning checkpoint of the voxel denoiser into plain weights for the inference pipeline.

Rebuilds the denoiser, injects LoRA with the training rank/alpha, loads the checkpoint's model.safetensors
(base + adapters; or the EMA adapters from ema.safetensors with --ema), merges, and saves a state dict with the
original parameter names: --checkpoints.voxel-denoiser in scripts/run_inference.py loads it unchanged.

Example:
    python scripts/merge_lora_checkpoint.py --checkpoint-dir <run>/checkpoints/checkpoint_6000 --rank 32 \
        --out outputs/runs_full29k/inference_weights/voxel_lora_6000.safetensors [--ema]
"""
from dataclasses import dataclass

import torch
import tyro
from safetensors.torch import load_file, save_file

from models import dit_voxel
from utils.lora import inject_lora, merge_lora


@dataclass
class Args:
    checkpoint_dir: str
    rank: int
    out: str
    alpha: float | None = None
    patch_size: int = 1
    """1 = XL voxel denoiser (as trained), 2 = S"""
    ema: bool = False
    """merge the EMA adapters (ema.safetensors) instead of the raw ones"""


def main(args: Args):
    cfg = dit_voxel.VoxelDitDenoiserArgs()
    cfg.patch_size = args.patch_size
    model = getattr(dit_voxel, cfg.name)(**vars(cfg))
    inject_lora(model, args.rank, args.alpha)
    state = {k.removeprefix("_orig_mod."): v for k, v in load_file(f"{args.checkpoint_dir}/model.safetensors").items()}
    model.load_state_dict(state, strict=True)
    if args.ema:
        ema = {k.removeprefix("denoiser."): v for k, v in load_file(f"{args.checkpoint_dir}/ema.safetensors").items()}
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        assert set(ema) == trainable, f"EMA keys != LoRA parameters ({len(ema)} vs {len(trainable)})"
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in ema:
                    p.copy_(ema[n])
    n = merge_lora(model)
    out = {k: v.contiguous() for k, v in model.state_dict().items()}
    reference = getattr(dit_voxel, cfg.name)(**vars(cfg)).state_dict()
    assert out.keys() == reference.keys(), "merged keys differ from a plain denoiser"
    save_file(out, args.out)
    print(f"merged {n} LoRA layers ({'EMA' if args.ema else 'raw'} adapters) -> {args.out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
