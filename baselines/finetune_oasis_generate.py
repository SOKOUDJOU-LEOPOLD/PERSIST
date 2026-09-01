"""Qualitative rollout for a fine-tuned (23-dim action) Oasis checkpoint, driven by a real
Craftium test episode's actual seed frame + real action sequence -- unlike the original
zero-shot eval (own action sampling, no relation to ground truth), this uses the SAME actions
that produced the real episode, so the generated video is directly comparable to the real one
frame-by-frame, and "before" vs "after" checkpoints can be compared on identical conditioning.

DDIM sampling loop copied from baselines/open-oasis/generate.py (only the checkpoint-loading and
model construction differ -- action_cond_dim=23 instead of the default 25, loaded with the
matching load_oasis_for_finetuning-style key handling since our fine-tuned checkpoints are
already correctly shaped).

Example:
    cd baselines
    python finetune_oasis_generate.py \
        --ckpt-path ../outputs/finetune_oasis/final.pt \
        --level-id level_002 \
        --output-path ../outputs/finetune_oasis/rollout_level_002_after.mp4
"""
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
from einops import rearrange
from torch import autocast
from torchvision.io import write_video
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "open-oasis"))

from finetune_oasis import CRAFTIUM_ACTION_DIM, load_finetuned_dit, sigmoid_beta_schedule  # noqa: E402
from finetune_oasis_data import DATASET_ROOT  # noqa: E402


@dataclass
class Args:
    ckpt_path: str = "../outputs/finetune_oasis/final.pt"
    vae_ckpt: str = "open-oasis/vit-l-20.safetensors"
    level_id: str = "level_002"
    dataset_root: str = DATASET_ROOT
    num_frames: int = 150
    n_prompt_frames: int = 1
    ddim_steps: int = 10
    fps: int = 20
    output_path: str = "rollout.mp4"
    device: str = "cuda:0"


load_finetuned_model = load_finetuned_dit  # auto-detects gated vs. plain external_cond


@torch.no_grad()
def main(args: Args):
    device = args.device
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    model, step = load_finetuned_model(args.ckpt_path, device)
    print(f"Loaded checkpoint at step={step} from {args.ckpt_path}")

    from vae import VAE_models
    from safetensors.torch import load_file

    vae = VAE_models["vit-l-20-shallow-encoder"]()
    vae.load_state_dict(load_file(args.vae_ckpt))
    vae = vae.to(device).eval()

    # Real Craftium seed frame + real action sequence for this test episode (see module docstring
    # -- unlike zero-shot eval, driven by ground-truth actions for direct comparability).
    import av

    level_dir = os.path.join(args.dataset_root, args.level_id)
    container = av.open(os.path.join(level_dir, "rgb.mp4"))
    seed_frame = None
    for frame in container.decode(container.streams.video[0]):
        seed_frame = frame.to_ndarray(format="rgb24")
        break
    container.close()
    x = torch.from_numpy(seed_frame).float().permute(2, 0, 1)[None, None] / 255.0  # (1,1,3,H,W)

    npz = np.load(os.path.join(level_dir, "data.npz"))
    raw_action = npz["action"][: args.num_frames].astype(np.float32)  # (T, 23)
    actions = torch.from_numpy(raw_action)[None].to(device)  # (1, T, 23), no shift here --
    # the DDIM loop below indexes actions[:, start_frame : i + 1] per-frame directly, matching
    # generate.py's own indexing convention (it slices its own pre-shifted `actions` the same way).
    actions = torch.cat([torch.zeros_like(actions[:, :1]), actions[:, :-1]], dim=1)  # shift by one

    x = x.to(device)
    n_prompt_frames = args.n_prompt_frames
    total_frames = args.num_frames
    max_noise_level = 1000
    ddim_noise_steps = args.ddim_steps
    noise_range = torch.linspace(-1, max_noise_level - 1, ddim_noise_steps + 1)
    noise_abs_max = 20
    stabilization_level = 15

    B = x.shape[0]
    H, W = x.shape[-2:]
    scaling_factor = 0.07843137255
    x = rearrange(x, "b t c h w -> (b t) c h w")
    with autocast("cuda", dtype=torch.bfloat16):
        x = vae.encode(x * 2 - 1).mean * scaling_factor
    x = rearrange(x, "(b t) (h w) c -> b t c h w", t=n_prompt_frames, h=H // vae.patch_size, w=W // vae.patch_size)
    x = x[:, :n_prompt_frames]

    betas = sigmoid_beta_schedule(max_noise_level).float().to(device)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    alphas_cumprod = rearrange(alphas_cumprod, "T -> T 1 1 1")

    for i in tqdm(range(n_prompt_frames, total_frames), desc=f"Generating ({os.path.basename(args.output_path)})"):
        chunk = torch.randn((B, 1, *x.shape[-3:]), device=device)
        chunk = torch.clamp(chunk, -noise_abs_max, noise_abs_max)
        x = torch.cat([x, chunk], dim=1)
        start_frame = max(0, i + 1 - model.max_frames)

        for noise_idx in reversed(range(1, ddim_noise_steps + 1)):
            t_ctx = torch.full((B, i), stabilization_level - 1, dtype=torch.long, device=device)
            t = torch.full((B, 1), noise_range[noise_idx], dtype=torch.long, device=device)
            t_next = torch.full((B, 1), noise_range[noise_idx - 1], dtype=torch.long, device=device)
            t_next = torch.where(t_next < 0, t, t_next)
            t = torch.cat([t_ctx, t], dim=1)
            t_next = torch.cat([t_ctx, t_next], dim=1)

            x_curr = x.clone()[:, start_frame:]
            t = t[:, start_frame:]
            t_next = t_next[:, start_frame:]

            with autocast("cuda", dtype=torch.bfloat16):
                v = model(x_curr, t, actions[:, start_frame : i + 1])

            x_start = alphas_cumprod[t].sqrt() * x_curr - (1 - alphas_cumprod[t]).sqrt() * v
            x_noise = ((1 / alphas_cumprod[t]).sqrt() * x_curr - x_start) / (1 / alphas_cumprod[t] - 1).sqrt()

            alpha_next = alphas_cumprod[t_next]
            alpha_next[:, :-1] = torch.ones_like(alpha_next[:, :-1])
            if noise_idx == 1:
                alpha_next[:, -1:] = torch.ones_like(alpha_next[:, -1:])
            x_pred = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()
            x[:, -1:] = x_pred[:, -1:]

    x = rearrange(x, "b t c h w -> (b t) (h w) c")
    x = (vae.decode(x / scaling_factor) + 1) / 2
    x = rearrange(x, "(b t) c h w -> b t h w c", t=total_frames)
    x = torch.clamp(x, 0, 1)
    x = (x * 255).byte()
    write_video(args.output_path, x[0].cpu(), fps=args.fps)
    print(f"Saved rollout to {args.output_path}")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
