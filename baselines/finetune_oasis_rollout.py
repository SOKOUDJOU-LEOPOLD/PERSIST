"""Shared DDIM rollout function for a fine-tuned (23-dim action) Oasis checkpoint, driven by a
real Craftium seed frame + real action sequence. Factored out of finetune_oasis_generate.py so
both the standalone qualitative-video script and the FVD evaluation script
(eval_oasis_finetuned_fvd.py) use the exact same generation code -- no risk of the two diverging.

DDIM loop copied from baselines/open-oasis/generate.py (see that file / finetune_oasis_generate.py
for the original, single-episode, video-writing version this was factored out of).
"""
import torch
from einops import rearrange
from torch import autocast
from tqdm import tqdm

from finetune_oasis import sigmoid_beta_schedule


@torch.no_grad()
def generate_rollout(
    model,
    vae,
    seed_frame: torch.Tensor,
    actions: torch.Tensor,
    num_frames: int,
    device: str,
    n_prompt_frames: int = 1,
    ddim_steps: int = 10,
    desc: str = "Generating",
    show_progress: bool = True,
) -> torch.Tensor:
    """
    Args:
        seed_frame: (3, H, W) float tensor in [0, 1].
        actions: (T, 23) float tensor, Craftium's native action vector, ALREADY shifted by one
            frame (actions[0] = zeros, actions[t] = the raw action that produced frame t) --
            matching finetune_oasis_data.py's convention. Caller shifts, not this function, so
            both training-data loading and generation share one shift implementation.
        num_frames: total frames to generate (including the n_prompt_frames seed frame(s)).

    Returns:
        (num_frames, H, W, 3) uint8 tensor, CPU.
    """
    x = seed_frame[None, None].to(device)  # (1, 1, 3, H, W)
    actions = actions[None].to(device)  # (1, T, 23)

    max_noise_level = 1000
    noise_range = torch.linspace(-1, max_noise_level - 1, ddim_steps + 1)
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

    iterator = range(n_prompt_frames, num_frames)
    if show_progress:
        iterator = tqdm(iterator, desc=desc)
    for i in iterator:
        chunk = torch.randn((B, 1, *x.shape[-3:]), device=device)
        chunk = torch.clamp(chunk, -noise_abs_max, noise_abs_max)
        x = torch.cat([x, chunk], dim=1)
        start_frame = max(0, i + 1 - model.max_frames)

        for noise_idx in reversed(range(1, ddim_steps + 1)):
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
    x = rearrange(x, "(b t) c h w -> b t h w c", t=num_frames)
    x = torch.clamp(x, 0, 1)
    x = (x * 255).byte()
    return x[0].cpu()


def shift_actions(raw_action: torch.Tensor) -> torch.Tensor:
    """raw_action: (T, 23) -> (T, 23) shifted by one frame (load_actions convention: actions[0]
    is unavailable/zero, actions[t] = raw_action[t-1] for t >= 1)."""
    return torch.cat([torch.zeros_like(raw_action[:1]), raw_action[:-1]], dim=0)
