"""Zero-shot Oasis baseline: score against persist-eval-sample ground truth.

Per the agreed approach ("same seed frame + own action sampling"): each held-out
persist-eval-sample episode's first frame is used as Oasis's image prompt (its own
`generate.py` already supports arbitrary image prompts via `--prompt-path`), Oasis
then generates its own rollout using its bundled native action sequence
(sample_data/sample_actions_0.one_hot_actions.pt -- a real recorded Minecraft
action sequence, not derived from Craftium in any way), and the result is scored
via FVD against the SAME real ground truth used for the PERSIST row -- giving a
shared real-distribution basis across methods, unlike each-baseline's-own-domain
evaluation (which would need a separate Minecraft/MineDojo real-clip source).

This intentionally does NOT try to replicate Craftium's actual action trajectory
or action space (Oasis's 25-dim VPT-style keyboard/mouse encoding has no direct
mapping to Craftium's 23-dim one) -- that would be the "full data adapter" version
of zero-shot evaluation, which is materially the same work as the later toy
finetuning phase.

Reuses:
  - open-oasis's own model/VAE loading and DDIM sampling loop (generate.py),
    factored into a function here rather than shelling out to the script once per
    episode (would reload the 2.4GB checkpoint 256 times).
  - PERSIST's utils/fvd_metric.py for I3D features + Frechet distance, so the
    metric implementation is identical to the PERSIST row's -- not a second,
    possibly-inconsistent FVD implementation.

Runs in PERSIST's existing venv (open-oasis's only extra dependency,
rotary_embedding_torch, was added there directly -- no separate environment
needed, unlike WorldMem).

Example:
    cd baselines/open-oasis
    python ../eval_oasis_zeroshot.py \
        --persist-eval-root <persist-eval-sample snapshot dir> \
        --frame-lengths 200 400 600 \
        --output-json ../../outputs/eval_fvd/oasis_zeroshot.json
"""
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import av
import torch
import tyro
from einops import rearrange
from loguru import logger
from safetensors.torch import load_model
from torch import autocast
from torchvision.io import read_video
from tqdm import tqdm

# open-oasis's own modules (this script is meant to be run with CWD=baselines/open-oasis,
# or with that directory on sys.path).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "open-oasis"))
from dit import DiT_models  # noqa: E402
from utils import load_actions, load_prompt, sigmoid_beta_schedule  # noqa: E402
from vae import VAE_models  # noqa: E402

# PERSIST's utils/fvd_metric.py, for the shared FVD implementation. Loaded by file path
# rather than `sys.path.insert + from utils.fvd_metric import ...`: open-oasis's own flat
# utils.py (imported just above) already occupies the `utils` name in sys.modules, so a
# package-style import of PERSIST's utils/ package would silently resolve to the wrong
# module (or fail with "'utils' is not a package") instead of raising anything obviously
# wrong at the collision site.
import importlib.util

PERSIST_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_fvd_metric_spec = importlib.util.spec_from_file_location(
    "persist_fvd_metric", os.path.join(PERSIST_ROOT, "utils", "fvd_metric.py")
)
_fvd_metric = importlib.util.module_from_spec(_fvd_metric_spec)
_fvd_metric_spec.loader.exec_module(_fvd_metric)
I3D_FEATURE_DIM = _fvd_metric.I3D_FEATURE_DIM
clips_from_video = _fvd_metric.clips_from_video
compute_frechet_distance = _fvd_metric.compute_frechet_distance
extract_i3d_features = _fvd_metric.extract_i3d_features
load_i3d_model = _fvd_metric.load_i3d_model

DEVICE = "cuda:0"


def extract_frame0_png(video_path: str, out_path: str) -> str:
    """Save a video's first frame as a PNG.

    open-oasis's own `load_prompt` supports reading directly from a video path, but its
    VIDEO_EXTENSIONS branch turns out to be broken (untransposed read_video output fed
    straight into `resize`/the VAE -- see the shape-mismatch traceback this replaced).
    Their IMAGE_EXTENSIONS branch works correctly, so we go through that instead, rather
    than patching their vendored code.
    """
    container = av.open(video_path)
    frame = next(container.decode(container.streams.video[0]))
    frame.to_image().save(out_path)
    container.close()
    return out_path


@dataclass
class Args:
    persist_eval_root: str
    """Path to the persist-eval-sample snapshot directory (each level_XXX/rgb.mp4)."""
    oasis_ckpt: str = "oasis500m.safetensors"
    vae_ckpt: str = "vit-l-20.safetensors"
    actions_path: str = "sample_data/sample_actions_0.one_hot_actions.pt"

    num_instances: Optional[int] = None
    """Cap on how many episodes to evaluate (default: all found under persist_eval_root)."""
    frame_lengths: List[int] = field(default_factory=lambda: [200, 400, 600])
    ddim_steps: int = 10

    i3d_checkpoint: Optional[str] = None
    i3d_batch_size: int = 8

    output_json: str = "outputs/eval_fvd/oasis_zeroshot.json"
    method_name: str = "Oasis"


def load_oasis(oasis_ckpt: str, vae_ckpt: str):
    model = DiT_models["DiT-S/2"]()
    logger.info(f"Loading Oasis-500M from {oasis_ckpt}")
    load_model(model, oasis_ckpt)
    model = model.to(DEVICE).eval()

    vae = VAE_models["vit-l-20-shallow-encoder"]()
    logger.info(f"Loading ViT-VAE-L/20 from {vae_ckpt}")
    load_model(vae, vae_ckpt)
    vae = vae.to(DEVICE).eval()
    return model, vae


@torch.no_grad()
def oasis_rollout(model, vae, prompt_path: str, actions: torch.Tensor, total_frames: int, ddim_steps: int) -> torch.Tensor:
    """Generate `total_frames` frames from a single image prompt. Adapted directly from
    open-oasis/generate.py's main() sampling loop (same math, just factored into a function
    reusable across many episodes without reloading the model each time)."""
    n_prompt_frames = 1
    max_noise_level = 1000
    noise_range = torch.linspace(-1, max_noise_level - 1, ddim_steps + 1)
    noise_abs_max = 20
    stabilization_level = 15

    x = load_prompt(prompt_path, n_prompt_frames=n_prompt_frames).to(DEVICE)
    actions = actions[:, :total_frames].to(DEVICE)

    B = x.shape[0]
    H, W = x.shape[-2:]
    scaling_factor = 0.07843137255
    x = rearrange(x, "b t c h w -> (b t) c h w")
    with autocast("cuda", dtype=torch.half):
        x = vae.encode(x * 2 - 1).mean * scaling_factor
    x = rearrange(x, "(b t) (h w) c -> b t c h w", t=n_prompt_frames, h=H // vae.patch_size, w=W // vae.patch_size)

    betas = sigmoid_beta_schedule(max_noise_level).float().to(DEVICE)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    alphas_cumprod = rearrange(alphas_cumprod, "T -> T 1 1 1")

    for i in range(n_prompt_frames, total_frames):
        chunk = torch.clamp(torch.randn((B, 1, *x.shape[-3:]), device=DEVICE), -noise_abs_max, noise_abs_max)
        x = torch.cat([x, chunk], dim=1)
        start_frame = max(0, i + 1 - model.max_frames)

        for noise_idx in reversed(range(1, ddim_steps + 1)):
            t_ctx = torch.full((B, i), stabilization_level - 1, dtype=torch.long, device=DEVICE)
            t = torch.full((B, 1), noise_range[noise_idx], dtype=torch.long, device=DEVICE)
            t_next = torch.full((B, 1), noise_range[noise_idx - 1], dtype=torch.long, device=DEVICE)
            t_next = torch.where(t_next < 0, t, t_next)
            t = torch.cat([t_ctx, t], dim=1)
            t_next = torch.cat([t_ctx, t_next], dim=1)

            x_curr = x.clone()[:, start_frame:]
            t = t[:, start_frame:]
            t_next = t_next[:, start_frame:]

            with autocast("cuda", dtype=torch.half):
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
    x = rearrange(x, "(b t) c h w -> b t c h w", t=total_frames)[0]
    x = torch.clamp(x, 0, 1) * 255.0
    return x.cpu()  # (T, C, H, W), [0, 255] float


def main(args: Args):
    model, vae = load_oasis(args.oasis_ckpt, args.vae_ckpt)
    actions = load_actions(args.actions_path)
    max_frames = max(args.frame_lengths)
    if actions.shape[1] < max_frames:
        raise ValueError(
            f"Bundled action file only has {actions.shape[1]} steps, need {max_frames}. "
            "Pass a longer --actions-path."
        )

    root = Path(args.persist_eval_root)
    levels = sorted(p.parent.name for p in root.glob("level_*/rgb.mp4"))
    if args.num_instances:
        levels = levels[: args.num_instances]
    if not levels:
        raise ValueError(f"No level_*/rgb.mp4 found under {root}")
    logger.info(f"Found {len(levels)} persist-eval-sample episodes under {root}")

    i3d = load_i3d_model(args.i3d_checkpoint, device=DEVICE)

    local_real_feats = {length: [] for length in args.frame_lengths}
    local_gen_feats = {length: [] for length in args.frame_lengths}

    import tempfile

    for level in tqdm(levels, desc="Oasis zero-shot rollouts"):
        rgb_path = str(root / level / "rgb.mp4")
        with tempfile.TemporaryDirectory() as tmp_dir:
            prompt_png = extract_frame0_png(rgb_path, os.path.join(tmp_dir, "frame0.png"))
            gen = oasis_rollout(model, vae, prompt_png, actions, max_frames, args.ddim_steps)  # (T, C, H, W)

        real, _, _ = read_video(rgb_path, pts_unit="sec")  # (T, H, W, C) uint8
        real = rearrange(real[:max_frames].float(), "t h w c -> t c h w")
        # Oasis fixes its working resolution to 360x640 (see open-oasis/utils.py:load_prompt);
        # resize the real clip to match so both sides feed I3D at the same resolution.
        if real.shape[-2:] != gen.shape[-2:]:
            real = torch.nn.functional.interpolate(real, size=gen.shape[-2:], mode="bilinear", align_corners=False)

        n = min(gen.shape[0], real.shape[0])
        if n < max_frames:
            logger.warning(f"{level}: only {n} frames available (wanted {max_frames}); truncating.")
        gen, real = gen[:n], real[:n]

        for length in args.frame_lengths:
            if gen.shape[0] < length:
                continue
            gen_clips = clips_from_video(gen[:length])
            real_clips = clips_from_video(real[:length])
            local_gen_feats[length].append(
                extract_i3d_features(gen_clips, i3d, device=DEVICE, batch_size=args.i3d_batch_size)
            )
            local_real_feats[length].append(
                extract_i3d_features(real_clips, i3d, device=DEVICE, batch_size=args.i3d_batch_size)
            )

    import numpy as np

    results = {}
    for length in sorted(args.frame_lengths):
        gen_feats = (
            np.concatenate(local_gen_feats[length], axis=0)
            if local_gen_feats[length]
            else np.zeros((0, I3D_FEATURE_DIM), dtype=np.float32)
        )
        real_feats = (
            np.concatenate(local_real_feats[length], axis=0)
            if local_real_feats[length]
            else np.zeros((0, I3D_FEATURE_DIM), dtype=np.float32)
        )
        fvd = compute_frechet_distance(real_feats, gen_feats)
        logger.info(
            f"{args.method_name} FVD @ {length} frames: {fvd:.1f} "
            f"({real_feats.shape[0]} real / {gen_feats.shape[0]} generated clips)"
        )
        results[str(length)] = fvd

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({"method": args.method_name, "fvd": results}, f, indent=2)
    logger.info(f"Saved results to {args.output_json}")


if __name__ == "__main__":
    main(tyro.cli(Args))
