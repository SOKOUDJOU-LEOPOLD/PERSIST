"""Qualitative sanity check for the WorldMem-native-action-layout fine-tune: generates one
comparison video -- Input Frame (frozen seed) | GT video | WorldMem rollout -- with live per-frame
PSNR burned into the WorldMem panel, for a single episode.

Reuses build_worldmem_shell_enriched / load_episode_batch_enriched from
eval_worldmem_pose_causality_cleancam.py (action_layout="native") so this uses the SAME
augment_actions_to_worldmem_native_layout mapping the checkpoint was actually fine-tuned with --
NOT the old 23-dim finetune_worldmem_generate.py path, which would silently feed the wrong action
layout to a checkpoint trained on the native 25-slot one.

Example:
    cd baselines/WorldMem
    .venv/bin/python ../generate_native_grid.py \
        --ckpt-path ../../outputs/overfit_worldmem_level_006_native_v1/final.pt \
        --level-id level_006 --device cuda:1
"""
import os
import sys
from dataclasses import dataclass

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "WorldMem"))

from eval_worldmem_pose_causality_cleancam import (  # noqa: E402
    build_worldmem_shell_enriched,
    load_episode_batch_enriched,
    ACTION_DIM_BY_LAYOUT,
)
from finetune_worldmem_data import DATASET_ROOT  # noqa: E402


def psnr(gen: np.ndarray, real: np.ndarray) -> float:
    mse = np.mean((gen.astype(np.float32) - real.astype(np.float32)) ** 2)
    if mse < 1e-10:
        return 99.0
    return float(10 * np.log10(255.0**2 / mse))


def label_panel(frame: np.ndarray, title: str, psnr_text: str) -> np.ndarray:
    img = Image.fromarray(frame)
    canvas = Image.new("RGB", (img.width, img.height + 36), (20, 20, 20))
    canvas.paste(img, (0, 36))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, 6), title, fill=(255, 255, 255), font=font)
    if psnr_text:
        draw.rectangle([(0, canvas.height - 30), (220, canvas.height)], fill=(0, 0, 0))
        draw.text((8, canvas.height - 26), psnr_text, fill=(120, 255, 120), font=font)
    return np.array(canvas)


@dataclass
class Args:
    ckpt_path: str = "../../outputs/overfit_worldmem_level_006_native_v1/final.pt"
    config_path: str = "configurations/huggingface.yaml"
    level_id: str = "level_006"
    dataset_root: str = DATASET_ROOT
    num_frames: int = 150
    memory_condition_length: int = 8
    fps: int = 20
    output_path: str = "../../outputs/overfit_worldmem_level_006_native_v1/grid_level_006.mp4"
    device: str = "cuda:1"
    seed: int = 0
    start_frame: int = 0
    """Starts the rollout at this episode frame instead of 0 (memory frames also relocate to all
    point here) -- lets a specific episode segment sit only a few autoregressive steps into a
    FRESH rollout, to test whether a stall seen deep in a long rollout is drift-depth-dependent or
    episode-position-dependent. See load_episode_batch_enriched's docstring."""


def main(args: Args):
    device = args.device
    torch.manual_seed(args.seed)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    action_dim = ACTION_DIM_BY_LAYOUT["native"]
    worldmem = build_worldmem_shell_enriched(args.config_path, device, action_dim)
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    worldmem.diffusion_model.load_state_dict(ckpt["model"])
    worldmem.diffusion_model.eval()
    print(f"Loaded checkpoint at step={ckpt.get('step')} from {args.ckpt_path}")

    from omegaconf import OmegaConf
    from huggingface_hub import hf_hub_download

    cfg = OmegaConf.load(args.config_path)
    vae_ckpt_path = hf_hub_download(
        repo_id="/".join(cfg.vae_path.split("/")[:2]), filename="/".join(cfg.vae_path.split("/")[2:])
    )
    vae_ckpt = torch.load(vae_ckpt_path, map_location="cpu")
    worldmem.vae.load_state_dict(vae_ckpt["state_dict"] if "state_dict" in vae_ckpt else vae_ckpt, strict=True)

    video, actions, poses, timestamp = load_episode_batch_enriched(
        args.dataset_root, args.level_id, args.num_frames, args.memory_condition_length, args.seed, "native",
        start_frame=args.start_frame,
    )
    video, actions, poses, timestamp = (
        video.to(device), actions.to(device), poses.to(device), timestamp.to(device)
    )

    worldmem.validation_step_outputs = []
    with torch.no_grad():
        worldmem.validation_step((video, actions, poses, timestamp), 0)
    xs_pred, xs_decode = worldmem.validation_step_outputs[-1]  # (T, B, C, H, W), [0,1] float

    def to_np(x):
        x = torch.clamp(x[:, 0], 0, 1) * 255.0
        return x.permute(0, 2, 3, 1).byte().cpu().numpy()

    gen_np, real_np = to_np(xs_pred), to_np(xs_decode)
    n = min(gen_np.shape[0], real_np.shape[0])
    gen_np, real_np = gen_np[:n], real_np[:n]
    psnr_per_frame = np.array([psnr(gen_np[t], real_np[t]) for t in range(n)], dtype=np.float32)
    print(f"Rollout: {n} frames, PSNR mean={psnr_per_frame.mean():.2f}dB "
          f"min={psnr_per_frame.min():.2f}dB max={psnr_per_frame.max():.2f}dB")

    seed_frame = real_np[0]
    frames_out = []
    for t in range(n):
        p_input = label_panel(seed_frame, "Input Frame (seed)", "")
        p_gt = label_panel(real_np[t], "GT video", "")
        p_wm = label_panel(gen_np[t], "WorldMem (native layout)", f"PSNR: {psnr_per_frame[t]:.1f}dB")
        frames_out.append(np.concatenate([p_input, p_gt, p_wm], axis=1))

    imageio.mimsave(args.output_path, frames_out, fps=args.fps)
    print(f"Saved grid video to {args.output_path} ({n} frames)")

    npz_path = args.output_path.replace(".mp4", ".npz")
    np.savez(npz_path, real=real_np, generated=gen_np, psnr=psnr_per_frame, level_id=args.level_id, fps=args.fps)
    print(f"Saved raw arrays to {npz_path}")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
