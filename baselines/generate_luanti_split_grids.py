"""Generates one comparison-grid video PER SPLIT (train_eval / val / test) for the Luanti
motworld-s1nav-10k fine-tune: each grid has one row per episode, 3 columns --
Input Frame (frozen seed) | GT video | Oasis video -- with live per-frame PSNR burned into the
Oasis panel, mirroring generate_oasis_native_grid.py's single-episode version but batched across
several episodes of a given split and stitched into one video per split.

The checkpoint is loaded ONCE and reused across every episode of every split (loading dominates
per-call overhead relative to a single rollout). Uses augment_actions_to_oasis_native_layout (NOT
the raw 23-dim path) since this checkpoint was fine-tuned with --use-oasis-native-action-layout --
same mapping generate_oasis_native_grid.py uses, so results are directly comparable.

"train_eval" (not the full 1000-episode "train" list) is used for the training-split row by
default -- it's this split file's own small held-out-style sample of training levels, the same
subset eval_oasis_finetuned_fvd.py uses for a "train" qualitative/quantitative readout.

PSNR (per-frame, plus mean/min/max per episode and per split) is the codebase's standard
qualitative-fidelity metric (see eval_oasis_finetuned_fvd.py, generate_oasis_native_grid.py) --
same formula, reused here rather than reinvented, and written out to metrics.json alongside the
videos so the numbers aren't locked inside the burned-in overlay.

Example:
    cd baselines
    .venv/bin/python generate_luanti_split_grids.py \
        --ckpt-path ../outputs/finetune_oasis_luanti_1000ep/step_029500.pt \
        --dataset-root /scratch/user/lsokoudj/PERSIST/data/luanti/extracted/OpenWorldCreative-v0 \
        --split-path ../outputs/oasis_luanti_1000ep.json \
        --output-dir ../outputs/finetune_oasis_luanti_1000ep/qualitative_grids
"""
import json
import os
import sys
from dataclasses import dataclass, field
from typing import List

import av
import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "open-oasis"))

from craftium_action_features import augment_actions_to_oasis_native_layout  # noqa: E402
from finetune_oasis import load_finetuned_dit  # noqa: E402
from finetune_oasis_data import DATASET_ROOT, load_split  # noqa: E402
from finetune_oasis_rollout import generate_rollout, shift_actions  # noqa: E402


@dataclass
class Args:
    ckpt_path: str = "../outputs/finetune_oasis_luanti_1000ep/step_029500.pt"
    split_path: str = "../outputs/oasis_luanti_1000ep.json"
    dataset_root: str = DATASET_ROOT
    vae_ckpt: str = "open-oasis/vit-l-20.safetensors"
    splits: List[str] = field(default_factory=lambda: ["train_eval", "val", "test"])
    split_labels: List[str] = field(default_factory=lambda: ["Train", "Val", "Test"])
    episodes_per_split: int = 3
    episode_offset: int = 0
    num_frames: int = 150
    ddim_steps: int = 10
    fps: int = 20
    output_dir: str = "../outputs/finetune_oasis_luanti_1000ep/qualitative_grids"
    device: str = "cuda:0"
    seed: int = 0


def psnr(gen: np.ndarray, real: np.ndarray) -> float:
    """Verbatim of the rest of the codebase's psnr() (eval_oasis_finetuned_fvd.py,
    generate_oasis_native_grid.py) -- same formula, same 99.0 cap for identical frames."""
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


def load_real_and_native_actions(dataset_root: str, level_id: str, num_frames: int):
    level_dir = os.path.join(dataset_root, level_id)
    container = av.open(os.path.join(level_dir, "rgb.mp4"))
    real_frames = []
    for i, frame in enumerate(container.decode(container.streams.video[0])):
        if i >= num_frames:
            break
        real_frames.append(frame.to_ndarray(format="rgb24"))
    container.close()
    real = np.stack(real_frames)  # (T, H, W, 3) uint8

    npz = np.load(os.path.join(level_dir, "data.npz"))
    action_pool = augment_actions_to_oasis_native_layout(
        npz["action"], npz["player_yaw"], npz["player_pitch"]
    )  # (600, 25)
    raw_action = torch.from_numpy(action_pool[:num_frames].astype(np.float32))
    actions = shift_actions(raw_action)
    seed_frame = torch.from_numpy(real[0]).float().permute(2, 0, 1) / 255.0
    return real, actions, seed_frame


@torch.no_grad()
def main(args: Args):
    device = args.device
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if len(args.split_labels) != len(args.splits):
        raise ValueError("--split-labels must have the same length as --splits.")

    from vae import VAE_models
    from safetensors.torch import load_file

    vae = VAE_models["vit-l-20-shallow-encoder"]()
    vae.load_state_dict(load_file(args.vae_ckpt))
    vae = vae.to(device).eval()

    model, step = load_finetuned_dit(args.ckpt_path, device)
    print(f"Loaded checkpoint at step={step} from {args.ckpt_path}")

    all_metrics = {"checkpoint": args.ckpt_path, "step": step, "splits": {}}

    for split, split_label in zip(args.splits, args.split_labels):
        level_ids = load_split(args.split_path, split)[
            args.episode_offset : args.episode_offset + args.episodes_per_split
        ]
        print(f"\n=== split='{split}' ({split_label}): {len(level_ids)} episode(s) ===")

        rows = []  # list of (level_id, real, gen, per_frame_psnr)
        for level_id in level_ids:
            real, actions, seed_frame = load_real_and_native_actions(args.dataset_root, level_id, args.num_frames)
            torch.manual_seed(args.seed)
            gen = generate_rollout(
                model, vae, seed_frame, actions, args.num_frames, device,
                ddim_steps=args.ddim_steps, desc=f"{split}/{level_id}",
            ).numpy()  # (T, H, W, 3) uint8

            n = min(real.shape[0], gen.shape[0])
            real_n, gen_n = real[:n], gen[:n]
            per_frame_psnr = np.array([psnr(gen_n[t], real_n[t]) for t in range(n)], dtype=np.float32)
            print(f"  {level_id}: {n} frames, PSNR mean={per_frame_psnr.mean():.2f}dB "
                  f"min={per_frame_psnr.min():.2f}dB max={per_frame_psnr.max():.2f}dB")
            rows.append((level_id, real_n, gen_n, per_frame_psnr))

        n_common = min(r[1].shape[0] for r in rows)
        frames_out = []
        for t in range(n_common):
            row_panels = []
            for level_id, real_n, gen_n, per_frame_psnr in rows:
                p_input = label_panel(real_n[0], f"Input Frame ({split_label})", "")
                p_gt = label_panel(real_n[t], "GT video", "")
                p_oasis = label_panel(gen_n[t], "Oasis video", f"PSNR: {per_frame_psnr[t]:.1f}dB")
                row_panels.append(np.concatenate([p_input, p_gt, p_oasis], axis=1))
            frames_out.append(np.concatenate(row_panels, axis=0))

        video_path = os.path.join(args.output_dir, f"grid_{split}.mp4")
        imageio.mimsave(video_path, frames_out, fps=args.fps)
        print(f"Saved {split} grid ({len(rows)} episode(s) x {n_common} frames) to {video_path}")

        npz_path = os.path.join(args.output_dir, f"grid_{split}.npz")
        np.savez(
            npz_path,
            level_ids=np.array([r[0] for r in rows]),
            real=np.stack([r[1][:n_common] for r in rows]),
            generated=np.stack([r[2][:n_common] for r in rows]),
            psnr=np.stack([r[3][:n_common] for r in rows]),
            fps=args.fps,
        )

        split_metrics = {
            "episodes": {
                level_id: {
                    "num_frames": int(per_frame_psnr.shape[0]),
                    "psnr_mean_db": float(per_frame_psnr.mean()),
                    "psnr_min_db": float(per_frame_psnr.min()),
                    "psnr_max_db": float(per_frame_psnr.max()),
                }
                for level_id, _, _, per_frame_psnr in rows
            },
            "psnr_mean_db": float(np.mean([r[3].mean() for r in rows])),
        }
        all_metrics["splits"][split] = split_metrics
        print(f"  {split} split mean PSNR: {split_metrics['psnr_mean_db']:.2f}dB")

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
