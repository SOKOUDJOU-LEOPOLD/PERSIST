"""Side-by-side comparison of rollouts: ground truth | pretrained | finetuned_raw | finetuned_ema.

For every episode with rollouts under <rollouts>/<N>f/<config>/XL_<sha256>.mp4 writes
  <rollouts>/compare_<N>f/<sha256>.mp4   (labelled columns, one row per frame)
  <rollouts>/compare_<N>f/<sha256>.png   (still grid: rows = selected frames, columns = configs)
Ground truth is frames [0, N) of the episode's rgb.mp4 (the rollouts start from frame 0).

Usage: python compose_rollouts.py <frames, e.g. 16 or 600> [rollouts subdir] [comma-separated configs]
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
import torchvision.io as io
from PIL import Image, ImageDraw, ImageFont

ROOT = "/scratch/user/lsokoudj/PERSIST"
DS = f"{ROOT}/datasets/luanti_full29k"
ROLL = f"{ROOT}/outputs/runs_full29k/rollouts"
CONFIGS = ["pretrained", "finetuned_raw", "finetuned_ema"]


def label(frames, text):
    """Burn a text label into the top-left corner of (T, H, W, 3) uint8 frames."""
    font = ImageFont.load_default(size=22) if hasattr(ImageFont, "load_default") else None
    out = []
    for f in frames:
        img = Image.fromarray(f)
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, 14 * len(text) + 16, 32], fill=(0, 0, 0))
        d.text((8, 4), text, fill=(255, 255, 255), font=font)
        out.append(np.asarray(img))
    return np.stack(out)


def main(n, roll=ROLL, configs=CONFIGS):
    meta = pd.read_csv(f"{DS}/metadata.csv").set_index("sha256")
    out_dir = f"{roll}/compare_{n}f"
    os.makedirs(out_dir, exist_ok=True)
    shas = sorted(f[3:-4] for f in os.listdir(f"{roll}/{n}f/{configs[0]}") if f.endswith(".mp4"))
    for sha in shas:
        paths = {c: f"{roll}/{n}f/{c}/XL_{sha}.mp4" for c in configs}
        if not all(os.path.exists(p) for p in paths.values()):
            print(f"skip {sha[:12]}: not all configs done"); continue
        gt, _, _ = io.read_video(f"{DS}/{meta.loc[sha, 'local_path']}/rgb.mp4", pts_unit="sec")
        cols = [label(gt[:n].numpy(), "ground truth")]
        for c in configs:
            v, _, _ = io.read_video(paths[c], pts_unit="sec")
            cols.append(label(v[:n].numpy(), c.replace("_", " ")))
        t = min(len(c) for c in cols)
        grid = np.concatenate([c[:t] for c in cols], axis=2)  # (T, H, 4W, 3)
        io.write_video(f"{out_dir}/{sha}.mp4", torch.from_numpy(grid), fps=24, video_codec="libx264",
                       options={"crf": "20", "preset": "veryfast"})
        rows = sorted(set(np.linspace(0, t - 1, 4).round().astype(int)))
        still = np.concatenate([grid[i] for i in rows], axis=0)
        Image.fromarray(still).resize((still.shape[1] // 2, still.shape[0] // 2)).save(f"{out_dir}/{sha}.png")
        src = "hudnav" if "Hud" in meta.loc[sha, "local_path"] else "s1nav"
        print(f"{sha[:12]} ({src}): {t} frames, still rows at frames {rows} -> {out_dir}/{sha}.png")


if __name__ == "__main__":
    roll = f"{ROOT}/outputs/runs_full29k/{sys.argv[2]}" if len(sys.argv) > 2 else ROLL
    configs = sys.argv[3].split(",") if len(sys.argv) > 3 else CONFIGS
    main(int(sys.argv[1]), roll, configs)
