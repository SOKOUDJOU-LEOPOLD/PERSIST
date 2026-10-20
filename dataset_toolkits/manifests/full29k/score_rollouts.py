"""Per-frame scores of rollout videos against the ground-truth episode (frames [0, N) of rgb.mp4).

For each <rollout_dir>/<config>/XL_<sha256>.mp4 computes, every STRIDE frames:
  lpips   perceptual distance to the ground-truth frame (AlexNet LPIPS, 0 = identical, ~0.7+ = unrelated)
  detail  image detail: mean absolute Laplacian of the grayscale frame (collapsed "inside the terrain" frames are
          flat, close-ups of a single texture, or uniform colour, so their detail drops far below a real scene)
and writes <rollout_dir>/scores.csv (config, sha256, source, frame, lpips, detail).

Usage: python score_rollouts.py <rollout_dir> <num_frames> [configs comma-separated]
"""
import os
import sys

import lpips
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.io as io

DS = "/scratch/user/lsokoudj/PERSIST/datasets/luanti_full29k"
STRIDE = 5
torch.set_grad_enabled(False)


def to_tensor(frames):  # (T, H, W, 3) uint8 -> (T, 3, 180, 320) in [-1, 1]
    x = frames.permute(0, 3, 1, 2).float() / 127.5 - 1
    return F.interpolate(x, size=(180, 320), mode="bilinear", antialias=True, align_corners=False)


def detail(x):  # x in [-1, 1], (T, 3, H, W) -> (T,)
    gray = x.mean(1, keepdim=True)
    lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=x.dtype).view(1, 1, 3, 3)
    return F.conv2d(gray, lap).abs().mean((1, 2, 3))


def main(roll, n, configs):
    meta = pd.read_csv(f"{DS}/metadata.csv").set_index("sha256")
    net = lpips.LPIPS(net="alex", verbose=False)
    rows, gt_cache = [], {}
    for cfg in configs:
        for f in sorted(os.listdir(f"{roll}/{cfg}")):
            if not f.endswith(".mp4"):
                continue
            sha = f[3:-4]
            if sha not in gt_cache:
                gt, _, _ = io.read_video(f"{DS}/{meta.loc[sha, 'local_path']}/rgb.mp4", pts_unit="sec")
                gt_cache[sha] = to_tensor(gt[:n:STRIDE])
            g = gt_cache[sha]
            v, _, _ = io.read_video(f"{roll}/{cfg}/{f}", pts_unit="sec")
            r = to_tensor(v[:n:STRIDE])
            t = min(len(r), len(g))
            lp = torch.cat([net(r[i:i + 16], g[i:i + 16]).flatten() for i in range(0, t, 16)])
            de = detail(r[:t])
            src = "hudnav" if "Hud" in meta.loc[sha, "local_path"] else "s1nav"
            for k in range(t):
                rows.append((cfg, sha, src, k * STRIDE, lp[k].item(), de[k].item()))
            print(f"{cfg:>40} {sha[:12]} ({src}): lpips@end {lp[-10:].mean():.3f}  detail@end {de[-10:].mean():.4f}", flush=True)
    out = pd.DataFrame(rows, columns=["config", "sha256", "source", "frame", "lpips", "detail"])
    if os.path.exists(f"{roll}/scores.csv"):  # keep scores of configs not rescored in this run
        old = pd.read_csv(f"{roll}/scores.csv")
        out = pd.concat([old[~old.config.isin(configs)], out], ignore_index=True)
    out.to_csv(f"{roll}/scores.csv", index=False)
    print(f"wrote {roll}/scores.csv ({len(out)} rows)")


if __name__ == "__main__":
    roll, n = sys.argv[1], int(sys.argv[2])
    configs = sys.argv[3].split(",") if len(sys.argv) > 3 else sorted(
        d for d in os.listdir(roll) if os.path.isdir(f"{roll}/{d}"))
    main(roll, n, configs)
