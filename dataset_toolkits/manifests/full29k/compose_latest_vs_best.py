"""Grid videos GT | latest | best for the latest-vs-best LoRA comparison, with per-frame metrics.

For every episode with both <roll>/latest/XL_<sha>.mp4 and <roll>/best/XL_<sha>.mp4 writes
  <roll>/grids/<episode>_<seen|heldout>.mp4  top: the three videos (each model panel shows its LPIPS and PSNR against
                                             the ground-truth frame); bottom: LPIPS and PSNR curves of both models
                                             over the whole rollout, with a cursor at the current frame
  <roll>/grids/<episode>_<seen|heldout>.png  still: frames 0, 150, 300, 450, 599 (rows) x GT | latest | best
  <roll>/grids/per_frame.csv                 config, episode, frame, lpips, psnr, detail
  <roll>/grids/summary.csv                   per episode and config: means over all frames and over the last 50
Metrics (rollout frame i vs ground-truth frame i of rgb.mp4):
  lpips   perceptual distance, AlexNet LPIPS at 180x320 (0 = identical, ~0.7 = unrelated scenes); lower is better
  psnr    pixel fidelity in dB at full resolution; higher is better
  detail  mean |Laplacian| of the grayscale frame at 180x320; collapsed rollouts (flat or single-texture views) drop
          below ~0.035. "collapsed" = last-50-frame lpips > 0.55 and detail < 0.035, the rule used for eval16.

Usage: python compose_latest_vs_best.py <roll dir>     (run where a GPU is available; CPU works but is slow)
"""
import os
import sys

import lpips
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.io as io
from PIL import Image, ImageDraw, ImageFont

DS = "datasets/luanti_full29k"
VAL = "dataset_toolkits/manifests/full29k/val_sha256.txt"
CONFIGS = {"latest": "latest (voxel 70k, camera 140k, pixel 70k)", "best": "best (voxel 20k, camera 116k, pixel 61k)"}
COLORS = {"latest": "#d95f02", "best": "#1b9e77"}
STRIP_H = 240
torch.set_grad_enabled(False)
dev = "cuda" if torch.cuda.is_available() else "cpu"


def small(x):  # (T, H, W, 3) uint8 -> (T, 3, 180, 320) in [-1, 1]
    x = x.permute(0, 3, 1, 2).float() / 127.5 - 1
    return F.interpolate(x, size=(180, 320), mode="bilinear", antialias=True, align_corners=False)


def metrics(net, video, gt):
    lp, ps, de = [], [], []
    lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=dev).view(1, 1, 3, 3)
    for i in range(0, len(video), 32):
        v, g = video[i:i + 32].to(dev), gt[i:i + 32].to(dev)
        vs, gs = small(v), small(g)
        lp.append(net(vs, gs).flatten())
        mse = ((v.float() - g.float()) ** 2).mean((1, 2, 3))
        ps.append(10 * torch.log10(255 ** 2 / mse.clamp(min=1e-8)))
        de.append(F.conv2d(vs.mean(1, keepdim=True), lap).abs().mean((1, 2, 3)))
    return [torch.cat(m).cpu().numpy() for m in (lp, ps, de)]


def text(img, xy, s, size=22):
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=size)
    box = d.textbbox(xy, s, font=font)
    d.rectangle([box[0] - 6, box[1] - 4, box[2] + 6, box[3] + 4], fill=(0, 0, 0))
    d.text(xy, s, fill=(255, 255, 255), font=font)


def strip(m, width, t):
    """LPIPS and PSNR curves of both configs; returns the image and a function frame -> cursor x in pixels."""
    fig, axes = plt.subplots(1, 2, figsize=(width / 100, STRIP_H / 100), dpi=100)
    for ax, key, name in zip(axes, ("lpips", "psnr"), ("LPIPS vs GT (lower = better)", "PSNR vs GT, dB (higher = better)")):
        for c in CONFIGS:
            ax.plot(m[c][key], color=COLORS[c], lw=1.4, label=c)
        ax.set_title(name, fontsize=10)
        ax.set_xlim(0, t - 1)
        ax.set_xlabel("frame", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left" if key == "lpips" else "upper right")
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    xs = [(ax.transData.transform((0, 0))[0], ax.transData.transform((t - 1, 0))[0]) for ax in axes]
    plt.close(fig)
    h = img.shape[0]
    return img, lambda i: [int(round(x0 + (x1 - x0) * i / (t - 1))) for x0, x1 in xs], h


def main(roll):
    meta = pd.read_csv(f"{DS}/metadata.csv").set_index("sha256")
    val = set(open(VAL).read().split())
    out_dir = f"{roll}/grids"
    os.makedirs(out_dir, exist_ok=True)
    net = lpips.LPIPS(net="alex", verbose=False).to(dev)
    shas = sorted(f[3:-4] for f in os.listdir(f"{roll}/latest") if f.endswith(".mp4"))
    rows, summary = [], []
    for sha in shas:
        if not os.path.exists(f"{roll}/best/XL_{sha}.mp4"):
            print(f"skip {sha[:12]}: best not done"); continue
        ep = meta.loc[sha, "local_path"].rsplit("/", 1)[1]
        split = "heldout" if sha in val else "seen"
        gt = io.read_video(f"{DS}/{meta.loc[sha, 'local_path']}/rgb.mp4", pts_unit="sec")[0]
        vids = {c: io.read_video(f"{roll}/{c}/XL_{sha}.mp4", pts_unit="sec")[0] for c in CONFIGS}
        t = min(len(gt), *(len(v) for v in vids.values()))
        gt = gt[:t]
        m = {}
        for c, v in vids.items():
            lp, ps, de = metrics(net, v[:t], gt)
            m[c] = {"lpips": lp, "psnr": ps, "detail": de}
            rows += [(c, ep, split, sha, i, lp[i], ps[i], de[i]) for i in range(t)]
            last = slice(t - 50, t)
            summary.append(dict(episode=ep, split=split, config=c, frames=t,
                                lpips_mean=lp.mean(), lpips_last50=lp[last].mean(),
                                psnr_mean=ps.mean(), psnr_last50=ps[last].mean(), detail_last50=de[last].mean(),
                                collapsed=bool(lp[last].mean() > 0.55 and de[last].mean() < 0.035)))
        W = gt.shape[2]
        strip_img, cursor, sh = strip(m, 3 * W, t)
        frames = []
        for i in range(t):
            panels = [Image.fromarray(gt[i].numpy())]
            text(panels[0], (10, 8), f"ground truth   ep {ep} ({split})")
            for c in CONFIGS:
                p = Image.fromarray(vids[c][i].numpy())
                text(p, (10, 8), CONFIGS[c])
                text(p, (10, p.height - 36), f"LPIPS {m[c]['lpips'][i]:.3f}   PSNR {m[c]['psnr'][i]:.1f} dB")
                panels.append(p)
            text(panels[0], (10, panels[0].height - 36), f"frame {i}/{t - 1}")
            top = np.concatenate([np.asarray(p) for p in panels], axis=1)
            bottom = strip_img.copy()
            for x in cursor(i):
                bottom[:, max(x - 1, 0):x + 1] = 0
            frames.append(np.concatenate([top, bottom[:, :top.shape[1]]], axis=0))
        grid = np.stack(frames)
        if grid.shape[1] % 2:  # libx264 needs even sizes
            grid = grid[:, :-1]
        name = f"{out_dir}/{ep}_{split}"
        io.write_video(f"{name}.mp4", torch.from_numpy(grid), fps=24, video_codec="libx264",
                       options={"crf": "20", "preset": "veryfast"})
        still = np.concatenate([grid[i, :gt.shape[1]] for i in sorted({0, 150, 300, 450, t - 1})], axis=0)
        Image.fromarray(still).resize((still.shape[1] // 2, still.shape[0] // 2)).save(f"{name}.png")
        print(f"{ep} ({split}): " + "  ".join(
            f"{c} lpips {m[c]['lpips'].mean():.3f}/{m[c]['lpips'][-50:].mean():.3f}"
            f" psnr {m[c]['psnr'].mean():.1f}" for c in CONFIGS), flush=True)
    pd.DataFrame(rows, columns=["config", "episode", "split", "sha256", "frame", "lpips", "psnr", "detail"]) \
        .to_csv(f"{out_dir}/per_frame.csv", index=False)
    s = pd.DataFrame(summary)
    s.to_csv(f"{out_dir}/summary.csv", index=False)
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
