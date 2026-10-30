"""Grid videos GT | static61k | lora | full for the seg10k pixel-denoiser trials, with whole-frame and agent-only metrics.

Inputs: <roll>/{lora,full}/XL_<sha>.mp4 and <roll>/../static61k/XL_<sha>.mp4 (rollouts_pixel_trials.slurm).
Writes
  <roll>/grids/<episode>.mp4   top: the four videos (each model panel shows its LPIPS and agent LPIPS for that frame);
                               bottom: LPIPS and agent-LPIPS curves of the three models, cursor at the current frame
  <roll>/grids/<episode>.png   still: frames 0, 150, 300, 450, 599 (rows) x the four panels
  <roll>/grids/per_frame.csv   config, episode, frame, lpips, psnr, agent_frac, agent_lpips, agent_psnr
  <roll>/grids/summary.csv     per episode and config: means over all frames
Metrics (rollout frame i vs ground-truth frame i of rgb.mp4; the camera is ground truth, so the frames stay aligned):
  lpips / psnr              whole frame, as in compose_latest_vs_best.py (AlexNet LPIPS at 180x320, PSNR at full res)
  agent_lpips / agent_psnr  only over the pixels where seg.mp4 shows an agent (classes of kind "entity" in
                            seg_classes.json; seg.mp4 is lossless, so colours match exactly). agent_lpips averages the
                            spatial LPIPS map over the agent mask. NaN on frames with under 0.5% agent pixels.
Usage: python compose_pixel_trials.py <roll dir>
"""
import json
import os
import sys

import av
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

DS = "datasets/seg10k/train"
CONFIGS = {"static61k": "static best (start, 61k)", "lora": "LoRA r32 trial", "full": "full fine-tune trial"}
COLORS = {"static61k": "#7570b3", "lora": "#d95f02", "full": "#1b9e77"}
MIN_AGENT_FRAC = 0.005
STRIP_H = 240
torch.set_grad_enabled(False)
dev = "cuda" if torch.cuda.is_available() else "cpu"


def small(x):  # (T, H, W, C) -> (T, C, 180, 320); uint8 frames go to [-1, 1]
    x = x.permute(0, 3, 1, 2).float()
    x = x / 127.5 - 1 if x.shape[1] == 3 else x
    return F.interpolate(x, size=(180, 320), mode="bilinear", antialias=True, align_corners=False)


def agent_mask(path):
    """(T, H, W, 1) float mask of agent pixels from seg.mp4."""
    classes = json.load(open(f"{path}/seg_classes.json"))["classes"]
    colors = np.array([c["color"] for c in classes if c["kind"] == "entity"], dtype=np.uint8).reshape(-1, 3)
    with av.open(f"{path}/seg.mp4") as container:  # frame by frame: the whole video at once needs several GB
        masks = [(f.to_ndarray(format="rgb24")[..., None, :] == colors).all(-1).any(-1)
                 for f in container.decode(video=0)]
    return torch.from_numpy(np.stack(masks)[..., None]).float()


def metrics(net, video, gt, mask):
    out = {k: [] for k in ("lpips", "psnr", "agent_frac", "agent_lpips", "agent_psnr")}
    for i in range(0, len(video), 32):
        v, g, m = video[i:i + 32].to(dev), gt[i:i + 32].to(dev), mask[i:i + 32].to(dev)
        lp_map = net(small(v), small(g))  # (b, 1, 180, 320)
        ms = small(m)
        se = ((v.float() - g.float()) ** 2).mean(-1, keepdim=True)  # (b, H, W, 1)
        frac = m.mean((1, 2, 3))
        ok = frac >= MIN_AGENT_FRAC
        agent_mse = (se * m).sum((1, 2, 3)) / m.sum((1, 2, 3)).clamp(min=1)
        out["lpips"].append(lp_map.mean((1, 2, 3)))
        out["psnr"].append(10 * torch.log10(255 ** 2 / se.mean((1, 2, 3)).clamp(min=1e-8)))
        out["agent_frac"].append(frac)
        out["agent_lpips"].append(torch.where(ok, (lp_map * ms).sum((1, 2, 3)) / ms.sum((1, 2, 3)).clamp(min=1e-6), torch.nan))
        out["agent_psnr"].append(torch.where(ok, 10 * torch.log10(255 ** 2 / agent_mse.clamp(min=1e-8)), torch.nan))
    return {k: torch.cat(v).cpu().numpy() for k, v in out.items()}


def text(img, xy, s, size=22):
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=size)
    box = d.textbbox(xy, s, font=font)
    d.rectangle([box[0] - 6, box[1] - 4, box[2] + 6, box[3] + 4], fill=(0, 0, 0))
    d.text(xy, s, fill=(255, 255, 255), font=font)


def strip(m, width, t):
    fig, axes = plt.subplots(1, 2, figsize=(width / 100, STRIP_H / 100), dpi=100)
    for ax, key, name in zip(axes, ("lpips", "agent_lpips"),
                             ("LPIPS, whole frame (lower = better)", "LPIPS, agent pixels only (lower = better)")):
        for c in CONFIGS:
            ax.plot(m[c][key], color=COLORS[c], lw=1.4, label=c)
        ax.set_title(name, fontsize=10)
        ax.set_xlim(0, t - 1)
        ax.set_xlabel("frame", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    xs = [(ax.transData.transform((0, 0))[0], ax.transData.transform((t - 1, 0))[0]) for ax in axes]
    plt.close(fig)
    return img, lambda i: [int(round(x0 + (x1 - x0) * i / (t - 1))) for x0, x1 in xs]


def main(roll):
    dirs = {"static61k": f"{roll}/../static61k", "lora": f"{roll}/lora", "full": f"{roll}/full"}
    meta = pd.read_csv(f"{DS}/metadata.csv").set_index("sha256")
    out_dir = f"{roll}/grids"
    os.makedirs(out_dir, exist_ok=True)
    net = lpips.LPIPS(net="alex", spatial=True, verbose=False).to(dev)
    shas = sorted(f[3:-4] for f in os.listdir(dirs["full"]) if f.endswith(".mp4"))
    rows, summary = [], []
    for sha in shas:
        if not all(os.path.exists(f"{d}/XL_{sha}.mp4") for d in dirs.values()):
            print(f"skip {sha[:12]}: not all configs done"); continue
        path = f"{DS}/{meta.loc[sha, 'local_path']}"
        ep = path.rsplit("/", 1)[1]
        gt = io.read_video(f"{path}/rgb.mp4", pts_unit="sec")[0]
        mask = agent_mask(path)
        vids = {c: io.read_video(f"{d}/XL_{sha}.mp4", pts_unit="sec")[0] for c, d in dirs.items()}
        t = min(len(gt), len(mask), *(len(v) for v in vids.values()))
        gt, mask = gt[:t], mask[:t]
        m = {}
        for c, v in vids.items():
            m[c] = metrics(net, v[:t], gt, mask)
            rows += [(c, ep, sha, i, *(m[c][k][i] for k in m[c])) for i in range(t)]
            summary.append(dict(episode=ep, config=c, frames=t, agent_frames=int((m[c]["agent_frac"] >= MIN_AGENT_FRAC).sum()),
                                **{f"{k}_mean": np.nanmean(m[c][k]) for k in ("lpips", "psnr", "agent_lpips", "agent_psnr")}))
        W = gt.shape[2]
        strip_img, cursor = strip(m, 4 * W, t)
        frames = []
        for i in range(t):
            panels = [Image.fromarray(gt[i].numpy())]
            text(panels[0], (10, 8), f"ground truth   ep {ep}")
            text(panels[0], (10, panels[0].height - 36), f"frame {i}/{t - 1}   agents {100 * m['full']['agent_frac'][i]:.1f}% of pixels")
            for c in CONFIGS:
                p = Image.fromarray(vids[c][i].numpy())
                text(p, (10, 8), CONFIGS[c])
                a = m[c]["agent_lpips"][i]
                text(p, (10, p.height - 36), f"LPIPS {m[c]['lpips'][i]:.3f}   agents {'-' if np.isnan(a) else f'{a:.3f}'}")
                panels.append(p)
            top = np.concatenate([np.asarray(p) for p in panels], axis=1)
            bottom = strip_img.copy()
            for x in cursor(i):
                bottom[:, max(x - 1, 0):x + 1] = 0
            frames.append(np.concatenate([top, bottom[:, :top.shape[1]]], axis=0))
        grid = np.stack(frames)
        if grid.shape[1] % 2:  # libx264 needs even sizes
            grid = grid[:, :-1]
        name = f"{out_dir}/{ep}"
        io.write_video(f"{name}.mp4", torch.from_numpy(grid), fps=24, video_codec="libx264",
                       options={"crf": "20", "preset": "veryfast"})
        still = np.concatenate([grid[i, :gt.shape[1]] for i in sorted({0, 150, 300, 450, t - 1})], axis=0)
        Image.fromarray(still).resize((still.shape[1] // 2, still.shape[0] // 2)).save(f"{name}.png")
        print(f"{ep}: " + "  ".join(f"{c} lpips {m[c]['lpips'].mean():.3f} agent {np.nanmean(m[c]['agent_lpips']):.3f}"
                                    for c in CONFIGS), flush=True)
    pd.DataFrame(rows, columns=["config", "episode", "sha256", "frame", "lpips", "psnr", "agent_frac", "agent_lpips",
                                "agent_psnr"]).to_csv(f"{out_dir}/per_frame.csv", index=False)
    s = pd.DataFrame(summary)
    s.to_csv(f"{out_dir}/summary.csv", index=False)
    print(s.round(3).to_string(index=False))
    print(s.groupby("config")[[c for c in s.columns if c.endswith("_mean")]].mean().round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
