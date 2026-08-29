"""10-column, 1-row-per-episode grid built from the dense (--voxel-render-stride 1) re-run of
outputs/frame_analysis_persist{,_w0} -- the only two runs where the voxel 3D-isometric and
camera-POV panels actually change frame-to-frame (the earlier stride=150 runs held those panels
frozen at t=0, which is what triggered this script's existence).

Columns (left to right):
  input | GT video | PERSIST video | PERSIST+w0 video |
  GT voxel 3D | PERSIST voxel 3D | PERSIST+w0 voxel 3D |
  GT voxel POV | PERSIST voxel POV | PERSIST+w0 voxel POV
(10 columns x 6 episode rows, one output file -- unlike scripts/assemble_comparison_grid.py's
32-episode/4-file split, 6 rows comfortably fits one video.)

No re-generation, no GPU: every panel is cropped from the already-rendered 7-panel
episode_XX.mp4 files that scripts/eval_qualitative.py saves (same PANEL_WIDTHS/PANEL_OFFSETS
convention as scripts/assemble_comparison_grid.py -- see that file for why the height crop is
necessary). Per-frame metrics (gen PSNR on the video panels, gen voxel IoU/class-accuracy on the
3D-isometric panels) are already baked into the source pixels by eval_qualitative.py's own
draw_text calls, so they come along for free without a dedicated metrics column -- GT panels have
no baked metric (nothing to compare GT against itself).
"""
import os
import sys
from dataclasses import dataclass

import av
import imageio
import numpy as np
import tyro
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# input, gen, GT, gen_iso, GT_iso, gen_pov, GT_pov -- eval_qualitative.py's own 7-panel layout.
PANEL_WIDTHS = [320, 320, 320, 180, 180, 320, 320]
PANEL_OFFSETS = np.cumsum([0] + PANEL_WIDTHS)
PANEL_H = 180
HEADER_H = 28

COLUMNS = [
    ("input", 320), ("GT video", 320), ("PERSIST video", 320), ("PERSIST+w0 video", 320),
    ("GT voxel 3D", 180), ("PERSIST voxel 3D", 180), ("PERSIST+w0 voxel 3D", 180),
    ("GT voxel POV", 320), ("PERSIST voxel POV", 320), ("PERSIST+w0 voxel POV", 320),
]
TOTAL_W = sum(w for _, w in COLUMNS)


def read_video_frames(path: str) -> np.ndarray:
    container = av.open(path)
    frames = [f.to_ndarray(format="rgb24") for f in container.decode(container.streams.video[0])]
    container.close()
    return np.stack(frames)


def crop_panel(frames: np.ndarray, panel_idx: int, panel_height: int = PANEL_H) -> np.ndarray:
    x0, x1 = PANEL_OFFSETS[panel_idx], PANEL_OFFSETS[panel_idx + 1]
    return frames[:, :panel_height, x0:x1]


def pad_or_trim(frames: np.ndarray, target_n: int) -> np.ndarray:
    if frames.shape[0] >= target_n:
        return frames[:target_n]
    pad = np.repeat(frames[-1:], target_n - frames.shape[0], axis=0)
    return np.concatenate([frames, pad], axis=0)


def make_header() -> np.ndarray:
    img = Image.new("RGB", (TOTAL_W, HEADER_H), color=(20, 20, 20))
    draw = ImageDraw.Draw(img)
    x = 0
    for label, w in COLUMNS:
        draw.text((x + 4, 6), label, fill=(255, 255, 0))
        draw.line([(x, 0), (x, HEADER_H)], fill=(80, 80, 80))
        x += w
    return np.array(img)


def row_label(summary_by_ep: dict, ep_idx: int) -> str:
    p = summary_by_ep.get("persist", {}).get(ep_idx)
    w = summary_by_ep.get("w0", {}).get(ep_idx)
    if p is None or w is None:
        return f"Ep {ep_idx}"
    return (
        f"Ep {ep_idx}  mean PSNR persist={p['mean_psnr_db']:.1f}dB w0={w['mean_psnr_db']:.1f}dB  "
        f"IoU persist={p['mean_voxel_iou']:.2f} w0={w['mean_voxel_iou']:.2f}  "
        f"acc persist={p['mean_class_accuracy']:.2f} w0={w['mean_class_accuracy']:.2f}"
    )


def draw_row_label(row: np.ndarray, text: str) -> np.ndarray:
    """Burn a one-line per-episode summary-metric label into the top-left corner of a row's first
    frame's input panel -- the 10 columns already carry per-frame metrics baked in by
    eval_qualitative.py (see module docstring); this adds the episode-level means from
    summary.json on top, since those aren't baked into any panel."""
    row = row.copy()
    img = Image.fromarray(row[0])
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, PANEL_H - 14), (min(TOTAL_W, 900), PANEL_H)], fill=(0, 0, 0))
    draw.text((2, PANEL_H - 13), text, fill=(255, 255, 0))
    row[0] = np.array(img)
    return row


@dataclass
class Args:
    persist_dir: str = "outputs/frame_analysis_persist"
    persist_w0_dir: str = "outputs/frame_analysis_persist_w0"
    num_episodes: int = 6
    fps: int = 12
    output_dir: str = "outputs/dense_voxel_grid"


def main(args: Args):
    os.makedirs(args.output_dir, exist_ok=True)
    header = make_header()

    import json

    def load_summary(path: str) -> dict:
        # Tolerate a summary.json that doesn't exist yet -- e.g. smoke-testing this script against
        # a run that has written some episode_XX.mp4 files but hasn't finished (summary.json is
        # only written once, at the very end of scripts.eval_qualitative.py's run).
        try:
            with open(path) as f:
                return {e["episode_index"]: e for e in json.load(f)}
        except FileNotFoundError:
            print(f"WARNING: {path} not found yet -- per-episode metric labels will be omitted")
            return {}

    summary_by_ep = {
        "persist": load_summary(os.path.join(args.persist_dir, "summary.json")),
        "w0": load_summary(os.path.join(args.persist_w0_dir, "summary.json")),
    }

    episode_rows = []
    for ep_idx in range(args.num_episodes):
        persist_video_path = os.path.join(args.persist_dir, f"episode_{ep_idx:02d}.mp4")
        w0_video_path = os.path.join(args.persist_w0_dir, f"episode_{ep_idx:02d}.mp4")
        if not (os.path.exists(persist_video_path) and os.path.exists(w0_video_path)):
            print(f"Skipping episode {ep_idx}: missing PERSIST/PERSIST+w0 output")
            continue

        persist_frames = read_video_frames(persist_video_path)
        w0_frames = read_video_frames(w0_video_path)
        n = min(persist_frames.shape[0], w0_frames.shape[0])
        p, w = persist_frames[:n], w0_frames[:n]

        panels = [
            crop_panel(p, 0),  # input
            crop_panel(p, 2),  # GT video
            crop_panel(p, 1),  # PERSIST video
            crop_panel(w, 1),  # PERSIST+w0 video
            crop_panel(p, 4),  # GT voxel 3D
            crop_panel(p, 3),  # PERSIST voxel 3D
            crop_panel(w, 3),  # PERSIST+w0 voxel 3D
            crop_panel(p, 6),  # GT voxel POV
            crop_panel(p, 5),  # PERSIST voxel POV
            crop_panel(w, 5),  # PERSIST+w0 voxel POV
        ]
        panels = [pad_or_trim(panel, n) for panel in panels]
        row = np.concatenate(panels, axis=2)  # (n, PANEL_H, TOTAL_W, 3)
        row = draw_row_label(row, row_label(summary_by_ep, ep_idx))
        episode_rows.append(row)
        print(f"Episode {ep_idx}: assembled row {row.shape}")

    max_len = max(r.shape[0] for r in episode_rows)
    frames = []
    for i in range(max_len):
        rows = [r[min(i, r.shape[0] - 1)] for r in episode_rows]
        frames.append(np.concatenate([header] + rows, axis=0))
    out_path = os.path.join(args.output_dir, "dense_voxel_grid.mp4")
    imageio.mimsave(out_path, frames, fps=args.fps)
    print(f"Saved {out_path} ({len(episode_rows)} episodes, {frames[0].shape})")


if __name__ == "__main__":
    main(tyro.cli(Args))
