"""TODO 4: assemble the 32-episode, 5-method (PERSIST / PERSIST+w0 / WorldMem / Oasis / GT)
comparison grid from the outputs already produced by scripts/eval_qualitative.py (PERSIST,
PERSIST+w0) and baselines/eval_{oasis,worldmem}_zeroshot.py (--save-video-dir).

No re-generation: this only crops/recombines already-rendered video panels and images.

Per episode, three stacked rows (5 columns: GT, PERSIST, PERSIST+w0, WorldMem, Oasis -- N/A
placeholders where a method doesn't support a modality, confirmed earlier: Oasis has no voxel or
camera-pose output at all; WorldMem has camera pose but no voxel/3D state):
  1. video strip
  2. voxel camera-POV strip (GT / PERSIST / PERSIST+w0 only)
  3. camera-trajectory thumbnails (PERSIST-vs-GT / PERSIST+w0-vs-GT / WorldMem-solo -- WorldMem's
     pose is in its own coordinate frame, not directly overlayable with PERSIST's voxel-local
     camera representation without a conversion this script doesn't attempt, so it's plotted
     alone rather than misleadingly overlaid against GT in the wrong frame)

Cropping constants below (PANEL_WIDTHS) are derived from scripts/eval_qualitative.py's own
panel-sizing logic (panel_height=180; native pixel/POV renders are 640x360 -> resized width 320;
the isometric matplotlib render is a 3x3in@80dpi=240x240 square -> resized width 180) and were
confirmed against an actual rendered row's width (1960px = 320*5 + 180*2) during that script's
own smoke test -- not re-derived independently here, so if eval_qualitative.py's panel_height or
render sizes change, these constants need updating too.

Output: 4 grid files of 8 episodes each (grid_01.mp4 .. grid_04.mp4), per the chosen packaging.
"""
import glob
import os
from dataclasses import dataclass
from typing import Optional

import av
import imageio
import numpy as np
import tyro
from PIL import Image

PANEL_WIDTHS = [320, 320, 320, 180, 180, 320, 320]  # input, gen, GT, gen_iso, GT_iso, gen_pov, GT_pov
PANEL_OFFSETS = np.cumsum([0] + PANEL_WIDTHS)  # start x of each panel


def read_video_frames(path: str) -> np.ndarray:
    container = av.open(path)
    frames = [f.to_ndarray(format="rgb24") for f in container.decode(container.streams.video[0])]
    container.close()
    return np.stack(frames)  # (T, H, W, C)


def crop_panel(frames: np.ndarray, panel_idx: int, panel_height: int) -> np.ndarray:
    """Crop panel `panel_idx` out of a rendered row. Also crops height down to `panel_height`
    (not just the panel's x-range): imageio's ffmpeg writer pads saved videos to a macro-block-
    compatible size (observed: 1960x180 -> 1968x192 -- see the "resizing from ... to ..." warning
    at save time), so a saved row's actual height is >= panel_height, with the padding appended
    at the bottom/right. Left-anchored x offsets are unaffected; height needs explicit cropping."""
    x0, x1 = PANEL_OFFSETS[panel_idx], PANEL_OFFSETS[panel_idx + 1]
    return frames[:, :panel_height, x0:x1]


def na_panel(height: int, width: int, n_frames: int, text: str = "N/A") -> np.ndarray:
    img = Image.new("RGB", (width, height), color=(30, 30, 30))
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img)
    draw.text((width // 2 - 12, height // 2 - 6), text, fill=(150, 150, 150))
    frame = np.array(img)
    return np.repeat(frame[None], n_frames, axis=0)


def resize_frames(frames: np.ndarray, height: int) -> np.ndarray:
    out = []
    for f in frames:
        img = Image.fromarray(f)
        w = int(img.width * height / img.height)
        out.append(np.array(img.resize((w, height))))
    return np.stack(out)


def load_and_resize_image(path: str, height: int, width: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((width, height))
    return np.array(img)


@dataclass
class Args:
    persist_dir: str = "outputs/rich32_persist"
    persist_w0_dir: str = "outputs/rich32_persist_w0"
    worldmem_video_dir: str = "outputs/rich32_worldmem_video"
    oasis_video_dir: str = "outputs/rich32_oasis_video"
    num_episodes: int = 32
    episodes_per_grid: int = 8
    panel_height: int = 180
    fps: int = 12
    output_dir: str = "outputs/comparison_grid"


def main(args: Args):
    os.makedirs(args.output_dir, exist_ok=True)
    h = args.panel_height

    # Map PERSIST's episode index (0..31, in dataset order) to the level_XXX ids used by the
    # baseline scripts (sorted level_*/rgb.mp4 glob, same ordering both scripts already use).
    level_ids = [f"level_{i+1:03d}" for i in range(args.num_episodes)]

    grid_rows_all = []
    for ep_idx, level_id in enumerate(level_ids):
        persist_video_path = os.path.join(args.persist_dir, f"episode_{ep_idx:02d}.mp4")
        w0_video_path = os.path.join(args.persist_w0_dir, f"episode_{ep_idx:02d}.mp4")
        worldmem_video_path = os.path.join(args.worldmem_video_dir, f"{level_id}_gen.mp4")
        oasis_video_path = os.path.join(args.oasis_video_dir, f"{level_id}_gen.mp4")
        worldmem_pose_path = os.path.join(args.worldmem_video_dir, f"{level_id}_pose.npy")

        if not (os.path.exists(persist_video_path) and os.path.exists(w0_video_path)):
            print(f"Skipping {level_id}: missing PERSIST/PERSIST+w0 output")
            continue

        persist_frames = read_video_frames(persist_video_path)
        w0_frames = read_video_frames(w0_video_path)
        n = min(persist_frames.shape[0], w0_frames.shape[0])

        gt_video = crop_panel(persist_frames[:n], 2, h)
        persist_video = crop_panel(persist_frames[:n], 1, h)
        w0_video = crop_panel(w0_frames[:n], 1, h)
        gt_pov = crop_panel(persist_frames[:n], 6, h)
        persist_pov = crop_panel(persist_frames[:n], 5, h)
        w0_pov = crop_panel(w0_frames[:n], 5, h)

        worldmem_video = (
            resize_frames(read_video_frames(worldmem_video_path), h)[:n]
            if os.path.exists(worldmem_video_path)
            else na_panel(h, 320, n, "no video")
        )
        oasis_video = (
            resize_frames(read_video_frames(oasis_video_path), h)[:n]
            if os.path.exists(oasis_video_path)
            else na_panel(h, 320, n, "no video")
        )

        def pad_or_trim(frames, target_n):
            if frames.shape[0] >= target_n:
                return frames[:target_n]
            pad = np.repeat(frames[-1:], target_n - frames.shape[0], axis=0)
            return np.concatenate([frames, pad], axis=0)

        worldmem_video = pad_or_trim(worldmem_video, n)
        oasis_video = pad_or_trim(oasis_video, n)

        video_row = np.concatenate([gt_video, persist_video, w0_video, worldmem_video, oasis_video], axis=2)

        pov_na = na_panel(h, 320, n, "no voxels")
        pov_row = np.concatenate([gt_pov, persist_pov, w0_pov, pov_na, pov_na], axis=2)

        traj_w = 240
        persist_traj_path = os.path.join(args.persist_dir, f"episode_{ep_idx:02d}_trajectory.png")
        w0_traj_path = os.path.join(args.persist_w0_dir, f"episode_{ep_idx:02d}_trajectory.png")
        persist_traj = load_and_resize_image(persist_traj_path, h, traj_w) if os.path.exists(persist_traj_path) else na_panel(h, traj_w, 1, "?")[0]
        w0_traj = load_and_resize_image(w0_traj_path, h, traj_w) if os.path.exists(w0_traj_path) else na_panel(h, traj_w, 1, "?")[0]

        if os.path.exists(worldmem_pose_path):
            worldmem_traj = _plot_worldmem_solo_trajectory(worldmem_pose_path, h, traj_w)
        else:
            worldmem_traj = na_panel(h, traj_w, 1, "no pose")[0]

        traj_na = na_panel(h, traj_w, 1, "n/a")[0]
        traj_still = np.concatenate([np.zeros((h, 320, 3), dtype=np.uint8), persist_traj, w0_traj, worldmem_traj, traj_na], axis=1)
        traj_row = np.repeat(traj_still[None], n, axis=0)

        max_w = max(video_row.shape[2], pov_row.shape[2], traj_row.shape[2])

        def pad_w(frames, w):
            if frames.shape[2] >= w:
                return frames
            return np.pad(frames, ((0, 0), (0, 0), (0, w - frames.shape[2]), (0, 0)))

        episode_frames = np.concatenate(
            [pad_w(video_row, max_w), pad_w(pov_row, max_w), pad_w(traj_row, max_w)], axis=1
        )
        grid_rows_all.append(episode_frames)
        print(f"{level_id}: assembled ({episode_frames.shape})")

    n_grids = (len(grid_rows_all) + args.episodes_per_grid - 1) // args.episodes_per_grid
    for g in range(n_grids):
        chunk = grid_rows_all[g * args.episodes_per_grid : (g + 1) * args.episodes_per_grid]
        max_len = max(c.shape[0] for c in chunk)
        frames = []
        for i in range(max_len):
            rows = [c[min(i, c.shape[0] - 1)] for c in chunk]
            frames.append(np.concatenate(rows, axis=0))
        out_path = os.path.join(args.output_dir, f"grid_{g+1:02d}.mp4")
        imageio.mimsave(out_path, frames, fps=args.fps)
        print(f"Saved {out_path} ({len(chunk)} episodes)")


def _plot_worldmem_solo_trajectory(pose_path: str, height: int, width: int) -> np.ndarray:
    """WorldMem's predicted (x, y, z, pitch, yaw) pose is in its own coordinate frame -- not
    directly overlayable with PERSIST's voxel-local camera representation without a conversion
    this script doesn't attempt (utils/camera_util.py has minetest_cam_to_worldmem_cam /
    worldmem_cam_to_minetest_cam for exactly this, unused here for scope reasons), so this plots
    WorldMem's own path alone rather than misleadingly overlaying it against GT in the wrong
    frame."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    poses = np.load(pose_path)
    fig, ax = plt.subplots(figsize=(width / 80, height / 80), dpi=80)
    ax.plot(poses[:, 0], poses[:, 2], color="tab:orange", linewidth=1.2)
    ax.scatter([poses[0, 0]], [poses[0, 2]], color="tab:orange", s=10)
    ax.set_title("WorldMem (own frame)", fontsize=6)
    ax.tick_params(labelsize=5)
    fig.tight_layout(pad=0.2)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return np.array(Image.fromarray(buf).resize((width, height)))


if __name__ == "__main__":
    main(tyro.cli(Args))
