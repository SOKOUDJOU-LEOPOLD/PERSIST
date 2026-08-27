"""TODO 4 (v2, per feedback on the first layout): one row per episode, 13 labeled columns,
instead of 3 stacked rows per episode. The original design packed 8 episodes x 3 rows into a
tall, narrow 1600x4320 video that most players shrink to illegibility; laying every modality out
in a single wide row per episode gives a much better aspect ratio for actual viewing.

No re-generation: every panel here is cropped or reused from the SAME already-rendered outputs as
the first version (scripts/eval_qualitative.py's PERSIST/PERSIST+w0 runs,
baselines/eval_{oasis,worldmem}_zeroshot.py's --save-video-dir). The one new computation is a
standalone GT-only camera trajectory plot per episode -- the original design only ever saved GT
*overlaid* with a prediction (see plot_camera_trajectory in eval_qualitative.py), never alone --
but this is a cheap, CPU-only dataset read (no GPU, no model), not a rollout.

Columns (left to right), each with a small header label:
  input | GT | PERSIST | PERSIST+w0 | WorldMem | Oasis |
  GT voxels | PERSIST voxels | PERSIST+w0 voxels |
  GT trajectory | PERSIST trajectory | PERSIST+w0 trajectory | WorldMem trajectory
(13 columns x up to 32 episode rows, split into 4 grid files of 8 rows each per the earlier
choice.) No Oasis trajectory column: confirmed earlier it has no pose output whatsoever.
WorldMem's trajectory is plotted in its own coordinate frame, not overlaid on GT -- see
_plot_worldmem_solo_trajectory's docstring for why.

Cropping constants (PANEL_WIDTHS) are derived from scripts/eval_qualitative.py's own panel-sizing
logic and were confirmed against an actual rendered row's width during that script's own smoke
test -- see that constant's comment. If eval_qualitative.py's panel_height or render sizes
change, these need updating too.
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

# input, gen, GT, gen_iso, GT_iso, gen_pov, GT_pov -- the 7-panel layout eval_qualitative.py
# already saves each PERSIST/PERSIST+w0 episode video in; unchanged from the first grid version.
PANEL_WIDTHS = [320, 320, 320, 180, 180, 320, 320]
PANEL_OFFSETS = np.cumsum([0] + PANEL_WIDTHS)

PANEL_H = 180
VIDEO_W = 320
TRAJ_W = 240
HEADER_H = 28

COLUMNS = [
    ("input", VIDEO_W), ("GT", VIDEO_W), ("PERSIST", VIDEO_W), ("PERSIST+w0", VIDEO_W),
    ("WorldMem", VIDEO_W), ("Oasis", VIDEO_W),
    ("GT voxels", VIDEO_W), ("PERSIST voxels", VIDEO_W), ("PERSIST+w0 voxels", VIDEO_W),
    ("GT traj", TRAJ_W), ("PERSIST traj", TRAJ_W), ("PERSIST+w0 traj", TRAJ_W), ("WorldMem traj", TRAJ_W),
]
TOTAL_W = sum(w for _, w in COLUMNS)


def read_video_frames(path: str) -> np.ndarray:
    container = av.open(path)
    frames = [f.to_ndarray(format="rgb24") for f in container.decode(container.streams.video[0])]
    container.close()
    return np.stack(frames)


def crop_panel(frames: np.ndarray, panel_idx: int, panel_height: int = PANEL_H) -> np.ndarray:
    """See module docstring: saved rows are padded by imageio's ffmpeg writer to macro-block-
    compatible dimensions, so height must be cropped explicitly, not assumed to match exactly."""
    x0, x1 = PANEL_OFFSETS[panel_idx], PANEL_OFFSETS[panel_idx + 1]
    return frames[:, :panel_height, x0:x1]


def resize_frames(frames: np.ndarray, height: int, width: int) -> np.ndarray:
    return np.stack([np.array(Image.fromarray(f).resize((width, height))) for f in frames])


def na_panel(height: int, width: int, n_frames: int, text: str = "N/A") -> np.ndarray:
    img = Image.new("RGB", (width, height), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)
    draw.text((width // 2 - 12, height // 2 - 6), text, fill=(150, 150, 150))
    return np.repeat(np.array(img)[None], n_frames, axis=0)


def static_image(path: str, height: int, width: int, n_frames: int) -> np.ndarray:
    img = np.array(Image.open(path).convert("RGB").resize((width, height)))
    return np.repeat(img[None], n_frames, axis=0)


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


def plot_solo_trajectory(pos_xz: np.ndarray, color: str, title: str, height: int, width: int) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(width / 80, height / 80), dpi=80)
    # A plain single-color line only shows *where* the camera went, not *which order* -- with a
    # real recorded action sequence the path densely criss-crosses itself (see this function's
    # module-level usage notes / the chat explanation), so direction of travel is impossible to
    # read from a flat line alone. Coloring by timestep (viridis: dark purple=start, yellow=end)
    # plus explicit start/end markers makes the temporal order visible.
    ax.plot(pos_xz[:, 0], pos_xz[:, 1], color="lightgray", linewidth=0.6, zorder=1)
    t = np.arange(len(pos_xz))
    ax.scatter(pos_xz[:, 0], pos_xz[:, 1], c=t, cmap="viridis", s=4, zorder=2)
    ax.scatter([pos_xz[0, 0]], [pos_xz[0, 1]], color="green", marker="s", s=25, zorder=3, label="start")
    ax.scatter([pos_xz[-1, 0]], [pos_xz[-1, 1]], color="red", marker="X", s=30, zorder=3, label="end")
    ax.set_title(title, fontsize=6)
    ax.tick_params(labelsize=5)
    ax.legend(fontsize=4, loc="upper right")
    fig.tight_layout(pad=0.2)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return np.array(Image.fromarray(buf).resize((width, height)))


def gt_camera_positions(dataset_root: str, instance: str, num_frames: int) -> np.ndarray:
    """GT camera (x, z) positions only, no model/GPU involved -- a plain dataset read. The
    original grid design never saved a GT-only trajectory (only GT overlaid with a prediction),
    so this is computed fresh here rather than reused, but it's the same order of cost as reading
    a file, not a rollout."""
    from data_loaders.minetest_latent_camera_action_dataset import MinetestLatentCameraActionEval

    dataset = MinetestLatentCameraActionEval(dataset_root, clip_len=num_frames, instances=instance, num_instances=1)
    camera = dataset[0]["camera"]  # (T, 10): [rot6d(6), trans(3), fov(1)]
    pos = camera[:, 6:9].numpy()
    return pos[:, [0, 2]]  # (T, 2): x, z


@dataclass
class Args:
    persist_dir: str = "outputs/rich32_persist"
    persist_w0_dir: str = "outputs/rich32_persist_w0"
    worldmem_video_dir: str = "outputs/rich32_worldmem_video"
    oasis_video_dir: str = "outputs/rich32_oasis_video"
    dataset_root: str = (
        "/home/lsokoudj/.cache/huggingface/hub/datasets--PERSIST-team--persist-eval-sample/"
        "snapshots/d3a3bee6560304397093ae66523451b9e0d067f7"
    )
    num_episodes: int = 32
    episodes_per_grid: int = 8
    fps: int = 12
    output_dir: str = "outputs/comparison_grid_v2"


def main(args: Args):
    os.makedirs(args.output_dir, exist_ok=True)
    level_ids = [f"level_{i+1:03d}" for i in range(args.num_episodes)]
    header = make_header()

    episode_rows = []
    for ep_idx, level_id in enumerate(level_ids):
        persist_video_path = os.path.join(args.persist_dir, f"episode_{ep_idx:02d}.mp4")
        w0_video_path = os.path.join(args.persist_w0_dir, f"episode_{ep_idx:02d}.mp4")
        worldmem_video_path = os.path.join(args.worldmem_video_dir, f"{level_id}_gen.mp4")
        oasis_video_path = os.path.join(args.oasis_video_dir, f"{level_id}_gen.mp4")
        worldmem_pose_path = os.path.join(args.worldmem_video_dir, f"{level_id}_pose.npy")
        persist_traj_path = os.path.join(args.persist_dir, f"episode_{ep_idx:02d}_trajectory.png")
        w0_traj_path = os.path.join(args.persist_w0_dir, f"episode_{ep_idx:02d}_trajectory.png")

        if not (os.path.exists(persist_video_path) and os.path.exists(w0_video_path)):
            print(f"Skipping {level_id}: missing PERSIST/PERSIST+w0 output")
            continue

        persist_frames = read_video_frames(persist_video_path)
        w0_frames = read_video_frames(w0_video_path)
        n = min(persist_frames.shape[0], w0_frames.shape[0])

        panels = [
            crop_panel(persist_frames[:n], 0),  # input
            crop_panel(persist_frames[:n], 2),  # GT video
            crop_panel(persist_frames[:n], 1),  # PERSIST video
            crop_panel(w0_frames[:n], 1),  # PERSIST+w0 video
        ]
        panels.append(
            resize_frames(read_video_frames(worldmem_video_path), PANEL_H, VIDEO_W)[:n]
            if os.path.exists(worldmem_video_path) else na_panel(PANEL_H, VIDEO_W, n, "no video")
        )
        panels.append(
            resize_frames(read_video_frames(oasis_video_path), PANEL_H, VIDEO_W)[:n]
            if os.path.exists(oasis_video_path) else na_panel(PANEL_H, VIDEO_W, n, "no video")
        )
        panels += [
            crop_panel(persist_frames[:n], 6),  # GT voxels
            crop_panel(persist_frames[:n], 5),  # PERSIST voxels
            crop_panel(w0_frames[:n], 5),  # PERSIST+w0 voxels
        ]

        # Standalone trajectory PNGs -- eval_qualitative.py only ever saved GT overlaid with a
        # prediction (persist_traj_path / w0_traj_path below), never a plain openable image for
        # GT alone or for WorldMem at all (its pose was only ever an .npy array, not viewable
        # directly). Saved here so both can be opened independently, not just seen inside the grid
        # video.
        traj_out_dir = os.path.join(args.output_dir, "trajectories")
        os.makedirs(traj_out_dir, exist_ok=True)

        try:
            gt_pos = gt_camera_positions(args.dataset_root, level_id, n)
            gt_traj = plot_solo_trajectory(gt_pos, "black", "GT", PANEL_H, TRAJ_W)
        except Exception as e:  # noqa: BLE001
            print(f"{level_id}: GT trajectory failed ({e}), using placeholder")
            gt_traj = na_panel(PANEL_H, TRAJ_W, 1, "?")[0]
        else:
            Image.fromarray(gt_traj).save(os.path.join(traj_out_dir, f"{level_id}_gt_trajectory.png"))
        panels.append(np.repeat(gt_traj[None], n, axis=0))

        panels.append(
            static_image(persist_traj_path, PANEL_H, TRAJ_W, n) if os.path.exists(persist_traj_path)
            else na_panel(PANEL_H, TRAJ_W, n, "?")
        )
        panels.append(
            static_image(w0_traj_path, PANEL_H, TRAJ_W, n) if os.path.exists(w0_traj_path)
            else na_panel(PANEL_H, TRAJ_W, n, "?")
        )
        if os.path.exists(worldmem_pose_path):
            poses = np.load(worldmem_pose_path)
            wm_traj = plot_solo_trajectory(poses[:, [0, 2]], "tab:orange", "WorldMem (own frame)", PANEL_H, TRAJ_W)
            Image.fromarray(wm_traj).save(os.path.join(traj_out_dir, f"{level_id}_worldmem_trajectory.png"))
            panels.append(np.repeat(wm_traj[None], n, axis=0))
        else:
            panels.append(na_panel(PANEL_H, TRAJ_W, n, "no pose"))

        panels = [pad_or_trim(p, n) for p in panels]
        row = np.concatenate(panels, axis=2)  # (n, PANEL_H, TOTAL_W, 3)
        episode_rows.append(row)
        print(f"{level_id}: assembled row {row.shape}")

    n_grids = (len(episode_rows) + args.episodes_per_grid - 1) // args.episodes_per_grid
    for g in range(n_grids):
        chunk = episode_rows[g * args.episodes_per_grid : (g + 1) * args.episodes_per_grid]
        max_len = max(c.shape[0] for c in chunk)
        frames = []
        for i in range(max_len):
            rows = [c[min(i, c.shape[0] - 1)] for c in chunk]
            frames.append(np.concatenate([header] + rows, axis=0))
        out_path = os.path.join(args.output_dir, f"grid_{g+1:02d}.mp4")
        imageio.mimsave(out_path, frames, fps=args.fps)
        print(f"Saved {out_path} ({len(chunk)} episodes, {frames[0].shape})")


if __name__ == "__main__":
    main(tyro.cli(Args))
