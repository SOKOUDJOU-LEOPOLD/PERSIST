"""Qualitative comparison grid: input frame / generated video / GT video / generated voxels 3D /
GT voxels 3D, per episode, stacked into one grid video -- for visually sanity-checking rollout
quality alongside the aggregate FVD number from scripts/eval_fvd.py.

Motivated directly by a manager review request: the aggregate FVD score doesn't show *why* a
rollout is good or bad, and no script in this repo (including our own eval_fvd.py) saves any
rollout output -- everything gets reduced to I3D features and discarded. This fills that gap.

Reuses:
  - scripts/run_inference.py's checkpoint/dataset resolution and context-building helpers.
  - pipelines/pipeline.py's VoxelFirstPipeline.decode_voxels() -- the same "generate voxel
    latents, decode to a class grid" capability confirmed to exist but go unused anywhere in
    the repo (see the "does PERSIST evaluate against voxels" discussion this project started
    from).
  - The ground-truth voxel_classes already loaded per episode by MinetestLatentCameraActionEval
    (sample_voxel_classes=True), likewise unused by any evaluation code before this script.

New pieces (nothing in the repo does any of this):
  - Per-frame PSNR between generated and ground-truth pixel frames.
  - Per-frame voxel occupancy IoU between generated and ground-truth voxel grids. Occupancy is
    defined as "class id != empty_class_id", where empty_class_id is resolved dynamically from
    the dataset's own mt_voxel_classdict.json as whichever class maps to node_id 126 -- the
    Craftium/Luanti engine's CONTENT_AIR constant (gym_envs/craftium/src/mapnode.h:46) -- rather
    than a hardcoded index, since mt_voxel_classdict.json is rebuilt per dataset (index 38 means
    "air" in persist-eval-sample specifically; a hardcoded default of 0 was silently wrong here
    because class 0 never occurs in this dataset at all, making occupancy a degenerate all-True
    mask and IoU a meaningless 1.000 regardless of generation quality -- caught by exactly that
    symptom on a first smoke test). Override with --empty-class-id if a dataset's classdict
    doesn't expose node 126 for some reason.
  - A 3D voxel-grid renderer (matplotlib's ax.voxels(), colored by class id) -- there's no
    existing voxel-to-3D-image visualization anywhere in this repo (utils/voxel_rasterizer.py
    renders voxels into 2D screen-space *features* for conditioning the pixel denoiser, not a
    human-viewable 3D render). Rendered at a coarser stride than the pixel video (matplotlib's
    voxel plot is too slow to render every frame of a 150+-frame rollout) and held/repeated to
    stay in sync with the pixel panels.
  - A camera-accurate "what the camera sees" voxel render (utils/voxel_pov_render.py, reusing
    the actual rasterizer/camera-matrix chain instead of a schematic isometric view) -- directly
    comparable to the pixel panels at the same timestep, for spotting whether a pixel-space
    artifact is also present in the model's own 3D state or is a pixel-decoder-only hallucination.
  - class_accuracy: exact per-voxel class match rate, complementing the binary occupancy IoU.
  - A per-episode camera trajectory plot (predicted vs GT top-down path) and, optionally
    (--export-glb), a .glb export of occupied voxels per rendered frame for interactive
    inspection in a standard 3D viewer.

Rows now have 7 panels: input frame / gen video / GT video / gen voxels (isometric) / GT voxels
(isometric) / gen voxels (camera POV) / GT voxels (camera POV).

Example:
    uv run python -m scripts.eval_qualitative \
        --pipeline-variant XL \
        --checkpoints-namespace PERSIST-team \
        --dataset-repo PERSIST-team/persist-eval-sample \
        --num-instances 8 \
        --num-frames 150 \
        --output-dir outputs/qualitative
"""
import json
import os
import sys
from dataclasses import dataclass, field
from typing import List, Literal, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from accelerate import Accelerator
from einops import rearrange
from loguru import logger
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loaders.minetest_latent_camera_action_dataset import MinetestLatentCameraActionEval
from pipelines.pipeline import VoxelFirstPipeline, pipeline_variant_overrides
from scripts.run_inference import Checkpoints, build_context, resolve_checkpoints, resolve_dataset_root
from utils.voxel_glb_export import voxel_classes_to_glb
from utils.voxel_pov_render import render_voxel_pov


@dataclass
class Args:
    checkpoints: Checkpoints = field(default_factory=Checkpoints)
    checkpoints_namespace: Optional[str] = None
    checkpoints_repo_prefix: str = "persist-"
    checkpoints_revision: Optional[str] = None

    dataset_path: Optional[str] = None
    dataset_repo: Optional[str] = None
    dataset_revision: Optional[str] = None

    pipeline_variant: Literal["S", "XL"] = "XL"
    pipeline_path: str = "pipelines/persist_S_pipeline"

    eval_instances: Optional[str] = None
    num_instances: int = 8
    """How many episodes to include in the grid."""
    num_frames: int = 150
    """Rollout length for the qualitative sample -- deliberately shorter than the 200/400/600
    used for FVD scoring; this is for visual inspection, not a benchmark number."""

    use_camera_gt: bool = False
    include_initial_voxel_frame: bool = False
    use_kv_cache: bool = True

    empty_class_id: Optional[int] = None
    """Voxel class id treated as 'empty' for occupancy IoU. Auto-resolved from the dataset's
    mt_voxel_classdict.json (whichever class maps to Minetest node id 126, CONTENT_AIR) if
    unset -- see module docstring for why this can't be a fixed default."""
    voxel_render_stride: int = 20
    """Render one 3D voxel frame every N pixel frames (matplotlib's voxel plot is too slow to
    render every frame); held/repeated to stay in sync with the pixel panels' duration."""

    panel_height: int = 180
    fps: int = 12
    mixed_precision: Literal["no", "fp16", "bf16"] = "bf16"
    device: str = "cuda:0"
    output_dir: str = "outputs/qualitative"
    method_name: str = "PERSIST"
    """Label used in the camera trajectory plot legend -- set to "PERSIST+w0" when running with
    --include-initial-voxel-frame, so the two variants' trajectory plots are distinguishable."""
    export_glb: bool = False
    """Also export occupied voxels as .glb meshes at the same voxel_render_stride cadence
    (predicted and GT). Off by default -- real extra disk/time cost, opt-in."""


CONTENT_AIR = 126  # gym_envs/craftium/src/mapnode.h:46 -- Craftium/Luanti engine constant.


def resolve_empty_class_id(dataset_root: str) -> int:
    """Find the class index that maps to Minetest's air node in this dataset's own
    mt_voxel_classdict.json. Class indices are reassigned per dataset build (see
    dataset_toolkits/build_metadata.py), so this can't be a fixed constant -- and defaulting to
    0 is actively wrong on persist-eval-sample specifically (class 0 never occurs there at all,
    which silently made occupancy IoU degenerate to 1.000 regardless of generation quality on
    the first version of this script)."""
    with open(os.path.join(dataset_root, "mt_voxel_classdict.json")) as f:
        node_classes = json.load(f)["node_classes"]
    for class_id, (node_id, node_param) in node_classes.items():
        if node_id == CONTENT_AIR and node_param == 0:
            return int(class_id)
    raise ValueError(
        f"No class in {dataset_root}/mt_voxel_classdict.json maps to CONTENT_AIR "
        f"(node_id={CONTENT_AIR}, param=0). Pass --empty-class-id explicitly."
    )


def build_dataloader(args: Args, clip_len: int) -> DataLoader:
    root = resolve_dataset_root(args)
    dataset = MinetestLatentCameraActionEval(
        root, clip_len=clip_len, instances=args.eval_instances, num_instances=args.num_instances,
    )
    if len(dataset) == 0:
        raise ValueError(f"No episodes selected from {root}.")
    logger.info(f"Loaded {len(dataset)} episode(s) from {root}.")
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False, collate_fn=dataset.collate_fn)


def to_uint8_np(x: torch.Tensor) -> np.ndarray:
    """[-1, 1] float (C, H, W) -> uint8 (H, W, C)."""
    x = torch.clamp(((x + 1) / 2) * 255, 0, 255).byte()
    return rearrange(x, "c h w -> h w c").cpu().numpy()


def resize_panel(frame: np.ndarray, height: int) -> np.ndarray:
    img = Image.fromarray(frame)
    w = int(img.width * height / img.height)
    return np.array(img.resize((w, height)))


def draw_text(frame: np.ndarray, text: str) -> np.ndarray:
    img = Image.fromarray(frame.copy())
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, min(180, img.width), 16], fill=(0, 0, 0))
    draw.text((2, 1), text, fill=(255, 255, 0))
    return np.array(img)


def psnr(gen: np.ndarray, real: np.ndarray) -> float:
    mse = np.mean((gen.astype(np.float32) - real.astype(np.float32)) ** 2)
    if mse < 1e-10:
        return 99.0
    return float(10 * np.log10(255.0**2 / mse))


def voxel_iou(gen_classes: np.ndarray, real_classes: np.ndarray, empty_id: int) -> float:
    """Binary occupancy IoU: does the generated grid agree with GT on occupied vs empty,
    ignoring which specific class occupies a cell. See `class_accuracy` for the stricter
    per-class version (TODO 8: pixel-to-voxel occupancy AND class accuracy table)."""
    gen_occ = gen_classes != empty_id
    real_occ = real_classes != empty_id
    union = np.logical_or(gen_occ, real_occ).sum()
    if union == 0:
        return 1.0
    inter = np.logical_and(gen_occ, real_occ).sum()
    return float(inter / union)


def class_accuracy(gen_classes: np.ndarray, real_classes: np.ndarray) -> float:
    """Exact per-voxel class-match rate (argmax class == GT class), over every cell including
    empty/air -- a stricter complement to `voxel_iou`'s binary occupied-vs-empty agreement.
    Two grids that agree on occupancy everywhere but disagree on which block type fills each
    occupied cell would score 1.0 on voxel_iou but less than 1.0 here."""
    return float((gen_classes == real_classes).mean())


def camera_forward_local(camera_6d_rot: torch.Tensor) -> np.ndarray:
    """View direction in the voxel grid's local frame, from the dataset's 6D rotation
    representation (see data_loaders/minetest_latent_camera_action_dataset.py:_extract_camera_representation
    and utils/camera_util.py:rotation_6d_to_matrix). `extrinsics_local` -- what the 6D rotation is
    derived from -- is documented as "World-to-Camera transformation (standard CV convention)"
    (utils/camera_util.py's module docstring), so the camera's own forward axis ([0, 0, 1] in
    OpenCV convention: +Z out of the lens) is mapped into the voxel-local frame via the inverse
    (transpose, since the matrix is orthonormal) rotation. This sign/axis convention is a
    best-effort reading of that docstring, not independently verified against a rendered
    ground-truth trajectory -- it's for a visual direction indicator, not a numeric result, so
    shipping a reasonable, clearly-documented choice beats blocking on that verification."""
    from utils.camera_util import rotation_6d_to_matrix

    R = rotation_6d_to_matrix(camera_6d_rot.unsqueeze(0))[0]  # (3, 3), world(voxel-local)-to-camera
    forward_cam = torch.tensor([0.0, 0.0, 1.0])
    return (R.T @ forward_cam).numpy()


def plot_camera_trajectory(trajectories: dict, out_path: str) -> None:
    """Static top-down (x, z) plot of one or more camera paths over a rollout, for visually
    comparing drift between predicted and ground-truth camera trajectories (TODO 4/5: no such
    visualization existed anywhere in this repo before this addition).

    Args:
        trajectories: {label: (T, 10) camera tensor or None}. Position is `camera[:, 6:9]`
            (`cam_pos_local`, the dataset's own per-timestep camera translation in the voxel
            grid's local frame -- see `data_loaders/minetest_latent_camera_action_dataset.py:
            _extract_camera_representation`), plotted as x vs z (the ground plane in this
            engine's convention). Entries with value None (e.g. Oasis, which has no pose output
            at all) are skipped, not drawn as empty/flat lines.
    """
    fig, ax = plt.subplots(figsize=(3, 3), dpi=80)
    colors = {"PERSIST": "tab:blue", "PERSIST+w0": "tab:cyan", "WorldMem": "tab:orange", "GT": "black"}
    for label, camera in trajectories.items():
        if camera is None:
            continue
        pos = camera[:, 6:9].cpu().numpy() if isinstance(camera, torch.Tensor) else camera[:, 6:9]
        ax.plot(pos[:, 0], pos[:, 2], label=label, color=colors.get(label), linewidth=1.5)
        ax.scatter([pos[0, 0]], [pos[0, 2]], color=colors.get(label), s=15, marker="o")  # start
    ax.set_xlabel("x", fontsize=7)
    ax.set_ylabel("z", fontsize=7)
    ax.set_title("camera trajectory (top-down)", fontsize=8)
    ax.legend(fontsize=6, loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(out_path, dpi=80)
    plt.close(fig)


def render_voxel_3d(classes: np.ndarray, empty_id: int, title: str, camera_dir: Optional[np.ndarray] = None) -> np.ndarray:
    """Render a single (X, Y, Z) class grid as a colored 3D voxel plot, with an optional camera
    position + view-direction arrow. The grid is "centered on the agent" by construction (per
    the paper), so the camera position marker is simply the grid center rather than trying to
    place it via cam_pos_local's own (undetermined) scale/offset relative to voxel-index units --
    the direction is the informative part of a frustum indicator anyway."""
    occ = classes != empty_id
    fig = plt.figure(figsize=(3, 3), dpi=80)
    ax = fig.add_subplot(projection="3d")
    if occ.any():
        cmap = plt.get_cmap("tab20")
        norm_classes = (classes % 20) / 20.0
        colors = cmap(norm_classes)
        colors[..., 3] = np.where(occ, 0.9, 0.0)
        ax.voxels(occ, facecolors=colors, edgecolor=None, linewidth=0)

    if camera_dir is not None:
        cx, cy, cz = np.array(classes.shape) / 2
        arrow_len = max(classes.shape) * 0.4
        ax.quiver(
            cx, cy, cz, camera_dir[0], camera_dir[1], camera_dir[2],
            length=arrow_len, color="red", linewidth=2, arrow_length_ratio=0.3,
        )
        ax.scatter([cx], [cy], [cz], color="red", s=20)

    ax.set_title(title, fontsize=8)
    ax.set_axis_off()
    ax.view_init(elev=25, azim=45)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf


def main(args: Args):
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    device = torch.device(args.device)

    dataset_root = resolve_dataset_root(args)
    empty_class_id = args.empty_class_id
    if empty_class_id is None:
        empty_class_id = resolve_empty_class_id(dataset_root)
        logger.info(f"Resolved empty_class_id={empty_class_id} (Craftium CONTENT_AIR) from {dataset_root}")

    model_config_overrides, sampler_args = pipeline_variant_overrides(args.pipeline_variant)
    logger.info(f"Loading persist_S pipeline (variant {args.pipeline_variant}) from: {args.pipeline_path}")
    pipe = VoxelFirstPipeline.from_pretrained(
        args.pipeline_path, device=device, accelerator=accelerator,
        custom_checkpoint_cfg=resolve_checkpoints(args),
        model_config_overrides=model_config_overrides,
        voxel_sampler_args=sampler_args or None, pixel_sampler_args=sampler_args or None,
    )

    dataloader = build_dataloader(args, clip_len=args.num_frames)
    os.makedirs(args.output_dir, exist_ok=True)

    row_videos = []
    summary = []

    for ep_idx, batch in enumerate(tqdm(dataloader, desc="Qualitative rollouts")):
        context = build_context(batch, args.num_frames, args.use_camera_gt, args.include_initial_voxel_frame)
        with accelerator.autocast():
            rollout = pipe(
                context, args.num_frames, output_latents=True, output_decoded_pixels=True,
                pixel_use_x0=False, use_camera_gt=args.use_camera_gt, keep_on_device=False,
                use_kv_cache=args.use_kv_cache, verbose=False,
            )
        gen_pixel = rollout["pixel"][0]  # (T, C, H, W), [-1, 1]
        gen_voxel = pipe.decode_voxels(rollout["voxel_latents"])[0].cpu().numpy()  # (T, X, Y, Z)
        gen_camera = rollout["camera"][0].cpu()  # (T, 10): [rot6d(6), trans(3), fov(1)]

        real_pixel = batch["raw_images"][0, : gen_pixel.shape[0]]  # (T, C, H, W), [-1, 1]
        real_voxel = batch["voxel_classes"][0, : gen_pixel.shape[0]].numpy()  # (T, X, Y, Z)
        real_camera = batch["camera"][0, : gen_pixel.shape[0]]  # (T, 10)
        n = min(gen_pixel.shape[0], real_pixel.shape[0], gen_voxel.shape[0], real_voxel.shape[0])
        gen_pixel, real_pixel = gen_pixel[:n], real_pixel[:n]
        gen_voxel, real_voxel = gen_voxel[:n], real_voxel[:n]
        gen_camera, real_camera = gen_camera[:n], real_camera[:n]

        input_frame = to_uint8_np(real_pixel[0])

        psnrs, ious, class_accs = [], [], []
        voxel_frames_gen, voxel_frames_real = {}, {}
        pov_frames_gen, pov_frames_real = {}, {}
        frames = []
        for t in range(n):
            gen_f = to_uint8_np(gen_pixel[t])
            real_f = to_uint8_np(real_pixel[t])
            p = psnr(gen_f, real_f)
            iou = voxel_iou(gen_voxel[t], real_voxel[t], empty_class_id)
            cacc = class_accuracy(gen_voxel[t], real_voxel[t])
            psnrs.append(p)
            ious.append(iou)
            class_accs.append(cacc)

            if t % args.voxel_render_stride == 0:
                voxel_frames_gen[t] = render_voxel_3d(
                    gen_voxel[t], empty_class_id, "GEN VOXELS 3D", camera_forward_local(gen_camera[t, :6])
                )
                voxel_frames_real[t] = render_voxel_3d(
                    real_voxel[t], empty_class_id, "GT VOXELS 3D", camera_forward_local(real_camera[t, :6])
                )
                pov_frames_gen[t] = render_voxel_pov(
                    torch.from_numpy(gen_voxel[t]), gen_camera[t], empty_class_id, args.device
                )
                pov_frames_real[t] = render_voxel_pov(
                    torch.from_numpy(real_voxel[t]), real_camera[t], empty_class_id, args.device
                )
                if args.export_glb:
                    glb_dir = os.path.join(args.output_dir, "glb", f"episode_{ep_idx:02d}")
                    os.makedirs(glb_dir, exist_ok=True)
                    voxel_classes_to_glb(gen_voxel[t], empty_class_id, os.path.join(glb_dir, f"gen_t{t:03d}.glb"))
                    voxel_classes_to_glb(real_voxel[t], empty_class_id, os.path.join(glb_dir, f"gt_t{t:03d}.glb"))
            last_rendered = max(k for k in voxel_frames_gen if k <= t)

            h = args.panel_height
            panels = [
                resize_panel(input_frame, h),
                draw_text(resize_panel(gen_f, h), f"gen  PSNR {p:.1f}dB"),
                resize_panel(real_f, h),
                draw_text(resize_panel(voxel_frames_gen[last_rendered], h), f"gen voxels  IoU {iou:.2f} acc {cacc:.2f}"),
                resize_panel(voxel_frames_real[last_rendered], h),
                draw_text(resize_panel(pov_frames_gen[last_rendered], h), "gen voxels POV"),
                resize_panel(pov_frames_real[last_rendered], h),
            ]
            frames.append(np.concatenate(panels, axis=1))

        row_video_path = os.path.join(args.output_dir, f"episode_{ep_idx:02d}.mp4")
        imageio.mimsave(row_video_path, frames, fps=args.fps)
        row_videos.append(frames)
        plot_camera_trajectory(
            {args.method_name: gen_camera, "GT": real_camera},
            os.path.join(args.output_dir, f"episode_{ep_idx:02d}_trajectory.png"),
        )
        summary.append({
            "episode_index": ep_idx,
            "frames": n,
            "mean_psnr_db": float(np.mean(psnrs)),
            "mean_voxel_iou": float(np.mean(ious)),
            "mean_class_accuracy": float(np.mean(class_accs)),
            # Per-frame series, for the TODO 5 within-episode artifact-alignment analysis (does a
            # pixel-quality dip at frame t line up with a voxel-accuracy dip at the same t) --
            # the original 32-episode run only kept the means above, which can't answer that
            # question at all (only whether *episodes* correlate, not *frames* within one).
            "psnr_db": [float(x) for x in psnrs],
            "voxel_iou": [float(x) for x in ious],
            "class_accuracy": [float(x) for x in class_accs],
        })
        logger.info(
            f"Episode {ep_idx}: mean PSNR {np.mean(psnrs):.1f}dB, "
            f"mean voxel IoU {np.mean(ious):.3f}, mean class accuracy {np.mean(class_accs):.3f}"
        )

    max_len = max(len(v) for v in row_videos)
    grid_frames = []
    for i in range(max_len):
        rows = [v[min(i, len(v) - 1)] for v in row_videos]
        widths = [r.shape[1] for r in rows]
        w = max(widths)
        rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) if r.shape[1] < w else r for r in rows]
        grid_frames.append(np.concatenate(rows, axis=0))
    grid_path = os.path.join(args.output_dir, "grid.mp4")
    imageio.mimsave(grid_path, grid_frames, fps=args.fps)

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved grid video to {grid_path} and per-episode videos + summary.json to {args.output_dir}")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
