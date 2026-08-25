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
    gen_occ = gen_classes != empty_id
    real_occ = real_classes != empty_id
    union = np.logical_or(gen_occ, real_occ).sum()
    if union == 0:
        return 1.0
    inter = np.logical_and(gen_occ, real_occ).sum()
    return float(inter / union)


def render_voxel_3d(classes: np.ndarray, empty_id: int, title: str) -> np.ndarray:
    """Render a single (X, Y, Z) class grid as a colored 3D voxel plot."""
    occ = classes != empty_id
    fig = plt.figure(figsize=(3, 3), dpi=80)
    ax = fig.add_subplot(projection="3d")
    if occ.any():
        cmap = plt.get_cmap("tab20")
        norm_classes = (classes % 20) / 20.0
        colors = cmap(norm_classes)
        colors[..., 3] = np.where(occ, 0.9, 0.0)
        ax.voxels(occ, facecolors=colors, edgecolor=None, linewidth=0)
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

        real_pixel = batch["raw_images"][0, : gen_pixel.shape[0]]  # (T, C, H, W), [-1, 1]
        real_voxel = batch["voxel_classes"][0, : gen_pixel.shape[0]].numpy()  # (T, X, Y, Z)
        n = min(gen_pixel.shape[0], real_pixel.shape[0], gen_voxel.shape[0], real_voxel.shape[0])
        gen_pixel, real_pixel = gen_pixel[:n], real_pixel[:n]
        gen_voxel, real_voxel = gen_voxel[:n], real_voxel[:n]

        input_frame = to_uint8_np(real_pixel[0])

        psnrs, ious = [], []
        voxel_frames_gen, voxel_frames_real = {}, {}
        frames = []
        for t in range(n):
            gen_f = to_uint8_np(gen_pixel[t])
            real_f = to_uint8_np(real_pixel[t])
            p = psnr(gen_f, real_f)
            iou = voxel_iou(gen_voxel[t], real_voxel[t], empty_class_id)
            psnrs.append(p)
            ious.append(iou)

            if t % args.voxel_render_stride == 0:
                voxel_frames_gen[t] = render_voxel_3d(gen_voxel[t], empty_class_id, "GEN VOXELS 3D")
                voxel_frames_real[t] = render_voxel_3d(real_voxel[t], empty_class_id, "GT VOXELS 3D")
            last_rendered = max(k for k in voxel_frames_gen if k <= t)

            h = args.panel_height
            panels = [
                resize_panel(input_frame, h),
                draw_text(resize_panel(gen_f, h), f"gen  PSNR {p:.1f}dB"),
                resize_panel(real_f, h),
                draw_text(resize_panel(voxel_frames_gen[last_rendered], h), f"gen voxels  IoU {iou:.2f}"),
                resize_panel(voxel_frames_real[last_rendered], h),
            ]
            frames.append(np.concatenate(panels, axis=1))

        row_video_path = os.path.join(args.output_dir, f"episode_{ep_idx:02d}.mp4")
        imageio.mimsave(row_video_path, frames, fps=args.fps)
        row_videos.append(frames)
        summary.append({
            "episode_index": ep_idx,
            "frames": n,
            "mean_psnr_db": float(np.mean(psnrs)),
            "mean_voxel_iou": float(np.mean(ious)),
        })
        logger.info(f"Episode {ep_idx}: mean PSNR {np.mean(psnrs):.1f}dB, mean voxel IoU {np.mean(ious):.3f}")

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
