"""Compute FVD between PERSIST-pipeline rollouts and ground-truth episodes.

Reproduces the protocol described in the PERSIST paper's Appendix B.1 (arXiv:2603.03482v2):
rollouts are generated from each held-out episode's initial frame + full action sequence,
spliced into 16-frame clips against the matching ground-truth clips, and scored via I3D
features (see `utils/fvd_metric.py`).

No FVD/eval-metric script exists elsewhere in this repo -- `scripts/run_inference.py` only
saves rollout videos, it never scores them against anything. This script reuses
run_inference.py's checkpoint/dataset resolution and context-building helpers rather than
duplicating them, and is meant to be run once per method (PERSIST now; Oasis/WorldMem once
those baselines exist -- see the "Reproduce PERSIST paper Table 1" plan) with `--method-name`
set accordingly, then the resulting JSONs assembled into a table.

Multi-GPU: episodes are sharded across processes the same way run_inference.py shards them
(`accelerator.prepare(dataloader)`, `even_batches=True`). I3D features (not raw video, which
would be far more data to move) are extracted per-process on its own shard, then gathered onto
every process via `accelerate.utils.gather_object` before the Fréchet distance -- which needs
the full real/generated feature distributions, not a per-shard one -- is computed once on the
main process. Launch with e.g.:
    accelerate launch --multi_gpu --num_processes 2 -m scripts.eval_fvd ...
Known caveat: `even_batches` pads the last batch on some processes when episode count isn't
evenly divisible by `num_processes * batch_size`, duplicating a few clips into the gathered
features. Minor with 256 episodes over 2-4 processes; noted rather than silently ignored.

Other known limitations:
  - The three Table 1 columns (200/400/600 frames) are computed by truncating ONE 600-frame
    rollout per episode, not by regenerating three separate rollouts -- the paper does not
    specify which approach it used, so this is a documented assumption, not a confirmed match.
  - `--pipeline-variant XL` is inferred to correspond to the paper's "PERSIST" row (v2 naming
    dropped the "-XL" suffix used in v1); verify against the HF model card before trusting
    results.

Example:
    uv run python -m scripts.eval_fvd \
        --pipeline-variant XL \
        --checkpoints-namespace PERSIST-team \
        --dataset-repo PERSIST-team/persist-eval-sample \
        --frame-lengths 200 400 600 \
        --method-name PERSIST \
        --output-json outputs/eval_fvd/persist.json
"""
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

# Make the repo root importable whether invoked as `python -m scripts.eval_fvd`
# or `python scripts/eval_fvd.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import tyro
from accelerate import Accelerator
from accelerate.utils import gather_object
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loaders.minetest_latent_camera_action_dataset import MinetestLatentCameraActionEval
from pipelines.pipeline import VoxelFirstPipeline, pipeline_variant_overrides
from scripts.run_inference import Checkpoints, build_context, resolve_checkpoints, resolve_dataset_root
from utils.fvd_metric import (
    I3D_FEATURE_DIM,
    clips_from_video,
    compute_frechet_distance,
    extract_i3d_features,
    load_i3d_model,
)


@dataclass
class Args:
    """FVD evaluation configuration. Mirrors `scripts.run_inference.Args`' checkpoint/dataset
    fields (reused directly by `resolve_checkpoints`/`resolve_dataset_root`), extended with
    scoring-specific options."""

    checkpoints: Checkpoints = field(default_factory=Checkpoints)
    checkpoints_namespace: Optional[str] = None
    checkpoints_repo_prefix: str = "persist-"
    checkpoints_revision: Optional[str] = None

    dataset_path: Optional[str] = None
    dataset_repo: Optional[str] = None
    dataset_revision: Optional[str] = None

    pipeline_variant: Literal["S", "XL"] = "XL"
    pipeline_path: str = "pipelines/persist_S_pipeline"
    compile_transformer_blocks: bool = False

    eval_instances: Optional[str] = None
    """Comma-separated sha256/instance ids to roll out. Defaults to all held-out episodes."""
    num_instances: Optional[int] = None
    """Optional cap on how many held-out episodes to evaluate (default: all, e.g. 256 for
    persist-eval-sample)."""
    batch_size: int = 1

    frame_lengths: List[int] = field(default_factory=lambda: [200, 400, 600])
    """Table 1 columns to report. The longest value sets the rollout length; shorter columns
    truncate that same rollout rather than regenerating (see module docstring)."""
    use_camera_gt: bool = False
    include_initial_voxel_frame: bool = False
    include_initial_pixel_frame: bool = False
    use_kv_cache: bool = True

    i3d_checkpoint: Optional[str] = None
    """Local path to a Kinetics-400 I3D TorchScript checkpoint. Downloaded automatically if unset."""
    i3d_batch_size: int = 8

    mixed_precision: Literal["no", "fp16", "bf16"] = "bf16"
    device: str = "cuda:0"
    """CUDA device for single-process runs. Under `accelerate launch` the device is managed by
    accelerate (one process per GPU) and this is ignored."""

    output_json: str = "outputs/eval_fvd/results.json"
    method_name: str = "PERSIST"
    """Label recorded in the output JSON (e.g. "PERSIST", "Oasis", "WorldMem")."""


def build_dataloader(args: Args, clip_len: int) -> DataLoader:
    """Load held-out episodes with `clip_len` frames of ground truth + action context.

    A local re-implementation of `run_inference.build_dataloader` rather than a direct call,
    since that function hardcodes `clip_len=args.num_frames` (a field this script's `Args`
    doesn't have -- `frame_lengths` plays that role here). `resolve_dataset_root` and the
    dataset/collate classes are still reused as-is.
    """
    root = resolve_dataset_root(args)
    dataset = MinetestLatentCameraActionEval(
        root,
        clip_len=clip_len,
        instances=args.eval_instances,
        num_instances=args.num_instances,
    )
    if len(dataset) == 0:
        raise ValueError(f"No episodes selected from {root} (eval_instances={args.eval_instances}).")
    logger.info(f"Loaded {len(dataset)} held-out episode(s) from {root}.")
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=dataset.collate_fn,
    )


def to_uint8_range(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] float -> [0, 255] float, matching run_inference.py's `to_uint8` convention."""
    return torch.clamp(((x + 1) / 2) * 255, 0, 255)


def main(args: Args):
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    distributed = accelerator.num_processes > 1
    device = accelerator.device if distributed else torch.device(args.device)
    max_frames = max(args.frame_lengths)

    model_config_overrides, sampler_args = pipeline_variant_overrides(args.pipeline_variant)
    if accelerator.is_main_process:
        logger.info(f"Loading persist_S pipeline (variant {args.pipeline_variant}) from: {args.pipeline_path}")
        logger.info(f"Processes: {accelerator.num_processes} | device: {device} | mixed_precision: {args.mixed_precision}")
    pipe = VoxelFirstPipeline.from_pretrained(
        args.pipeline_path,
        device=device,
        accelerator=accelerator,
        custom_checkpoint_cfg=resolve_checkpoints(args),
        model_config_overrides=model_config_overrides,
        voxel_sampler_args=sampler_args or None,
        pixel_sampler_args=sampler_args or None,
    )
    if args.compile_transformer_blocks:
        pipe.compile_blocks()

    dataloader = build_dataloader(args, clip_len=max_frames)
    if distributed:
        # Shards episodes across processes; even_batches (default) pads so every process runs
        # the same number of pipe() calls, keeping the rollout's periodic wait_for_everyone
        # aligned (same rationale as run_inference.py). Caveat: this can duplicate a few clips
        # into the gathered features when len(dataset) isn't evenly divisible by
        # num_processes * batch_size -- a minor skew, not silently assumed away.
        dataloader = accelerator.prepare(dataloader)

    i3d = load_i3d_model(args.i3d_checkpoint, device=str(device))

    # Per-length I3D feature buffers, local to this process. Streaming features per-episode
    # (instead of accumulating raw video tensors for all episodes) keeps peak memory bounded
    # and is what makes the cross-process gather below cheap (400-dim vectors, not video).
    local_real_feats: Dict[int, List[np.ndarray]] = {length: [] for length in args.frame_lengths}
    local_gen_feats: Dict[int, List[np.ndarray]] = {length: [] for length in args.frame_lengths}

    for batch in tqdm(dataloader, desc=f"Rolling out {args.method_name}", disable=not accelerator.is_main_process):
        context = build_context(batch, max_frames, args.use_camera_gt, args.include_initial_voxel_frame)
        with accelerator.autocast():
            rollout = pipe(
                context,
                max_frames,
                output_latents=False,
                output_decoded_pixels=True,
                pixel_use_x0=args.include_initial_pixel_frame,
                use_camera_gt=args.use_camera_gt,
                keep_on_device=False,
                use_kv_cache=args.use_kv_cache,
                verbose=accelerator.is_main_process,
            )
        if distributed:
            accelerator.wait_for_everyone()

        for b in range(batch["raw_images"].shape[0]):
            gen = to_uint8_range(rollout["pixel"][b]).cpu()  # (T, C, H, W)
            real = to_uint8_range(batch["raw_images"][b, :max_frames]).cpu()  # (T, C, H, W)
            if gen.shape[0] != real.shape[0]:
                # `VoxelFirstPipeline.__call__` drops its seed frame to align with the GT
                # sequence (see pipelines/pipeline.py); this guards against any residual
                # off-by-one rather than assuming the alignment always holds exactly.
                n = min(gen.shape[0], real.shape[0])
                logger.warning(
                    f"Rollout/ground-truth length mismatch (generated={gen.shape[0]}, "
                    f"real={real.shape[0]}); truncating both to {n} frames."
                )
                gen, real = gen[:n], real[:n]

            for length in args.frame_lengths:
                if gen.shape[0] < length:
                    continue
                gen_clips = clips_from_video(gen[:length])
                real_clips = clips_from_video(real[:length])
                local_gen_feats[length].append(
                    extract_i3d_features(gen_clips, i3d, device=str(device), batch_size=args.i3d_batch_size)
                )
                local_real_feats[length].append(
                    extract_i3d_features(real_clips, i3d, device=str(device), batch_size=args.i3d_batch_size)
                )

        if distributed:
            accelerator.wait_for_everyone()

    results = {}
    for length in sorted(args.frame_lengths):
        local_g = (
            np.concatenate(local_gen_feats[length], axis=0)
            if local_gen_feats[length]
            else np.zeros((0, I3D_FEATURE_DIM), dtype=np.float32)
        )
        local_r = (
            np.concatenate(local_real_feats[length], axis=0)
            if local_real_feats[length]
            else np.zeros((0, I3D_FEATURE_DIM), dtype=np.float32)
        )
        if distributed:
            # gather_object moves arbitrary picklable python objects (unlike accelerator.gather,
            # which requires matching tensor shapes across processes) -- exactly what's needed
            # since each process may have accumulated a different number of clips.
            gathered_g = gather_object([local_g])
            gathered_r = gather_object([local_r])
        else:
            gathered_g, gathered_r = [local_g], [local_r]

        if accelerator.is_main_process:
            all_gen = np.concatenate(gathered_g, axis=0)
            all_real = np.concatenate(gathered_r, axis=0)
            fvd = compute_frechet_distance(all_real, all_gen)
            logger.info(
                f"{args.method_name} FVD @ {length} frames: {fvd:.1f} "
                f"({all_real.shape[0]} real / {all_gen.shape[0]} generated clips)"
            )
            results[str(length)] = fvd

    if accelerator.is_main_process:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump({"method": args.method_name, "fvd": results}, f, indent=2)
        logger.info(f"Saved results to {args.output_json}")


if __name__ == "__main__":
    main(tyro.cli(Args))
