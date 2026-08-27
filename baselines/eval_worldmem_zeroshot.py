"""Zero-shot WorldMem baseline: score against persist-eval-sample ground truth.

Same "same seed frame + own action sampling" approach as eval_oasis_zeroshot.py: each held-
out persist-eval-sample episode's first frame seeds WorldMem's own interactive() generation,
driven by a fixed, gentle, WorldMem-native key sequence (its own 25-dim VPT-style action
space, not derived from Craftium's 23-dim one), scored via the same utils/fvd_metric.py
against the same real ground truth used for the PERSIST row.

Runs in WorldMem's OWN isolated venv (baselines/WorldMem/.venv) -- unlike Oasis, its
requirements.txt pins torch~=2.4.0, incompatible with PERSIST's torch==2.7.1 venv.

The exact call sequence below (init call with all-zero action/pose, then a generation call
with memory carried forward) was reverse-engineered from and validated against WorldMem's own
app.py::generate() Gradio handler -- see baselines/worldmem_validate_example.py, which also
found that a hard, sustained camera-turn action sequence (8 consecutive turns) produces
visibly degraded/banded output. That was determined to be a genuine out-of-distribution-action
failure mode, not a bug in this adaptation (a gentle action sequence on the same seed frame
produced a coherent rollout). ACTION_PATTERN below is chosen to stay well clear of that
failure mode: mostly forward motion with single, isolated turns rather than sustained ones.

Example:
    cd baselines/WorldMem
    python ../eval_worldmem_zeroshot.py \
        --persist-eval-root <persist-eval-sample snapshot dir> \
        --frame-lengths 200 400 600 \
        --output-json ../../outputs/eval_fvd/worldmem.json
"""
import importlib.util
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import av
import imageio
import numpy as np
import torch
import torchvision.transforms as transforms
import tyro
from loguru import logger
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

WORLDMEM_ROOT = Path(__file__).resolve().parent / "WorldMem"
sys.path.insert(0, str(WORLDMEM_ROOT))
from algorithms.worldmem import WorldMemMinecraft  # noqa: E402
from experiments.exp_base import load_custom_checkpoint  # noqa: E402

# PERSIST's utils/fvd_metric.py, loaded by file path -- same rationale as
# eval_oasis_zeroshot.py: WorldMem's own repo has several of its own top-level modules that
# could collide with a package-style `from utils.fvd_metric import ...` depending on import
# order and sys.path state, so this sidesteps that class of bug entirely rather than relying
# on import order being right.
PERSIST_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_fvd_metric_spec = importlib.util.spec_from_file_location(
    "persist_fvd_metric", os.path.join(PERSIST_ROOT, "utils", "fvd_metric.py")
)
_fvd_metric = importlib.util.module_from_spec(_fvd_metric_spec)
_fvd_metric_spec.loader.exec_module(_fvd_metric)
I3D_FEATURE_DIM = _fvd_metric.I3D_FEATURE_DIM
clips_from_video = _fvd_metric.clips_from_video
compute_frechet_distance = _fvd_metric.compute_frechet_distance
extract_i3d_features = _fvd_metric.extract_i3d_features
load_i3d_model = _fvd_metric.load_i3d_model

ACTION_KEYS = [
    "inventory", "ESC", "hotbar.1", "hotbar.2", "hotbar.3", "hotbar.4", "hotbar.5",
    "hotbar.6", "hotbar.7", "hotbar.8", "hotbar.9", "forward", "back", "left", "right",
    "cameraY", "cameraX", "jump", "sneak", "sprint", "swapHands", "attack", "use",
    "pickItem", "drop",
]
KEY_TO_ACTION = {
    "Q": ("forward", 1), "E": ("back", 1), "W": ("cameraY", -1), "S": ("cameraY", 1),
    "A": ("cameraX", -1), "D": ("cameraX", 1), "U": ("drop", 1), "N": ("noop", 1), "1": ("hotbar.1", 1),
}
# Mostly forward, with single isolated turns (never consecutive) -- see module docstring for
# why sustained turns are avoided. Repeated/truncated to whatever length is needed.
ACTION_PATTERN = "QQQQQQQQQDQQQQQQQQQAQQQQQQQQQ"


def build_action_string(length: int) -> str:
    reps = length // len(ACTION_PATTERN) + 1
    return (ACTION_PATTERN * reps)[:length]


def parse_input_to_tensor(input_str: str) -> torch.Tensor:
    action_tensor = torch.zeros((len(input_str), 25))
    for i, char in enumerate(input_str):
        action, value = KEY_TO_ACTION[char.upper()]
        if action in ACTION_KEYS:
            action_tensor[i, ACTION_KEYS.index(action)] = value
    return action_tensor


def load_image_as_tensor(image_path: str) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    return transforms.Compose([transforms.ToTensor()])(image).numpy()


def extract_frame0_png(video_path: str, out_path: str) -> str:
    container = av.open(video_path)
    frame = next(container.decode(container.streams.video[0]))
    frame.to_image().save(out_path)
    container.close()
    return out_path


@dataclass
class Args:
    persist_eval_root: str
    config_path: str = "configurations/huggingface.yaml"
    num_instances: Optional[int] = None
    frame_lengths: List[int] = field(default_factory=lambda: [200, 400, 600])
    # WorldMem generates in chunks of cfg.next_frame_length (1 by default) internally, but
    # accepts an arbitrarily long action tensor per interactive() call (it loops over it in
    # next_frame_length-sized chunks itself) -- chunk_frames caps how many frames we request
    # per Python-level call, purely to bound peak memory/allow logging progress, not because
    # interactive() requires it.
    chunk_frames: int = 50

    i3d_checkpoint: Optional[str] = None
    i3d_batch_size: int = 8

    output_json: str = "outputs/eval_fvd/worldmem_zeroshot.json"
    method_name: str = "WorldMem"

    save_video_dir: Optional[str] = None
    """If set, also save each episode's generated video + predicted camera pose trajectory
    (.npy, (T,5) x/y/z/pitch/yaw from WorldMem's own pose_prediction_model) here -- for the
    TODO 4 multi-model comparison grid."""


def load_worldmem(config_path: str, device: torch.device) -> WorldMemMinecraft:
    cfg = OmegaConf.load(config_path)
    worldmem = WorldMemMinecraft(cfg)
    load_custom_checkpoint(algo=worldmem.diffusion_model, checkpoint_path=cfg.diffusion_path)
    load_custom_checkpoint(algo=worldmem.vae, checkpoint_path=cfg.vae_path)
    load_custom_checkpoint(algo=worldmem.pose_prediction_model, checkpoint_path=cfg.pose_predictor_path)
    return worldmem.to(device).eval()


@torch.no_grad()
def worldmem_rollout(worldmem: WorldMemMinecraft, seed_frame: np.ndarray, total_frames: int, device, chunk_frames: int):
    """Generate `total_frames` frames from a single image, following the exact init ->
    generate call sequence validated in worldmem_validate_example.py.

    Returns (video, poses): video is (T, C, H, W) [0, 255] float; poses is the accumulated
    (T, 5) [x, y, z, pitch, yaw] trajectory from WorldMem's own pose_prediction_model (see
    df_video.py:833-838) -- previously computed and discarded like everything else this script
    used to throw away after FVD feature extraction, now exposed for the TODO 4 camera
    trajectory plot (WorldMem is one of the few methods here that actually has a pose output;
    Oasis has none at all)."""
    init_action = np.zeros(25, dtype=np.float32)
    init_pose = np.zeros(5, dtype=np.float32)

    _, mem_latents, mem_actions, mem_poses, mem_c2w, mem_idx = worldmem.interactive(
        seed_frame, init_action, init_pose, device=device,
        memory_latent_frames=None, memory_actions=None, memory_poses=None,
        memory_c2w=None, memory_frame_idx=None,
    )

    video_frames = seed_frame[None]  # (1, C, H, W), float [0, 1]
    remaining = total_frames - 1
    action_str = build_action_string(remaining)
    offset = 0
    while offset < remaining:
        n = min(chunk_frames, remaining - offset)
        input_actions = parse_input_to_tensor(action_str[offset : offset + n])
        new_frame, mem_latents, mem_actions, mem_poses, mem_c2w, mem_idx = worldmem.interactive(
            seed_frame, input_actions, None, device=device,
            memory_latent_frames=mem_latents, memory_actions=mem_actions, memory_poses=mem_poses,
            memory_c2w=mem_c2w, memory_frame_idx=mem_idx,
        )
        video_frames = np.concatenate([video_frames, new_frame[:, 0]])
        offset += n

    out = np.clip(video_frames, 0.0, 1.0) * 255.0
    return torch.from_numpy(out).float(), mem_poses  # (T, C, H, W); pose trajectory, shape TBD -- see call site


def main(args: Args):
    device = torch.device("cuda")
    worldmem = load_worldmem(args.config_path, device)
    logger.info("WorldMem model + checkpoints loaded.")

    root = Path(args.persist_eval_root)
    levels = sorted(p.parent.name for p in root.glob("level_*/rgb.mp4"))
    if args.num_instances:
        levels = levels[: args.num_instances]
    if not levels:
        raise ValueError(f"No level_*/rgb.mp4 found under {root}")
    logger.info(f"Found {len(levels)} persist-eval-sample episodes under {root}")

    i3d = load_i3d_model(args.i3d_checkpoint, device=str(device))

    max_frames = max(args.frame_lengths)
    local_real_feats = {length: [] for length in args.frame_lengths}
    local_gen_feats = {length: [] for length in args.frame_lengths}

    from torchvision.io import read_video
    from einops import rearrange

    for level in tqdm(levels, desc="WorldMem zero-shot rollouts"):
        rgb_path = str(root / level / "rgb.mp4")
        with tempfile.TemporaryDirectory() as tmp_dir:
            prompt_png = extract_frame0_png(rgb_path, os.path.join(tmp_dir, "frame0.png"))
            seed_frame = load_image_as_tensor(prompt_png)
            gen, poses = worldmem_rollout(worldmem, seed_frame, max_frames, device, args.chunk_frames)  # (T, C, H, W); pose traj

        real, _, _ = read_video(rgb_path, pts_unit="sec")  # (T, H, W, C) uint8
        real = rearrange(real[:max_frames].float(), "t h w c -> t c h w")
        if real.shape[-2:] != gen.shape[-2:]:
            real = torch.nn.functional.interpolate(real, size=gen.shape[-2:], mode="bilinear", align_corners=False)

        n = min(gen.shape[0], real.shape[0])
        if n < max_frames:
            logger.warning(f"{level}: only {n} frames available (wanted {max_frames}); truncating.")
        gen, real = gen[:n], real[:n]

        if args.save_video_dir:
            os.makedirs(args.save_video_dir, exist_ok=True)
            gen_np = rearrange(gen.clamp(0, 255).byte(), "t c h w -> t h w c").cpu().numpy()
            imageio.mimsave(os.path.join(args.save_video_dir, f"{level}_gen.mp4"), gen_np, fps=24)
            poses_np = poses.cpu().numpy() if isinstance(poses, torch.Tensor) else np.asarray(poses)
            pose_np = poses_np.reshape(poses_np.shape[0], -1)[:, :5]
            np.save(os.path.join(args.save_video_dir, f"{level}_pose.npy"), pose_np)

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

    results = {}
    for length in sorted(args.frame_lengths):
        gen_feats = (
            np.concatenate(local_gen_feats[length], axis=0)
            if local_gen_feats[length]
            else np.zeros((0, I3D_FEATURE_DIM), dtype=np.float32)
        )
        real_feats = (
            np.concatenate(local_real_feats[length], axis=0)
            if local_real_feats[length]
            else np.zeros((0, I3D_FEATURE_DIM), dtype=np.float32)
        )
        fvd = compute_frechet_distance(real_feats, gen_feats)
        logger.info(
            f"{args.method_name} FVD @ {length} frames: {fvd:.1f} "
            f"({real_feats.shape[0]} real / {gen_feats.shape[0]} generated clips)"
        )
        results[str(length)] = fvd

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({"method": args.method_name, "fvd": results}, f, indent=2)
    logger.info(f"Saved results to {args.output_json}")


if __name__ == "__main__":
    main(tyro.cli(Args))
