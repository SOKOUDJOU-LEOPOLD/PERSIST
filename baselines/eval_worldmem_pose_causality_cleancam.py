"""Same real/shuffled/zero causality ablation as before, adapted for the 25-dim clean-camera-feature
checkpoint (23 raw Craftium actions + yaw_delta + pitch_delta), now extended with two diagnostic
capabilities added to disambiguate why the original single-seed, single-checkpoint sweep (9
checkpoints, steps 0-20000) produced erratic, non-reproducible ordering (clean at only 1 of 9):

--num-seeds N: repeats the real/shuffled/zero measurement N times per checkpoint (re-seeding both
    the diffusion sampler and the shuffle permutation each time, exactly as the original single-seed
    path already did), loading the checkpoint/VAE only once. Targets Hypothesis A (the ablation is a
    single-episode, single-rollout metric -- is the erratic 9-point trajectory just measurement
    noise, or a real, reproducible property of each checkpoint?).

--ablate {pose,action}: with "action" (default remains "pose", unchanged behavior), holds `poses`
    real/fixed throughout and instead varies ONLY columns 23-24 (yaw_delta, pitch_delta) of the
    action tensor across real/shuffled/zero, leaving columns 0-22 untouched in all three variants --
    column-scoped, not a full-vector shuffle, so movement-causality isn't confounded in. Targets
    Hypothesis B (pose_conditions and the action-embedded camera-delta columns are two independent,
    architecturally-separate pathways into the diffusion model carrying the same real
    camera-orientation information -- has the model shifted causal reliance onto the action channel
    instead of pose_conditions?).

CAVEAT worth flagging (unchanged from before): since the 2 extra action columns are themselves
derived from real player_yaw/pitch, shuffling/zeroing pose_conditions no longer removes ALL
pose-correlated signal from the model's inputs -- the action channel now also carries a truthful
camera-delta signal. This is exactly what --ablate action is designed to probe directly, rather
than just caveat around.

--action-layout {bespoke,native}: selects WHICH of the two action reconstructions this checkpoint
was trained with, and therefore which columns actually carry camera information for --ablate
action to touch. This matters a lot: the two layouts put camera info in DIFFERENT columns of the
same 25-wide vector, and getting it wrong makes --ablate action a silent no-op rather than an
error.
    "bespoke" (the original craftium_action_features.augment_actions_with_camera_features):
        raw 23-dim Craftium action + 2 appended columns [23]=yaw_delta, [24]=pitch_delta
        (real degree magnitudes). Camera-ablation columns: [23, 24].
    "native" (augment_actions_to_worldmem_native_layout): Craftium's action reconstructed
        directly into WorldMem's own ACTION_KEYS column layout -- camera lives at [15]=cameraY,
        [16]=cameraX (sign-only, {-1,0,1}), and columns [23]/[24] (pickItem/drop) are provably
        always exactly 0 in this layout. Camera-ablation columns: [15, 16].
Running --ablate action with the wrong --action-layout against a "native" checkpoint would shuffle
columns [23,24], which are always-zero dead columns in that layout -- a no-op that would misreport
"the action channel has no causal effect" for a reason that has nothing to do with the model.
"""
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/lsokoudj/PERSIST/baselines")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "WorldMem"))

from finetune_worldmem import CRAFTIUM_ACTION_DIM, GatedActionEmbed  # noqa: E402
from finetune_worldmem_data import DATASET_ROOT, normalize_pose, pitch_yaw_from_cam_dir  # noqa: E402
from craftium_action_features import (  # noqa: E402
    CRAFTIUM_ACTION_DIM_ENRICHED,
    CRAFTIUM_ACTION_DIM_WORLDMEM_NATIVE,
    augment_actions_with_camera_features,
    augment_actions_to_worldmem_native_layout,
)

# Column indices carrying camera information, per action layout -- see module docstring's
# --action-layout section for the full derivation of why these differ.
CAMERA_COLUMNS_BY_LAYOUT = {
    "bespoke": [23, 24],   # yaw_delta, pitch_delta (raw degree magnitudes)
    "native": [15, 16],    # cameraY, cameraX (sign-only, {-1,0,1})
}
ACTION_DIM_BY_LAYOUT = {
    "bespoke": CRAFTIUM_ACTION_DIM_ENRICHED,
    "native": CRAFTIUM_ACTION_DIM_WORLDMEM_NATIVE,
}
CONVERTER_BY_LAYOUT = {
    "bespoke": augment_actions_with_camera_features,
    "native": augment_actions_to_worldmem_native_layout,
}


def build_worldmem_shell_enriched(config_path: str, device: str, action_dim: int):
    from algorithms.worldmem import WorldMemMinecraft

    cfg = OmegaConf.load(config_path)
    cfg.action_cond_dim = action_dim
    cfg.require_pose_prediction = False
    cfg.chunk_size = cfg.n_tokens
    worldmem = WorldMemMinecraft(cfg)
    action_layer = worldmem.diffusion_model.model.external_cond
    gated = GatedActionEmbed(action_dim, action_layer.out_features)
    worldmem.diffusion_model.model.external_cond = gated
    worldmem.vae.eval()
    for p in worldmem.vae.parameters():
        p.requires_grad_(False)
    return worldmem.to(device)


def load_episode_batch_enriched(dataset_root, level_id, num_frames, memory_condition_length, seed, action_layout,
                                 start_frame=0):
    """start_frame: shifts the whole window to begin at episode frame `start_frame` instead of 0
    (memory frames also shift to all point at `start_frame`, the same "no real past yet" degenerate
    case the start=0 default already uses -- just relocated). Lets a diagnostic put a specific
    episode segment (e.g. a segment that stalled deep in a long rollout) only a few autoregressive
    steps into a FRESH rollout, to separate "this episode position is inherently hard" from "this
    many autoregressive steps of accumulated drift is inherently hard" -- see
    generate_native_grid.py's --start-frame."""
    import av

    level_dir = os.path.join(dataset_root, level_id)
    npz = np.load(os.path.join(level_dir, "data.npz"))
    current_idx = list(range(start_frame, start_frame + num_frames))
    memory_idx = [start_frame] * memory_condition_length
    all_idx = current_idx + memory_idx

    wanted = set(all_idx)
    frames_by_idx = {}
    container = av.open(os.path.join(level_dir, "rgb.mp4"))
    for i, frame in enumerate(container.decode(container.streams.video[0])):
        if i in wanted:
            frames_by_idx[i] = frame.to_ndarray(format="rgb24")
        if i >= max(wanted):
            break
    container.close()
    frames = np.stack([frames_by_idx[i] for i in all_idx])
    video = torch.from_numpy(frames).float() / 255.0
    video = video.permute(0, 3, 1, 2).contiguous()

    converter = CONVERTER_BY_LAYOUT[action_layout]
    enriched_actions = converter(npz["action"], npz["player_yaw"], npz["player_pitch"])
    actions = torch.from_numpy(enriched_actions[all_idx].astype(np.float32))

    pos = npz["player_pos"][all_idx].astype(np.float64)
    pitch, yaw = pitch_yaw_from_cam_dir(npz["cam_dir"][all_idx].astype(np.float64))
    poses_raw = np.concatenate([pos, pitch[:, None], yaw[:, None]], axis=1)
    poses = normalize_pose(poses_raw, poses_raw[:1]).astype(np.float32)
    poses = torch.from_numpy(poses)
    timestamp = torch.tensor(all_idx, dtype=torch.long)

    return video[None], actions[None], poses[None], timestamp[None]


def psnr(gen, real):
    mse = np.mean((gen.astype(np.float32) - real.astype(np.float32)) ** 2)
    if mse < 1e-10:
        return 99.0
    return float(10 * np.log10(255.0 ** 2 / mse))


def to_np(x):
    x = torch.clamp(x[:, 0], 0, 1) * 255.0
    return x.permute(0, 2, 3, 1).byte().cpu().numpy()


def run_one_seed(worldmem, video, actions, poses, timestamp, seed, device, ablate, camera_columns):
    """Runs the real/shuffled/zero measurement once, for a given seed and ablation target.
    Returns {"real": psnr, "shuffled": psnr, "zero": psnr}.

    camera_columns: which action columns actually carry camera info for --ablate action -- see
    CAMERA_COLUMNS_BY_LAYOUT / module docstring. Must match the checkpoint's own action layout or
    this ablates dead, always-zero columns instead of the real camera signal."""
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    n_frames = poses.shape[1]
    perm = torch.randperm(n_frames, generator=rng)

    if ablate == "pose":
        real_actions, real_poses = actions, poses
        shuffled_actions, shuffled_poses = actions, poses[:, perm].clone()
        zero_actions, zero_poses = actions, torch.zeros_like(poses)
    elif ablate == "action":
        # Column-scoped: only vary the camera columns for this layout, leave everything else
        # (movement/hotbar columns and poses) untouched.
        real_actions, real_poses = actions, poses
        shuffled_actions = actions.clone()
        shuffled_actions[:, :, camera_columns] = actions[:, perm][:, :, camera_columns]
        shuffled_poses = poses
        zero_actions = actions.clone()
        zero_actions[:, :, camera_columns] = 0.0
        zero_poses = poses
    else:
        raise ValueError(f"unknown ablate target: {ablate}")

    results = {}
    for name, (a, p) in [("real", (real_actions, real_poses)), ("shuffled", (shuffled_actions, shuffled_poses)),
                          ("zero", (zero_actions, zero_poses))]:
        batch = (video, a, p, timestamp)
        worldmem.validation_step_outputs = []
        with torch.no_grad():
            worldmem.validation_step(batch, 0)
        xs_pred, xs_decode = worldmem.validation_step_outputs[-1]
        gen_np, real_np = to_np(xs_pred), to_np(xs_decode)
        n = min(gen_np.shape[0], real_np.shape[0])
        per_frame = [psnr(gen_np[t], real_np[t]) for t in range(n)]
        results[name] = float(np.mean(per_frame))
    return results


def main():
    import tyro
    from dataclasses import dataclass
    from typing import Literal

    @dataclass
    class Args:
        ckpt_path: str = "../../outputs/overfit_worldmem_level_006_cleancam/final.pt"
        config_path: str = "configurations/huggingface.yaml"
        level_id: str = "level_006"
        dataset_root: str = DATASET_ROOT
        num_frames: int = 150
        memory_condition_length: int = 8
        device: str = "cuda:0"
        seed: int = 0
        num_seeds: int = 1
        """Repeats the measurement this many times (seeds = args.seed .. args.seed+num_seeds-1),
        loading the checkpoint/VAE only once. Reports mean+-std per variant, targeting Hypothesis A
        (single-rollout measurement noise)."""
        ablate: Literal["pose", "action"] = "pose"
        """"pose" (default, unchanged): vary the pose tensor real/shuffled/zero, actions fixed.
        "action": vary ONLY this checkpoint's camera columns (see --action-layout) real/shuffled/
        zero, pose and all other action columns fixed -- targets Hypothesis B (channel
        competition)."""
        action_layout: Literal["bespoke", "native"] = "bespoke"
        """Which action reconstruction THIS CHECKPOINT was fine-tuned with -- selects both how raw
        Craftium actions are converted for this eval AND which columns --ablate action touches.
        "bespoke": the original 23-raw+2-appended (yaw_delta,pitch_delta) layout, camera at
        columns [23,24]. "native": Craftium reconstructed into WorldMem's own ACTION_KEYS layout,
        camera (sign-only cameraY/cameraX) at columns [15,16]. Must match how the checkpoint was
        trained (use_clean_camera_features vs use_worldmem_native_action_layout in
        finetune_worldmem.py) -- using the wrong one silently evaluates the wrong columns."""

    args = tyro.cli(Args)
    device = args.device
    action_dim = ACTION_DIM_BY_LAYOUT[args.action_layout]
    camera_columns = CAMERA_COLUMNS_BY_LAYOUT[args.action_layout]

    worldmem = build_worldmem_shell_enriched(args.config_path, device, action_dim)
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    worldmem.diffusion_model.load_state_dict(ckpt["model"])
    worldmem.diffusion_model.eval()
    print(f"Loaded checkpoint at step={ckpt.get('step')} from {args.ckpt_path}")
    print(f"ablate={args.ablate}, num_seeds={args.num_seeds}, action_layout={args.action_layout}, "
          f"camera_columns={camera_columns}")

    from huggingface_hub import hf_hub_download
    cfg = OmegaConf.load(args.config_path)
    vae_ckpt_path = hf_hub_download(
        repo_id="/".join(cfg.vae_path.split("/")[:2]), filename="/".join(cfg.vae_path.split("/")[2:])
    )
    vae_ckpt = torch.load(vae_ckpt_path, map_location="cpu")
    worldmem.vae.load_state_dict(vae_ckpt["state_dict"] if "state_dict" in vae_ckpt else vae_ckpt, strict=True)

    video, actions, poses, timestamp = load_episode_batch_enriched(
        args.dataset_root, args.level_id, args.num_frames, args.memory_condition_length, args.seed,
        args.action_layout,
    )
    video, actions, poses, timestamp = (
        video.to(device), actions.to(device), poses.to(device), timestamp.to(device)
    )

    per_seed_results = []
    for s in range(args.num_seeds):
        seed = args.seed + s
        results = run_one_seed(worldmem, video, actions, poses, timestamp, seed, device, args.ablate, camera_columns)
        clean = results["real"] > results["shuffled"] > results["zero"]
        per_seed_results.append(results)
        print(f"[seed {seed}] real={results['real']:.2f}dB shuffled={results['shuffled']:.2f}dB "
              f"zero={results['zero']:.2f}dB  {'CLEAN' if clean else 'not clean'}")

    reals = np.array([r["real"] for r in per_seed_results])
    shuffleds = np.array([r["shuffled"] for r in per_seed_results])
    zeros = np.array([r["zero"] for r in per_seed_results])
    n_clean = sum(r["real"] > r["shuffled"] > r["zero"] for r in per_seed_results)

    print()
    print(f"=== Summary: {args.level_id}, ablate={args.ablate}, action_layout={args.action_layout}, "
          f"{args.num_seeds} seed(s) ===")
    print(f"  real:     mean={reals.mean():.2f}dB  std={reals.std():.2f}dB")
    print(f"  shuffled: mean={shuffleds.mean():.2f}dB  std={shuffleds.std():.2f}dB")
    print(f"  zero:     mean={zeros.mean():.2f}dB  std={zeros.std():.2f}dB")
    print(f"  real-shuffled gap (mean): {reals.mean()-shuffleds.mean():+.2f}dB")
    print(f"  real-zero gap (mean): {reals.mean()-zeros.mean():+.2f}dB")
    print(f"  clean ordering in {n_clean}/{args.num_seeds} seeds")


if __name__ == "__main__":
    main()
