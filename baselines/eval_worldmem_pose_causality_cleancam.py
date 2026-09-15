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
from craftium_action_features import CRAFTIUM_ACTION_DIM_ENRICHED, augment_actions_with_camera_features  # noqa: E402

# Column indices of the 2 camera-delta features within the 25-dim enriched action vector --
# see craftium_action_features.py's augment_actions_with_camera_features docstring.
CAMERA_DELTA_COLUMNS = [23, 24]


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


def load_episode_batch_enriched(dataset_root, level_id, num_frames, memory_condition_length, seed):
    import av

    level_dir = os.path.join(dataset_root, level_id)
    npz = np.load(os.path.join(level_dir, "data.npz"))
    current_idx = list(range(num_frames))
    memory_idx = [0] * memory_condition_length
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

    enriched_actions = augment_actions_with_camera_features(npz["action"], npz["player_yaw"], npz["player_pitch"])
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


def run_one_seed(worldmem, video, actions, poses, timestamp, seed, device, ablate):
    """Runs the real/shuffled/zero measurement once, for a given seed and ablation target.
    Returns {"real": psnr, "shuffled": psnr, "zero": psnr}."""
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    n_frames = poses.shape[1]
    perm = torch.randperm(n_frames, generator=rng)

    if ablate == "pose":
        real_actions, real_poses = actions, poses
        shuffled_actions, shuffled_poses = actions, poses[:, perm].clone()
        zero_actions, zero_poses = actions, torch.zeros_like(poses)
    elif ablate == "action":
        # Column-scoped: only vary the 2 camera-delta columns, leave 0-22 and poses untouched.
        real_actions, real_poses = actions, poses
        shuffled_actions = actions.clone()
        shuffled_actions[:, :, CAMERA_DELTA_COLUMNS] = actions[:, perm][:, :, CAMERA_DELTA_COLUMNS]
        shuffled_poses = poses
        zero_actions = actions.clone()
        zero_actions[:, :, CAMERA_DELTA_COLUMNS] = 0.0
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
        "action": vary ONLY the 2 camera-delta action columns (23-24) real/shuffled/zero, pose and
        all other action columns fixed -- targets Hypothesis B (channel competition)."""

    args = tyro.cli(Args)
    device = args.device

    worldmem = build_worldmem_shell_enriched(args.config_path, device, CRAFTIUM_ACTION_DIM_ENRICHED)
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    worldmem.diffusion_model.load_state_dict(ckpt["model"])
    worldmem.diffusion_model.eval()
    print(f"Loaded checkpoint at step={ckpt.get('step')} from {args.ckpt_path}")
    print(f"ablate={args.ablate}, num_seeds={args.num_seeds}")

    from huggingface_hub import hf_hub_download
    cfg = OmegaConf.load(args.config_path)
    vae_ckpt_path = hf_hub_download(
        repo_id="/".join(cfg.vae_path.split("/")[:2]), filename="/".join(cfg.vae_path.split("/")[2:])
    )
    vae_ckpt = torch.load(vae_ckpt_path, map_location="cpu")
    worldmem.vae.load_state_dict(vae_ckpt["state_dict"] if "state_dict" in vae_ckpt else vae_ckpt, strict=True)

    video, actions, poses, timestamp = load_episode_batch_enriched(
        args.dataset_root, args.level_id, args.num_frames, args.memory_condition_length, args.seed
    )
    video, actions, poses, timestamp = (
        video.to(device), actions.to(device), poses.to(device), timestamp.to(device)
    )

    per_seed_results = []
    for s in range(args.num_seeds):
        seed = args.seed + s
        results = run_one_seed(worldmem, video, actions, poses, timestamp, seed, device, args.ablate)
        clean = results["real"] > results["shuffled"] > results["zero"]
        per_seed_results.append(results)
        print(f"[seed {seed}] real={results['real']:.2f}dB shuffled={results['shuffled']:.2f}dB "
              f"zero={results['zero']:.2f}dB  {'CLEAN' if clean else 'not clean'}")

    reals = np.array([r["real"] for r in per_seed_results])
    shuffleds = np.array([r["shuffled"] for r in per_seed_results])
    zeros = np.array([r["zero"] for r in per_seed_results])
    n_clean = sum(r["real"] > r["shuffled"] > r["zero"] for r in per_seed_results)

    print()
    print(f"=== Summary: {args.level_id}, ablate={args.ablate}, {args.num_seeds} seed(s) ===")
    print(f"  real:     mean={reals.mean():.2f}dB  std={reals.std():.2f}dB")
    print(f"  shuffled: mean={shuffleds.mean():.2f}dB  std={shuffleds.std():.2f}dB")
    print(f"  zero:     mean={zeros.mean():.2f}dB  std={zeros.std():.2f}dB")
    print(f"  real-shuffled gap (mean): {reals.mean()-shuffleds.mean():+.2f}dB")
    print(f"  real-zero gap (mean): {reals.mean()-zeros.mean():+.2f}dB")
    print(f"  clean ordering in {n_clean}/{args.num_seeds} seeds")


if __name__ == "__main__":
    main()
