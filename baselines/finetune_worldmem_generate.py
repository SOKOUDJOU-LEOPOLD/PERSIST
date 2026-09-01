"""Qualitative rollout for a fine-tuned (23-dim action) WorldMem checkpoint, driven by a real
Craftium test episode's real seed frame + real actions + real ground-truth poses throughout.

Uses `WorldMemMinecraft.validation_step()` (df_video.py:687-780) rather than the consumer-facing
`interactive()` API (baselines/eval_worldmem_zeroshot.py's path): `interactive()` needs
`pose_prediction_model` to autoregressively predict future camera poses from actions alone, but
that model has its own separate 25-dim action-input mismatch we deliberately left unfixed
(finetune_worldmem.py sets `require_pose_prediction=False` -- out of scope for this fine-tuning
pass). `validation_step()` instead consumes the exact same `(xs, conditions, pose_conditions,
frame_index)` batch format `CraftiumWorldMemDataset` already produces, using real ground-truth
poses/actions the whole way through -- exactly matching what the model was fine-tuned on, and
returning (predicted, ground-truth) decoded frames directly for comparison.

`build_worldmem_shell` overrides the config's `chunk_size=1` to `chunk_size=n_tokens` -- a real
bug workaround (see that function's own comment for the full diagnostic chain), not a style
choice. Without it, `validation_step` produces severe, structured pixel corruption starting
exactly at frame `memory_condition_length`, on both untrained and fine-tuned checkpoints alike --
proven independent of fine-tuning, memory-selection method, and pose data (all three were
separately verified and/or fixed first). The actual cause: `_prepare_conditions`'s sliding window
overlaps chunk to chunk whenever chunk_size < n_tokens, so already-generated frames still in the
window get their relative pose/Plucker conditioning silently recomputed each new step against a
freshly different memory-frame selection -- inconsistent with the content they already show.
chunk_size is read only inside validation_step (never training_step or interactive()), so this
needs no retraining.

Example:
    cd baselines/WorldMem
    .venv/bin/python ../finetune_worldmem_generate.py \
        --ckpt-path ../../outputs/finetune_worldmem/final.pt \
        --level-id level_002 --num-frames 150 \
        --output-path ../../outputs/finetune_worldmem/rollout_level_002_after.mp4
"""
import os
import sys

import imageio
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "WorldMem"))

from finetune_worldmem import CRAFTIUM_ACTION_DIM, GatedActionEmbed  # noqa: E402
from finetune_worldmem_data import DATASET_ROOT, normalize_pose, pitch_yaw_from_cam_dir  # noqa: E402


def build_worldmem_shell(config_path: str, device: str, gated_action: bool = True):
    from algorithms.worldmem import WorldMemMinecraft

    cfg = OmegaConf.load(config_path)
    cfg.action_cond_dim = CRAFTIUM_ACTION_DIM
    cfg.require_pose_prediction = False
    # Override the config's chunk_size=1 with chunk_size=n_tokens (=3). This is a real bug
    # workaround, not a stylistic choice -- found via a long diagnostic chain (see git history):
    # validation_step's sliding window re-invokes _prepare_conditions every new chunk, and that
    # function recomputes each in-window frame's relative Plucker/pose conditioning from scratch
    # using whatever memory-frame set _generate_condition_indices just picked for the CURRENT
    # step -- including frames from PRIOR chunks that are still in the window (whenever
    # chunk_size < n_tokens, the window overlaps chunk to chunk) and whose actual pixel content
    # was generated under a *different* memory selection. Those already-generated frames are held
    # at noise level 0 by _prepare_noise_levels (so their own content doesn't change), but the
    # model still attends to them using this newly-inconsistent conditioning as context for the
    # new frame -- confirmed as the cause by direct A/B: decoding the sampled (not yet decoded)
    # latent for the first affected frame reproduces the corruption in isolation (ruling out the
    # VAE/decode step and confirming it's baked in by sampling), and setting chunk_size=n_tokens
    # (horizon then always equals n_tokens, so start_frame == curr_frame every step -- zero
    # overlap, so no already-generated frame's conditioning is ever recomputed) eliminates the
    # corruption completely, verified on both an untrained and our actual fine-tuned checkpoint.
    # chunk_size is read ONLY inside validation_step -- training_step and interactive() never use
    # it -- so this is purely a generation-time setting; no retraining is required for this fix.
    cfg.chunk_size = cfg.n_tokens
    worldmem = WorldMemMinecraft(cfg)
    if gated_action:
        action_layer = worldmem.diffusion_model.model.external_cond
        gated = GatedActionEmbed(CRAFTIUM_ACTION_DIM, action_layer.out_features)
        worldmem.diffusion_model.model.external_cond = gated
    worldmem.vae.eval()
    for p in worldmem.vae.parameters():
        p.requires_grad_(False)
    return worldmem.to(device)


def load_single_episode_batch(dataset_root: str, level_id: str, num_frames: int, memory_condition_length: int, seed: int):
    """One (video, actions, poses, timestamp) tuple, batch-size-1, for a single test episode --
    same construction as CraftiumWorldMemDataset.__getitem__, factored out here since we only
    need exactly one fixed window (starting at frame 0, matching Oasis's qualitative-check
    convention), not the full windowed-dataset machinery."""
    import random

    import av

    level_dir = os.path.join(dataset_root, level_id)
    npz = np.load(os.path.join(level_dir, "data.npz"))
    current_idx = list(range(num_frames))
    rng = random.Random(seed)
    memory_idx = [0] * memory_condition_length  # start-of-episode window -> no real past frames yet
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

    actions = torch.from_numpy(npz["action"][all_idx].astype(np.float32))

    # Same treatment as CraftiumWorldMemDataset: pitch/yaw derived from cam_dir (player_pitch/
    # player_yaw are stale, not usable -- see finetune_worldmem_data.py's module docstring),
    # position made relative to this window's first frame + wrapped. Memory frames here are all
    # frame 0 itself (start=0, the same degenerate case the original code's own memory selection
    # falls back to for the very first window of an episode), so they normalize to the same
    # all-zero pose as the window's own first frame, which is correct and expected.
    pos = npz["player_pos"][all_idx].astype(np.float64)
    pitch, yaw = pitch_yaw_from_cam_dir(npz["cam_dir"][all_idx].astype(np.float64))
    poses_raw = np.concatenate([pos, pitch[:, None], yaw[:, None]], axis=1)
    poses = normalize_pose(poses_raw, poses_raw[:1]).astype(np.float32)
    poses = torch.from_numpy(poses)
    timestamp = torch.tensor(all_idx, dtype=torch.long)

    return video[None], actions[None], poses[None], timestamp[None]  # add batch dim


def main():
    import tyro
    from dataclasses import dataclass

    @dataclass
    class Args:
        ckpt_path: str = "../../outputs/finetune_worldmem/final.pt"
        config_path: str = "configurations/huggingface.yaml"
        level_id: str = "level_002"
        dataset_root: str = DATASET_ROOT
        num_frames: int = 150
        memory_condition_length: int = 8
        fps: int = 20
        output_path: str = "rollout.mp4"
        gt_output_path: str = ""
        device: str = "cuda:0"
        seed: int = 0

    args = tyro.cli(Args)
    device = args.device
    torch.manual_seed(args.seed)

    worldmem = build_worldmem_shell(args.config_path, device)
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    worldmem.diffusion_model.load_state_dict(ckpt["model"])
    worldmem.diffusion_model.eval()
    print(f"Loaded checkpoint at step={ckpt.get('step')} from {args.ckpt_path}")

    from huggingface_hub import hf_hub_download
    cfg = OmegaConf.load(args.config_path)
    vae_ckpt_path = hf_hub_download(
        repo_id="/".join(cfg.vae_path.split("/")[:2]), filename="/".join(cfg.vae_path.split("/")[2:])
    )
    vae_ckpt = torch.load(vae_ckpt_path, map_location="cpu")
    worldmem.vae.load_state_dict(vae_ckpt["state_dict"] if "state_dict" in vae_ckpt else vae_ckpt, strict=True)

    video, actions, poses, timestamp = load_single_episode_batch(
        args.dataset_root, args.level_id, args.num_frames, args.memory_condition_length, args.seed
    )
    batch = (video.to(device), actions.to(device), poses.to(device), timestamp.to(device))

    worldmem.validation_step_outputs = []
    with torch.no_grad():
        worldmem.validation_step(batch, 0)
    xs_pred, xs_decode = worldmem.validation_step_outputs[-1]  # (T, B, C, H, W) each, [0,1] float

    def save(x, path):
        x = torch.clamp(x[:, 0], 0, 1) * 255.0  # (T, C, H, W)
        x = x.permute(0, 2, 3, 1).byte().cpu().numpy()  # (T, H, W, C)
        imageio.mimsave(path, x, fps=args.fps)
        print(f"Saved to {path}")

    save(xs_pred, args.output_path)
    if args.gt_output_path:
        save(xs_decode, args.gt_output_path)


if __name__ == "__main__":
    main()
