"""Craftium episode -> WorldMem training-window dataset.

Restores fidelity to the original WorldMem codebase in the two places an earlier version of this
file deliberately simplified (see git history for that version's own docstring) -- both
restorations informed by empirically checking real Craftium data first, not blind copying:

1. **Pose source and normalization** (`pitch_yaw_from_cam_dir`, `normalize_pose`, both below):
   camera pitch/yaw are derived from Craftium's `cam_dir` field (a per-frame ground-truth 3D
   look-direction vector), NOT read from the dataset's own `player_pitch`/`player_yaw` fields.
   This isn't a style choice -- those two fields are confirmed broken on real Craftium data:
   `player_pitch` is frozen (identical to the previous frame) across 96% of consecutive frame
   pairs in a sampled episode, while the actual rendered video keeps rotating the whole time (and
   `player_yaw` is similarly stale ~62% of the time). Feeding the model a pose signal that says
   "the camera barely moved" while the real video shows continuous rotation is a severe
   ground-truth mismatch, and was traced (via a controlled diagnostic: the model's OWN memory
   selection function switches from a trivial early-window branch to real pose-based FOV math at
   exactly frame index `memory_condition_length`, and corrupted output was confirmed to begin at
   exactly that frame, in BOTH a fine-tuned checkpoint and the 100%-untouched pretrained one) as
   the root cause of a severe visual-corruption bug that survived three earlier, unrelated
   "fixes" (see git history) -- because none of them were the actual problem. `pitch_yaw_from_cam_dir`
   derives pitch/yaw from `cam_dir` via the exact algebraic inverse of df_video.py's own
   `euler_to_rotation_matrix(pitch, yaw)`, verified by round-tripping real Craftium data back
   through that matrix and reproducing `cam_dir` with 0.0 error -- so this is not a guess, it's
   the codebase's own convention read out directly from ground truth. `normalize_pose` still
   subtracts the window's first-frame position (positions relative, not absolute-world-coordinate)
   and wraps angles to [0, 360) (harmless -- both consumers, the FOV-membership test and the
   Plucker/rotation-matrix conditioning, are exactly 360-periodic in pitch/yaw). An earlier version
   of this function also negated yaw; that negation was compensating for using the broken
   player_yaw field in the first place and is removed now that pitch/yaw come from cam_dir.
2. **Memory-frame selection** (`_select_memory_indices`, mirrored below): the earlier version
   used pure random selection, then (still not the real fix) the pos_range/angle_range
   distance-threshold method from `MinecraftVideoDataset.load_data`. Neither is what the model
   actually uses at generation time. `training_step` (df_video.py) never calls any selection
   function at all -- it just trusts whatever memory frames the *dataset* hands it -- while
   `validation_step`/`interactive()` (i.e. every real rollout, including our own fine-tuned
   qualitative checks and FVD evaluation) select memory frames via
   `_generate_condition_indices`: a greedy "FOV overlap ratio + recency" matcher, which the
   paper's own ablation credits for the entire working-vs-broken difference (rFID 15.37 with it
   vs. 47.35 with random selection). Using a *different* selection rule for training than for
   generation is a real train/inference mismatch on top of whatever else was going on, so this
   file now ports `_generate_condition_indices` VERBATIM (including its "remove the newly-picked
   frame's covered FOV region before picking the next one" step, previously missed) rather than
   `MinecraftVideoDataset`'s separate, simpler distance-threshold heuristic -- training now sees
   exactly the same memory-selection distribution generation will use. All of this function's
   numeric constants (num_samples=10000, radius=30, +/-105 deg horizontal FOV, +/-75 deg vertical
   FOV, -0.2 recency weight) are copied unchanged from the original -- they describe a camera's
   field of view, not a Craftium-specific quantity, so there is nothing here to recalibrate.

Actions are still Craftium's native 23-dim vector (matching finetune_worldmem.py's model-surgery
swap), not MineDojo's `convert_action_space()` lossy remap -- that adaptation was correct from the
start and is unchanged. No pre-extraction to disk (disk space reasons, see finetune_oasis_data.py).
"""
import json
import os
import sys
from typing import List

import av
import numpy as np
import torch
from torch.utils.data import Dataset

# Needed both when this module is run standalone (its own __main__ smoke test below) and as a
# defensive no-op when imported from a caller that already did this (finetune_worldmem.py and
# every eval script that imports CraftiumWorldMemDataset insert the same two paths before
# importing this module, so on that path these are harmless duplicates).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "WorldMem"))

from algorithms.worldmem.df_video import generate_points_in_sphere, is_inside_fov_3d_hv  # noqa: E402

DATASET_ROOT = (
    "/home/lsokoudj/.cache/huggingface/hub/datasets--PERSIST-team--persist-eval-sample/"
    "snapshots/d3a3bee6560304397093ae66523451b9e0d067f7"
)


def load_split(split_path: str, split: str) -> List[str]:
    with open(split_path) as f:
        data = json.load(f)
    return data[split]


def read_frame_range(path: str, indices: List[int]) -> np.ndarray:
    """Decode exactly the frames at `indices` (need not be contiguous -- used for both the
    current window and the scattered memory frames) via one linear decode pass."""
    wanted = set(indices)
    frames_by_idx = {}
    container = av.open(path)
    for i, frame in enumerate(container.decode(container.streams.video[0])):
        if i in wanted:
            frames_by_idx[i] = frame.to_ndarray(format="rgb24")
        if i >= max(wanted):
            break
    container.close()
    return np.stack([frames_by_idx[i] for i in indices])  # (T, H, W, 3) uint8


def pitch_yaw_from_cam_dir(cam_dir: np.ndarray) -> tuple:
    """Derive (pitch, yaw) in degrees directly from Craftium's `cam_dir` field (a per-frame 3D
    camera-facing vector, ground truth from the engine) -- used INSTEAD of the dataset's own
    `player_pitch`/`player_yaw` fields, which turned out to be unreliable (see module docstring).

    The formula is the exact algebraic inverse of df_video.py's own
    `euler_to_rotation_matrix(pitch, yaw)` (verified by round-tripping: reconstructing a forward
    vector from this function's output and feeding it back through that exact matrix reproduces
    `cam_dir` with 0.0 error on real Craftium data):
        forward = R_yaw @ R_pitch @ (0, 0, 1)
                = (sin(yaw)*cos(pitch), -sin(pitch), cos(yaw)*cos(pitch))
    Inverting: yaw = atan2(forward_x, forward_z), pitch = -atan2(forward_y, sqrt(forward_x^2 +
    forward_z^2)). No sign flip on yaw is needed with this derivation (unlike the old
    normalize_pose, which negated a *different*, unreliable source -- see below)."""
    x, y, z = cam_dir[..., 0], cam_dir[..., 1], cam_dir[..., 2]
    yaw = np.degrees(np.arctan2(x, z))
    pitch = -np.degrees(np.arctan2(y, np.sqrt(x**2 + z**2)))
    return pitch, yaw


def normalize_pose(pose: np.ndarray, ref_pose: np.ndarray) -> np.ndarray:
    """Positions become relative to ref_pose's first frame; angle components (pitch, yaw) wrap to
    [0, 360) (harmless for both consumers -- df_video.py's own FOV-membership test and its
    Plucker/rotation-matrix conditioning are both exactly 360-periodic in pitch/yaw, verified from
    their source). `pose` and `ref_pose` may be the same array -- callers must pass ref_pose's
    un-normalized first row, i.e. call order matters (see __getitem__ below).

    Earlier versions of this function also negated yaw, following (what turned out to be a
    misreading of) MinecraftVideoDataset.load_data's own inline normalize_pose. That negation is
    REMOVED here: it was compensating for using player_yaw/player_pitch at all, two fields
    confirmed broken on real Craftium data (see module docstring) -- once pitch/yaw are derived
    from cam_dir via pitch_yaw_from_cam_dir (whose exact 0-error round-trip against
    euler_to_rotation_matrix's own convention was verified directly), no sign correction is
    needed or correct."""
    pose = pose.copy()
    pose[:, :3] -= ref_pose[:1, :3]
    pose[:, 3:] %= 360
    return pose


class CraftiumWorldMemDataset(Dataset):
    """Yields (video, actions, poses, timestamp) tuples for WorldMem fine-tuning.

    video: (n_frames + memory_condition_length, 3, 360, 640) float32 in [0, 1].
    actions: (n_frames + memory_condition_length, 23) float32, Craftium's native action vector.
    poses: (n_frames + memory_condition_length, 5) float32, (x, y, z, pitch, yaw) -- position
        relative to this window's first frame, yaw negated and wrapped -- see normalize_pose.
    timestamp: (n_frames + memory_condition_length,) int64 frame indices within the episode.
    """

    def __init__(
        self,
        split_path: str,
        split: str,
        n_frames: int = 8,
        memory_condition_length: int = 8,
        dataset_root: str = DATASET_ROOT,
        seed: int = 0,
    ):
        self.n_frames = n_frames
        self.memory_condition_length = memory_condition_length
        self.dataset_root = dataset_root
        self.level_ids = load_split(split_path, split)
        self.frames_per_episode = 600
        self.windows_per_episode = self.frames_per_episode // n_frames
        self._seed = seed

    def __len__(self) -> int:
        return len(self.level_ids) * self.windows_per_episode

    def _select_memory_indices(self, poses_pool: np.ndarray, start: int) -> np.ndarray:
        """EXACT port of df_video.py:_generate_condition_indices -- the real "greedy matching
        algorithm based on FOV overlap ratio and timestamp differences" the WorldMem paper's own
        ablation credits for the difference between working memory (rFID 15.37) and "severe
        quality degradation" (rFID 47.35, random selection) -- confirmed by reading the paper
        directly, not just the code. An earlier version of this file ported the WRONG memory-
        selection function (MinecraftVideoDataset.load_data's separate pos_range/angle_range
        distance-threshold matching, used only for the dataset class, not this one) -- that meant
        training and validation/generation (which calls _generate_condition_indices, unmodified)
        used two DIFFERENT selection methods, a real train/inference mismatch on top of whatever
        else was going on. This ports the SAME function actually used at generation time, so
        training sees exactly the same memory-selection distribution generation will use.

        Operates on the ALREADY-NORMALIZED full-episode pose pool (poses_pool, (600, 5), relative
        to this window's first frame -- see normalize_pose)."""
        curr_frame = start
        horizon = self.n_frames
        pose_conditions = torch.from_numpy(poses_pool).float()[:, None, :]  # (600, 1, 5) -- batch dim of 1

        if curr_frame < self.memory_condition_length:
            idx = [i for i in range(curr_frame)] + [0] * (self.memory_condition_length - curr_frame)
            return np.array(idx)

        # Every numeric literal below (num_samples, radius, the two FOV half-angles, the -0.2
        # recency weight) is copied verbatim from _generate_condition_indices -- these describe a
        # camera's field of view and a fixed search radius, not anything Craftium-specific, so
        # there is nothing here to recalibrate (unlike the old pos_range/angle_range approach).
        num_samples = 10000
        radius = 30
        points = generate_points_in_sphere(num_samples, radius)[:, None, :]  # (N, 1, 3)
        points = points + pose_conditions[curr_frame, :, :3][None]

        fov_half_h = torch.tensor(105 / 2)
        fov_half_v = torch.tensor(75 / 2)

        in_fov1 = torch.stack([
            is_inside_fov_3d_hv(points, pc[:, :3], pc[:, -2], pc[:, -1], fov_half_h, fov_half_v)
            for pc in pose_conditions[curr_frame : curr_frame + horizon]
        ])
        in_fov1 = torch.sum(in_fov1, 0) > 0  # (N, 1) -- visible in the upcoming window

        in_fov_list = torch.stack([
            is_inside_fov_3d_hv(points, pc[:, :3], pc[:, -2], pc[:, -1], fov_half_h, fov_half_v)
            for pc in pose_conditions[:curr_frame]
        ])  # (curr_frame, N, 1)

        frame_idx = torch.arange(curr_frame)  # absolute frame index of each candidate past frame
        selected = []
        for _ in range(self.memory_condition_length):
            denom = in_fov1.sum().clamp(min=1)  # recomputed each round: in_fov1 shrinks below
            overlap_ratio = ((in_fov1.bool() & in_fov_list).sum(1)) / denom
            confidence = overlap_ratio + (curr_frame - frame_idx)[:, None] / curr_frame * (-0.2)
            if len(selected) > 0:
                confidence[torch.cat(selected)] = -1e10
            _, r_idx = torch.topk(confidence, k=1, dim=0)
            selected.append(r_idx[0])

            # "choice 1" from the original: remove the region the just-picked frame already
            # covers, so the next greedy pick targets a different part of the FOV instead of
            # re-covering the same ground.
            occupied_mask = in_fov_list[r_idx[0, 0]]  # (N, 1) -- picked frame's own FOV mask
            in_fov1 = in_fov1 & ~occupied_mask
        return torch.cat(selected).numpy()

    def __getitem__(self, idx: int):
        level_idx, window_idx = divmod(idx, self.windows_per_episode)
        level_id = self.level_ids[level_idx]
        start = window_idx * self.n_frames
        level_dir = os.path.join(self.dataset_root, level_id)

        npz = np.load(os.path.join(level_dir, "data.npz"))
        current_idx = list(range(start, start + self.n_frames))

        pos_pool = npz["player_pos"].astype(np.float64)  # (600, 3)
        # player_pitch/player_yaw are NOT used -- confirmed stale (player_pitch alone is frozen
        # across 96% of consecutive frame pairs on real Craftium data, while the video visibly
        # keeps rotating) -- see pitch_yaw_from_cam_dir's docstring and the module docstring.
        pitch_pool, yaw_pool = pitch_yaw_from_cam_dir(npz["cam_dir"].astype(np.float64))
        poses_pool_raw = np.concatenate([pos_pool, pitch_pool[:, None], yaw_pool[:, None]], axis=1)
        poses_raw = poses_pool_raw[current_idx].copy()

        # Exact original call order: normalize the whole pool using the CURRENT window's
        # (pre-normalization) first frame as reference, then normalize the window itself the same way.
        poses_pool = normalize_pose(poses_pool_raw, poses_raw)
        poses_window = normalize_pose(poses_raw, poses_raw)

        memory_idx = self._select_memory_indices(poses_pool, start).tolist()

        all_idx = current_idx + memory_idx
        frames = read_frame_range(os.path.join(level_dir, "rgb.mp4"), all_idx)
        video = torch.from_numpy(frames).float() / 255.0  # (T, H, W, 3)
        video = video.permute(0, 3, 1, 2).contiguous()  # (T, 3, H, W)

        actions = torch.from_numpy(npz["action"][all_idx].astype(np.float32))  # (T, 23)

        poses_all = np.concatenate([poses_window, poses_pool[memory_idx]], axis=0).astype(np.float32)
        poses = torch.from_numpy(poses_all)

        timestamp = torch.tensor(all_idx, dtype=torch.long)

        return video, actions, poses, timestamp

    @property
    def level_ids_used(self) -> List[str]:
        return self.level_ids


if __name__ == "__main__":
    import tyro
    from dataclasses import dataclass

    @dataclass
    class _SmokeTestArgs:
        split_path: str = "outputs/finetune_split.json"

    args = tyro.cli(_SmokeTestArgs)
    ds = CraftiumWorldMemDataset(args.split_path, "train")
    print(f"Dataset: {len(ds)} windows from {len(ds.level_ids)} episodes")
    video, actions, poses, timestamp = ds[5]  # window_idx > 0, exercises the real memory-frame path
    print(f"video: {video.shape} {video.dtype} min={video.min():.3f} max={video.max():.3f}")
    print(f"actions: {actions.shape} sum={actions.sum():.1f}")
    print(f"poses: {poses.shape}\n{poses}")
    print(f"timestamp: {timestamp}")
