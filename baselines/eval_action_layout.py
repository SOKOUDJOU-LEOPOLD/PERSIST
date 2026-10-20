"""Shared eval-time data loading, factored out so the action-layout fix lives in exactly one
place instead of being duplicated across eval_oasis_finetuned_fvd.py and its shard variant.

The bug this fixes: both eval scripts historically built actions via the RAW 23-dim Craftium path
(npz["action"] -> shift_actions), which is correct for a checkpoint trained WITHOUT
--use-oasis-native-action-layout, but wrong for one trained WITH it (25-dim actions) --
load_finetuned_dit auto-detects action_dim from the checkpoint's own weight shape, so feeding it
the wrong width crashes on a shape mismatch. The fix mirrors the already-correct pattern in
generate_luanti_split_grids.py's load_real_and_native_actions: run
craftium_action_features.augment_actions_to_oasis_native_layout on the RAW 23-dim action pool
(plus player_yaw/player_pitch) to get the matching 25-dim layout before shifting.
"""
import glob
import os
from typing import List, Tuple

import av
import numpy as np
import torch

from craftium_action_features import augment_actions_to_oasis_native_layout
from finetune_oasis_rollout import shift_actions


def list_all_levels(dataset_root: str) -> List[str]:
    """Sorted list of level_* episode dirs directly under dataset_root -- for eval sets like
    persist-eval-sample that have no train/val/test split json (unlike the Luanti-specific runs'
    own split files), so there's nothing to load_split() from."""
    return sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(dataset_root, "level_*"))
        if os.path.isdir(p)
    )


def read_video(path: str, max_frames: int = None) -> np.ndarray:
    """Decodes up to max_frames frames (None = the whole video) via an early break in the decode
    loop -- NOT decode-everything-then-slice. For a 600-frame episode requested at, say, 64
    frames (e.g. probe_batch_size_memory.py's short memory probe), decode-then-slice wastes ~90%
    of the decode work per episode; under filesystem contention (e.g. another job decoding
    videos on the same node at the same time) that waste is what actually caused
    probe_batch_size_memory.py to time out at 30 minutes with ~0s of real CPU time consumed --
    it wasn't stuck, it was I/O-bound waiting on unnecessary decode work."""
    container = av.open(path)
    frames = []
    for i, frame in enumerate(container.decode(container.streams.video[0])):
        if max_frames is not None and i >= max_frames:
            break
        frames.append(frame.to_ndarray(format="rgb24"))
    container.close()
    return np.stack(frames)


def load_eval_actions_and_video(
    dataset_root: str, level_id: str, max_frames: int, use_oasis_native_action_layout: bool,
) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """Returns (real_video (T,H,W,3) uint8, actions (T,A) float tensor already shift_actions'd,
    seed_frame (3,H,W) float tensor in [0,1]) for one episode, T <= max_frames.

    A is 25 when use_oasis_native_action_layout=True (matching a checkpoint trained with
    --use-oasis-native-action-layout, e.g. the Luanti hudnav fine-tune), else 23 (the raw
    Craftium layout, for older non-native-layout checkpoints)."""
    level_dir = os.path.join(dataset_root, level_id)
    real_video = read_video(os.path.join(level_dir, "rgb.mp4"), max_frames)

    npz = np.load(os.path.join(level_dir, "data.npz"))
    if use_oasis_native_action_layout:
        action_pool = augment_actions_to_oasis_native_layout(
            npz["action"], npz["player_yaw"], npz["player_pitch"]
        )
    else:
        action_pool = npz["action"]

    raw_action = torch.from_numpy(action_pool[:max_frames].astype(np.float32))
    actions = shift_actions(raw_action)
    seed_frame = torch.from_numpy(real_video[0]).float().permute(2, 0, 1) / 255.0
    return real_video, actions, seed_frame
