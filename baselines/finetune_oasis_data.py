"""Craftium episode -> Oasis training-window dataset.

Reads directly from the already-downloaded persist-eval-sample HF cache snapshot (rgb.mp4 +
data.npz per episode) -- no pre-extraction to disk (see plan: disk is at 99% capacity, only
~32GB free; pre-extracting all 256 episodes' raw frames would be ~106GB and will not fit).
Each `__getitem__` seeks into the episode's video with `av` and decodes only the needed window,
matching the pattern already used elsewhere in this repo (e.g. utils/fvd_metric.py's clip
splicing) rather than decoding whole 600-frame episodes per sample.

Action convention matches baselines/open-oasis/utils.py:load_actions exactly: the action at
index t is the action that produced the transition INTO frame t, so actions are shifted by one
and frame 0 gets an all-zero action (there is no action preceding the first frame).
"""
import json
import os
from dataclasses import dataclass
from typing import List

import av
import numpy as np
import torch
from torch.utils.data import Dataset

DATASET_ROOT = (
    "/home/lsokoudj/.cache/huggingface/hub/datasets--PERSIST-team--persist-eval-sample/"
    "snapshots/d3a3bee6560304397093ae66523451b9e0d067f7"
)


def load_split(split_path: str, split: str) -> List[str]:
    with open(split_path) as f:
        data = json.load(f)
    return data[split]


def read_video_window(path: str, start: int, length: int) -> np.ndarray:
    """Decode exactly `length` frames starting at frame index `start`, via direct container
    decode + skip (Craftium videos are short -- 600 frames/25s -- so a linear decode-and-skip is
    simple and fast enough; avoids requiring a seek-table/keyframe-aligned seek which mp4 short
    clips don't reliably support anyway)."""
    container = av.open(path)
    frames = []
    for i, frame in enumerate(container.decode(container.streams.video[0])):
        if i < start:
            continue
        if i >= start + length:
            break
        frames.append(frame.to_ndarray(format="rgb24"))
    container.close()
    return np.stack(frames)  # (T, H, W, 3) uint8


class CraftiumOasisDataset(Dataset):
    """Yields (frames, actions) windows for Oasis fine-tuning.

    frames: (T, 3, 360, 640) float32 in [0, 1] -- matches baselines/open-oasis/utils.py:load_prompt.
    actions: (T, 23) float32 -- Craftium's native action vector, shifted by one frame per
        load_actions' convention (frame 0's action is all-zero).
    """

    def __init__(
        self, split_path: str, split: str, window_len: int = 32, dataset_root: str = DATASET_ROOT,
        use_oasis_native_action_layout: bool = False,
    ):
        """use_oasis_native_action_layout: if set, actions are reconstructed into Oasis's OWN
        25-slot ACTION_KEYS column layout (augment_actions_to_oasis_native_layout) instead of
        Craftium's raw 23-dim vector -- matches the exact column semantics (including real,
        Oasis-bucket-scaled camera magnitude, not sign) the pretrained checkpoint's
        action-conditioning weights were trained against. See craftium_action_features.py's
        module-level mapping table for the full derivation."""
        self.window_len = window_len
        self.dataset_root = dataset_root
        self.level_ids = load_split(split_path, split)
        self.use_oasis_native_action_layout = use_oasis_native_action_layout
        # Every episode has 600 frames (confirmed: data.npz's action field shape (600, 23) for
        # every level in persist-eval-sample) -- windows per episode computed from that, not
        # re-probed per __getitem__.
        self.frames_per_episode = 600
        self.windows_per_episode = self.frames_per_episode // window_len

    def __len__(self) -> int:
        return len(self.level_ids) * self.windows_per_episode

    def __getitem__(self, idx: int):
        level_idx, window_idx = divmod(idx, self.windows_per_episode)
        level_id = self.level_ids[level_idx]
        start = window_idx * self.window_len
        level_dir = os.path.join(self.dataset_root, level_id)

        frames = read_video_window(os.path.join(level_dir, "rgb.mp4"), start, self.window_len)
        frames = torch.from_numpy(frames).float() / 255.0  # (T, H, W, 3)
        frames = frames.permute(0, 3, 1, 2).contiguous()  # (T, 3, H, W)

        npz = np.load(os.path.join(level_dir, "data.npz"))
        if self.use_oasis_native_action_layout:
            from craftium_action_features import augment_actions_to_oasis_native_layout
            # Applied to the FULL 600-frame pool, not the window slice: camera-delta computation
            # (compute_clean_camera_deltas) needs consecutive full-pool frames to diff against,
            # same reason finetune_worldmem_data.py's dataset does this pool-side too.
            action_pool = augment_actions_to_oasis_native_layout(
                npz["action"], npz["player_yaw"], npz["player_pitch"]
            )  # (600, 25)
        else:
            action_pool = npz["action"].astype(np.float32)  # (600, 23)

        action = torch.from_numpy(action_pool[start : start + self.window_len])  # (T, 23) or (T, 25)
        # Shift by one frame: action[t] becomes "the action that produced frame t" (load_actions
        # convention). For a mid-episode window (start > 0), the action preceding this window's
        # first frame is the real action_pool[start - 1], not zero -- only true episode
        # boundaries (start == 0) get an all-zero first action.
        if start == 0:
            prev_action = torch.zeros(1, action.shape[1])
        else:
            prev_action = torch.from_numpy(action_pool[start - 1 : start])
        action = torch.cat([prev_action, action[:-1]], dim=0)

        return frames, action

    @property
    def level_ids_used(self) -> List[str]:
        return self.level_ids


@dataclass
class _SmokeTestArgs:
    split_path: str = "outputs/finetune_split.json"


if __name__ == "__main__":
    import tyro

    args = tyro.cli(_SmokeTestArgs)
    ds = CraftiumOasisDataset(args.split_path, "train", window_len=32)
    print(f"Dataset: {len(ds)} windows from {len(ds.level_ids)} episodes")
    frames, action = ds[0]
    print(f"frames: {frames.shape} {frames.dtype} min={frames.min():.3f} max={frames.max():.3f}")
    print(f"action: {action.shape} {action.dtype} sum={action.sum():.1f}")
