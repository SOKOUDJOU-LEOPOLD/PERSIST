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
from typing import List, Literal

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

    Two indexing modes, controlled by `mode`:

    "sliding" (the default, used for training): every episode is indexed into ALL possible
        contiguous windows (every start position in [0, 600 - window_len], not just a
        non-overlapping grid) -- these form one pooled index across every episode in the split,
        and __getitem__ draws from that full pool. This is deliberately a much larger, denser
        pool than the old fixed-grid scheme (600 // window_len non-overlapping windows per
        episode) it replaces -- see the Oasis data-scaling study plan's "virtual indexing across
        the entire dataset" section for why.
    "fixed" (used for validation / train-eval tracking): exactly ONE window per episode, always
        starting at frame 0 -- the same window every time __getitem__ is called for a given index,
        so a metric computed over this dataset is directly comparable across repeated calls (e.g.
        at different training steps), not resampled fresh each time the way "sliding" is.
    """

    def __init__(
        self, split_path: str, split: str, window_len: int = 32, dataset_root: str = DATASET_ROOT,
        use_oasis_native_action_layout: bool = False, mode: Literal["sliding", "fixed"] = "sliding",
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
        self.mode = mode
        # Every episode has 600 frames (confirmed: data.npz's action field shape (600, 23) for
        # every level in persist-eval-sample) -- the index pool is built from that, not re-probed
        # per __getitem__.
        self.frames_per_episode = 600

        if mode == "sliding":
            # Every contiguous start position across every episode, pooled into one flat index --
            # (level_idx, start) for start in [0, 600 - window_len].
            starts = list(range(self.frames_per_episode - window_len + 1))
            self._index = [(li, s) for li in range(len(self.level_ids)) for s in starts]
        elif mode == "fixed":
            # Exactly one window per episode, always starting at frame 0.
            self._index = [(li, 0) for li in range(len(self.level_ids))]
        else:
            raise ValueError(f"unknown mode: {mode}")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        level_idx, start = self._index[idx]
        level_id = self.level_ids[level_idx]
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

    ds = CraftiumOasisDataset(args.split_path, "train", window_len=32, mode="sliding")
    expected_pool = len(ds.level_ids) * (600 - 32 + 1)
    assert len(ds) == expected_pool, f"sliding pool size {len(ds)} != expected {expected_pool}"
    print(f"[sliding] {len(ds)} windows from {len(ds.level_ids)} episodes (= {600-32+1} starts/episode)")
    frames, action = ds[0]
    print(f"frames: {frames.shape} {frames.dtype} min={frames.min():.3f} max={frames.max():.3f}")
    print(f"action: {action.shape} {action.dtype} sum={action.sum():.1f}")

    ds_fixed = CraftiumOasisDataset(args.split_path, "train", window_len=32, mode="fixed")
    assert len(ds_fixed) == len(ds_fixed.level_ids), "fixed mode should have exactly one window per episode"
    frames_a, action_a = ds_fixed[0]
    frames_b, action_b = ds_fixed[0]
    assert torch.equal(frames_a, frames_b) and torch.equal(action_a, action_b), (
        "fixed mode must return the SAME window on repeated calls -- got different data"
    )
    print(f"[fixed] {len(ds_fixed)} windows (one per episode), repeated-call determinism verified")
