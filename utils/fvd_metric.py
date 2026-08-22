"""
Fréchet Video Distance (FVD) computation.

No FVD/FID/I3D implementation exists elsewhere in this repo (verified by full-repo
search) despite `README.md`'s "Cite us" section pointing at a paper whose Table 1
reports FVD numbers. This module implements the metric from scratch for the eval
harness in `scripts/eval_fvd.py`.

FVD follows Unterthiner et al. 2018 ("Towards Accurate Generative Models of Video:
A New Metric & Challenges", arXiv:1812.01717) -- the paper PERSIST's Appendix B.1/
B.2 cites for its own FVD protocol -- using a Kinetics-400-pretrained Inflated 3D
ConvNet (I3D; Carreira & Zisserman, CVPR 2017) for feature extraction.

The I3D loading/preprocessing/feature-extraction contract here matches the
TorchScript checkpoint and calling convention used across the video-generation
literature for reporting FVD (StyleGAN-V, VideoGPT, MCVD, and consolidated in
JunyaoHu/common_metrics_on_video_quality's "styleganv" FVD variant), so that
results are comparable to numbers reported by other papers using the same
convention -- rather than inventing a bespoke, non-comparable implementation.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from scipy import linalg

# Public mirror(s) of the standard Kinetics-400 I3D TorchScript checkpoint used for
# FVD reporting across the video-generation literature. Verify reachability before
# relying on this at execution time; pass --i3d-checkpoint with a local path to
# bypass the download entirely if these become unreachable.
I3D_CHECKPOINT_URLS = [
    "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1",
]
I3D_CLIP_LEN = 16  # frames per clip -- matches the PERSIST paper's splicing protocol (Appendix B.1)
I3D_INPUT_SIZE = 224  # spatial resolution expected by the checkpoint
I3D_FEATURE_DIM = 400  # raw pre-softmax Kinetics-400 logits, used as the FVD feature space


def _default_cache_dir() -> Path:
    root = Path(os.environ.get("VOXELWM_CACHE_DIR", Path.home() / ".cache" / "voxelwm"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def download_i3d_checkpoint(cache_dir: Optional[Path] = None) -> Path:
    """Download (and cache) the pretrained I3D TorchScript checkpoint used for FVD."""
    cache_dir = cache_dir or _default_cache_dir()
    dest = cache_dir / "i3d_torchscript.pt"
    if dest.exists():
        return dest

    last_err = None
    for url in I3D_CHECKPOINT_URLS:
        try:
            logger.info(f"Downloading I3D checkpoint from {url} ...")
            urllib.request.urlretrieve(url, dest)
            return dest
        except Exception as e:  # noqa: BLE001
            last_err = e
            logger.warning(f"Failed to download I3D checkpoint from {url}: {e}")
    raise RuntimeError(
        "Could not download the I3D checkpoint used for FVD from any known mirror. "
        "Pass --i3d-checkpoint with a local path to a Kinetics-400 I3D TorchScript "
        "checkpoint (see utils/fvd_metric.py:I3D_CHECKPOINT_URLS)."
    ) from last_err


def load_i3d_model(checkpoint_path: Optional[str] = None, device: str = "cuda") -> torch.jit.ScriptModule:
    """Load the pretrained I3D feature extractor used for FVD."""
    path = Path(checkpoint_path) if checkpoint_path else download_i3d_checkpoint()
    model = torch.jit.load(str(path), map_location=device)
    model.eval()
    return model


def preprocess_clips_for_i3d(clips: torch.Tensor) -> torch.Tensor:
    """Prepare clips for the I3D feature extractor.

    Matches the reference preprocessing (resize shorter side to 224, center-crop
    224x224, then map [0, 1] -> [-1, 1]) used by the FVD implementations this
    module is designed to be comparable with.

    Args:
        clips: (B, T, C, H, W) tensor, RGB, values in [0, 255].

    Returns:
        (B, C, T, 224, 224) float tensor in [-1, 1].
    """
    if clips.dtype != torch.float32:
        clips = clips.float()
    clips = clips / 255.0  # -> [0, 1]

    b, t, c, h, w = clips.shape
    frames = clips.reshape(b * t, c, h, w)

    # Resize shorter side to I3D_INPUT_SIZE, preserving aspect ratio.
    scale = I3D_INPUT_SIZE / min(h, w)
    new_h, new_w = round(h * scale), round(w * scale)
    frames = F.interpolate(frames, size=(new_h, new_w), mode="bilinear", align_corners=False)

    # Center crop to (I3D_INPUT_SIZE, I3D_INPUT_SIZE).
    top = (new_h - I3D_INPUT_SIZE) // 2
    left = (new_w - I3D_INPUT_SIZE) // 2
    frames = frames[:, :, top : top + I3D_INPUT_SIZE, left : left + I3D_INPUT_SIZE]

    frames = (frames - 0.5) * 2.0  # -> [-1, 1]
    clips = frames.reshape(b, t, c, I3D_INPUT_SIZE, I3D_INPUT_SIZE)
    clips = clips.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, C, H, W) -> (B, C, T, H, W)
    return clips


@torch.no_grad()
def extract_i3d_features(
    clips: torch.Tensor,
    i3d: torch.jit.ScriptModule,
    device: str = "cuda",
    batch_size: int = 8,
) -> np.ndarray:
    """Extract I3D features for a set of 16-frame clips.

    Args:
        clips: (N, T, C, H, W) tensor of N clips, RGB uint8/float in [0, 255].
            T should equal I3D_CLIP_LEN (16); the checkpoint requires >=10 frames.
        i3d: loaded I3D TorchScript model (see `load_i3d_model`).
        batch_size: clips per forward pass.

    Returns:
        (N, I3D_FEATURE_DIM) numpy array of I3D features.
    """
    feats = []
    for i in range(0, clips.shape[0], batch_size):
        batch = preprocess_clips_for_i3d(clips[i : i + batch_size]).to(device)
        # rescale/resize=False: we already normalized/resized above ourselves;
        # return_features=True: return pre-softmax logits as the FVD feature space.
        # This exact call contract matches the checkpoint's expected TorchScript API.
        f = i3d(batch, rescale=False, resize=False, return_features=True)
        feats.append(f.detach().cpu().numpy())
    return np.concatenate(feats, axis=0)


def compute_frechet_distance(feats_real: np.ndarray, feats_gen: np.ndarray, eps: float = 1e-6) -> float:
    """Fréchet distance between two feature distributions (Unterthiner et al., 2018)."""
    mu_real, sigma_real = feats_real.mean(axis=0), np.cov(feats_real, rowvar=False)
    mu_gen, sigma_gen = feats_gen.mean(axis=0), np.cov(feats_gen, rowvar=False)

    diff = mu_real - mu_gen
    covmean, _ = linalg.sqrtm(sigma_real @ sigma_gen, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_real.shape[0]) * eps
        covmean, _ = linalg.sqrtm((sigma_real + offset) @ (sigma_gen + offset), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fvd = diff @ diff + np.trace(sigma_real) + np.trace(sigma_gen) - 2 * np.trace(covmean)
    return float(fvd)


def clips_from_video(video: torch.Tensor, clip_len: int = I3D_CLIP_LEN) -> torch.Tensor:
    """Splice a (T, C, H, W) video into non-overlapping clip_len-frame clips.

    Leftover frames that don't fill a full clip are discarded, matching the
    protocol stated in the PERSIST paper (Appendix B.1): "we compute the FVD by
    splicing rollouts into 16-frame clips, discarding leftover frames."
    """
    t = video.shape[0]
    n_clips = t // clip_len
    if n_clips == 0:
        raise ValueError(f"Video has {t} frames, fewer than one {clip_len}-frame clip.")
    video = video[: n_clips * clip_len]
    return video.reshape(n_clips, clip_len, *video.shape[1:])


def compute_fvd(
    real_clips: torch.Tensor,
    gen_clips: torch.Tensor,
    i3d: torch.jit.ScriptModule,
    device: str = "cuda",
    batch_size: int = 8,
) -> float:
    """Top-level convenience: FVD between two sets of (N, T, C, H, W) clips."""
    feats_real = extract_i3d_features(real_clips, i3d, device=device, batch_size=batch_size)
    feats_gen = extract_i3d_features(gen_clips, i3d, device=device, batch_size=batch_size)
    return compute_frechet_distance(feats_real, feats_gen)
