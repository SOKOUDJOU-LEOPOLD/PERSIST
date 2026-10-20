"""FID (Frechet Inception Distance) feature extraction for the Oasis fine-tuning eval pipeline --
mirrors utils/fvd_metric.py's structure (load model once, extract_*_features per clip/frame,
compute_frechet_distance shared with FVD) so FVD and FID are computed the same way: pool raw
features across every episode/shard, then one Frechet-distance call over the pooled set.

FID did not exist anywhere in this repo's Oasis pipeline before this (only in the separate
WorldMem baseline, baselines/WorldMem/calculate_fid.py, using the same torchmetrics primitive).
Unlike FVD's I3D features (extracted per 16-frame CLIP), FID is a per-FRAME metric -- each frame
independently produces one 2048-dim Inception feature vector, no temporal clipping involved.

We use torchmetrics' FrechetInceptionDistance only as a feature extractor (its own .inception
submodule), bypassing its .update()/.compute() running-sum bookkeeping, so features can be pooled
across shards exactly like utils/fvd_metric.py's I3D features -- symmetric with the existing
FVD shard/merge pattern (Frechet distance isn't linear across differently-sized feature pools, so
it must be computed once on the fully pooled set, not averaged per-shard).
"""
import numpy as np
import torch

FID_FEATURE_DIM = 2048


def load_fid_inception(device: str = "cuda"):
    """Returns the InceptionV3 feature extractor (2048-dim pool3 features, torch-fidelity
    weights via torchmetrics), in eval mode on `device`. Requires the `torch-fidelity` package."""
    from torchmetrics.image.fid import FrechetInceptionDistance

    metric = FrechetInceptionDistance(feature=FID_FEATURE_DIM, normalize=True).to(device)
    metric.eval()
    return metric.inception


@torch.no_grad()
def extract_fid_features(
    frames: torch.Tensor, inception, device: str = "cuda", batch_size: int = 32,
) -> np.ndarray:
    """frames: (N, C, H, W) uint8 in [0, 255] -- one row per FRAME (not per-clip like I3D, FID is
    a per-image metric). Returns (N, 2048) float32 pooled Inception features."""
    feats = []
    for i in range(0, frames.shape[0], batch_size):
        batch = frames[i : i + batch_size].to(device)
        f = inception(batch)
        if f.dim() == 1:
            f = f.unsqueeze(0)
        feats.append(f.float().cpu().numpy())
    return np.concatenate(feats, axis=0) if feats else np.zeros((0, FID_FEATURE_DIM), dtype=np.float32)
