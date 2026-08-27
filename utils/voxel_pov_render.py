"""Camera-accurate voxel-to-image rendering: "what would the camera see if this voxel grid were
rendered from this pose". No such human-viewable visualization existed anywhere in this repo
before this file.

`utils/voxel_rasterizer.py`'s `VoxelMeshRasterizer` is used internally by `models/dit_pixel.py`
(`forward()`, ~line 841) purely to produce abstract learned feature maps that condition the pixel
denoiser -- never to produce a human-viewable image. But `rasterize()` is a pure per-voxel
feature lookup-and-gather (via nvdiffrast depth-peeled face-id recovery), never a differentiable
blend, so feeding it a class-to-RGB color instead of the model's learned 32-dim embedding is a
drop-in substitution, not a modification to the rasterizer itself.

Reuse chain: voxel class grid -> class-to-RGB lookup table -> `pipelines.pipeline.
VoxelFirstPipeline.project_voxel_local_to_worldcam` (the dataset's voxel-local camera
translation -> true world-to-camera) -> `utils.camera_util.camera_params_to_matrices` (6D
rotation + fov -> view/projection matrices) -> a fresh `VoxelMeshRasterizer` instance.

One thing the internal conditioning path does that a naive single-layer render must not skip:
the "nearest hit" at a given pixel is very often the empty/air class (the camera sits inside
mostly-air space), so taking literally the first depth layer would render a solid air-colored
wall immediately around the camera rather than seeing through to terrain. The pixel denoiser
sidesteps this by conditioning on many peeled depth layers at once (`max_layers=192`) and
learning to interpret them; here, since we want one flat image, we instead scan the peeled
layers per pixel and take the first *non-empty* class -- i.e. skip air layers explicitly, the
image-space equivalent of ray marching through empty space.
"""
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from pipelines.pipeline import VoxelFirstPipeline
from utils.camera_util import camera_params_to_matrices
from utils.voxel_rasterizer import VoxelMeshRasterizer

# Same tab20/class%20 convention as scripts/eval_qualitative.py:render_voxel_3d, so camera-POV
# renders and the existing isometric renders stay visually consistent with each other.
_CMAP = plt.get_cmap("tab20")


def class_to_rgb(class_ids: np.ndarray) -> np.ndarray:
    """(...,) int class ids -> (..., 3) float RGB in [0, 1], tab20-mapped (class_id % 20)."""
    return _CMAP((class_ids % 20) / 20.0)[..., :3]


_rasterizer_cache = {}


def _get_rasterizer(voxel_dim: int, width: int, height: int, device: str) -> VoxelMeshRasterizer:
    """Rasterizers own a fixed mesh sized to (voxel_dim, width, height) built once at
    construction; cached across calls since building it is the expensive part, not rasterizing."""
    key = (voxel_dim, width, height, device)
    if key not in _rasterizer_cache:
        _rasterizer_cache[key] = VoxelMeshRasterizer(
            dim=voxel_dim, width=width, height=height, max_layers=voxel_dim * 4, device=device
        )
    return _rasterizer_cache[key]


@torch.no_grad()
def render_voxel_pov(
    classes: torch.Tensor,
    camera: torch.Tensor,
    empty_id: int,
    device: str,
    width: int = 640,
    height: int = 360,
    background_rgb: Tuple[float, float, float] = (0.53, 0.81, 0.92),
) -> np.ndarray:
    """Render a single (X, Y, Z) voxel class grid as seen from `camera`.

    Args:
        classes: (X, Y, Z) int tensor of voxel class ids (e.g. from `pipe.decode_voxels()` or
            `batch["voxel_classes"]`).
        camera: (10,) tensor, the dataset's own camera representation
            [rot6d(6), cam_pos_local(3), fov(1)] (e.g. a single timestep of `rollout["camera"]`
            or `batch["camera"]`).
        empty_id: class id treated as empty/air (see `resolve_empty_class_id` in
            `scripts/eval_qualitative.py` -- resolved per-dataset from `mt_voxel_classdict.json`,
            not a fixed constant).
        background_rgb: color for pixels where every depth layer is empty (open sky).

    Returns:
        (height, width, 3) uint8 RGB image.
    """
    voxel_dim = classes.shape[0]
    assert classes.shape == (voxel_dim, voxel_dim, voxel_dim), classes.shape

    # Rasterize CLASS IDS (not colors) as a single-channel feature, so we can identify and skip
    # empty/air layers per pixel before mapping to color -- see module docstring.
    class_feature = classes.reshape(1, voxel_dim**3, 1).float().to(device)
    background_feature = torch.tensor([float(empty_id) - 1], device=device)  # never a real class id

    cam = camera.clone().reshape(1, 1, -1).to(device)
    cam = VoxelFirstPipeline.project_voxel_local_to_worldcam(cam)
    view, proj = camera_params_to_matrices(cam, image_width=width, image_height=height)
    view, proj = view.reshape(1, 4, 4), proj.reshape(1, 4, 4)

    rasterizer = _get_rasterizer(voxel_dim, width, height, device)
    pixel_class, _depth = rasterizer.rasterize(class_feature, view, proj, background_feature)
    # (layers, 1, H, W, 1) -> (layers, H, W)
    pixel_class = pixel_class[:, 0, :, :, 0].round().long().cpu().numpy()

    is_solid = pixel_class != empty_id  # also true for the background sentinel; handled below
    is_background = pixel_class == (empty_id - 1)
    is_solid &= ~is_background
    first_layer = np.argmax(is_solid, axis=0)  # (H, W); 0 (arbitrary) where no True at all
    any_solid = is_solid.any(axis=0)
    final_class = np.take_along_axis(pixel_class, first_layer[None], axis=0)[0]  # (H, W)

    rgb = class_to_rgb(final_class)
    rgb[~any_solid] = background_rgb
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
