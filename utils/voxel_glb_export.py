"""Export a voxel class grid as a colored `.glb` (binary glTF) mesh, viewable in standard 3D
tools (Blender, three.js, VS Code's built-in 3D preview, etc.) rather than only as a baked 2D
image -- per a MoTWorld-style `voxel_t0.glb` reference shown during this project.

No glTF/mesh-export code exists anywhere in this repo. `utils/voxel_rasterizer.py`'s
`VoxelMeshRasterizer.centers_to_cubes_mesh` already builds a valid one-cube-per-voxel mesh (with
an unused `voxel_rgbs` vertex-coloring parameter, and even a commented-out `trimesh.Trimesh(...).
export(...)` debug line at lines 77-78) -- but it's built for differentiable rasterization
against a camera at the origin (hence its per-voxel "adaptive inset" sized by distance-from-
camera, meaningless for a static export mesh) and always emits a cube for every one of the
`dim**3` grid cells, occupied or not. This module adapts that geometry (fixed small inset instead
of the camera-distance-adaptive one, occupied voxels only) rather than reusing the class as-is.
"""
import numpy as np
import trimesh
from matplotlib import pyplot as plt

# Same tab20/class%20 convention as scripts/eval_qualitative.py:render_voxel_3d and
# utils/voxel_pov_render.py, so all three visualization forms (isometric matplotlib, camera-POV
# render, glb mesh) stay visually consistent with each other.
_CMAP = plt.get_cmap("tab20")

_CORNER_OFFSETS = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
])  # (8, 3)
_TRI_FACES_LOCAL = np.array([
    [0, 3, 2], [0, 2, 1],  # Bottom
    [4, 5, 6], [4, 6, 7],  # Top
    [0, 1, 5], [0, 5, 4],  # Front
    [3, 7, 6], [3, 6, 2],  # Back
    [1, 2, 6], [1, 6, 5],  # Right
    [0, 4, 7], [0, 7, 3],  # Left
])  # (12, 3)


def voxel_classes_to_glb(classes: np.ndarray, empty_id: int, out_path: str, voxel_size: float = 1.0, inset: float = 0.97) -> None:
    """Export occupied voxels (class != empty_id) as a colored cube mesh.

    Args:
        classes: (X, Y, Z) int array of voxel class ids.
        empty_id: class id treated as empty/air, excluded from the mesh entirely (unlike the
            rasterizer, which always builds a dense dim**3-cube mesh -- an export mesh should
            only contain what's actually there to look at).
        out_path: destination path; extension determines format (trimesh dispatches on it --
            use `.glb` for binary glTF).
        inset: fixed per-cube shrink factor (unlike the rasterizer's per-voxel adaptive inset,
            which depends on distance from an assumed origin-camera and is meaningless here) to
            keep adjacent cube faces from z-fighting if ever re-rendered, and to make individual
            voxels visually distinguishable.
    """
    occ_idx = np.argwhere(classes != empty_id)  # (N, 3)
    if occ_idx.size == 0:
        raise ValueError("No occupied voxels to export (grid is entirely empty_id).")

    centers = (occ_idx.astype(np.float64) + 0.5) * voxel_size
    class_ids = classes[tuple(occ_idx.T)]
    colors = (_CMAP((class_ids % 20) / 20.0)[:, :3] * 255).astype(np.uint8)  # (N, 3) RGB

    n = centers.shape[0]
    half_size = (voxel_size / 2.0) * inset
    all_vertices = centers[:, None, :] + _CORNER_OFFSETS[None, :, :] * half_size  # (N, 8, 3)
    vertices = all_vertices.reshape(-1, 3)  # (N*8, 3)
    inv_idx = np.arange(vertices.shape[0]).reshape(n, 8)
    faces = inv_idx[:, _TRI_FACES_LOCAL].reshape(-1, 3)  # (N*12, 3)
    vertex_colors = np.repeat(colors, 8, axis=0)  # each voxel's 8 vertices share its color

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors, process=False)
    mesh.export(out_path)
