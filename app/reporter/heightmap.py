import logging
import math

import numpy as np

logger = logging.getLogger(__name__)

# Sentinel for xy columns with no model voxel above the candidate top surface.
# NaN is convenient: it cannot be confused with a real height and it is masked
# automatically by most plotting/analysis tools.
DEFAULT_MISSING_VALUE = np.nan


def compute_heightmap(
    voxel_matrix,
    voxel_size,
    z_min,
    k_first=0,
    missing_value=DEFAULT_MISSING_VALUE,
):
    """
    Extract a 2D top-surface heightmap from a voxel matrix.

    For every (x, y) column, the reported height is the highest model voxel
    that also lies above the candidate top surface, i.e. the top face of the
    highest filled voxel with `k * voxel_size >= z_min`.

    Parameters:
    -----------
    voxel_matrix : numpy.ndarray
        3D voxel space (any nonzero value counts as filled).
    voxel_size : float
        Voxel edge length in mm.
    z_min : float
        Candidate top surface: a horizontal plane given in model (G-code)
        Z coordinates in mm. Only voxels entirely at or above this plane
        contribute to the heightmap.
    k_first : int
        Global Z index of voxel_matrix[..., 0], for when voxel_matrix is a
        Z-cropped view of a larger space (default: 0).
    missing_value : float
        Value used for columns without any filled voxel at or above z_min
        (default: NaN).

    Returns:
    --------
    numpy.ndarray
        2D array of shape (nx, ny) with surface heights in mm (model Z
        coordinates), or missing_value where no model voxel at or above the
        candidate surface exists.
    """
    nx, ny, nz = voxel_matrix.shape

    # First Z-plane index whose voxels lie at or above the candidate surface.
    # Voxel k (global) spans (k*voxel_size, (k+1)*voxel_size], so it lies
    # above the plane iff k*voxel_size >= z_min.
    k_min_global = int(math.ceil(z_min / voxel_size - 1e-9))
    k_local_start = max(k_min_global - k_first, 0)

    if k_local_start >= nz:
        logger.warning(
            "[Heightmap]: the candidate top surface z >= %g mm is above the "
            "whole voxel space (top at %g mm); the heightmap is empty",
            z_min,
            (k_first + nz) * voxel_size,
        )
        return np.full((nx, ny), missing_value, dtype=float)

    filled = voxel_matrix[:, :, k_local_start:] > 0
    has_material = filled.any(axis=2)

    # Highest filled voxel per column, counting from the top of the subspace.
    local_max = filled.shape[2] - 1 - np.argmax(filled[:, :, ::-1], axis=2)
    k_global_max = k_first + k_local_start + local_max

    heights = np.full((nx, ny), missing_value, dtype=float)
    heights[has_material] = (k_global_max[has_material] + 1) * voxel_size

    valid = int(np.count_nonzero(has_material))
    logger.info(
        "[Heightmap]: %d of %d columns (%.1f%%) have material at or above "
        "z = %g mm; surface height spans %g..%g mm",
        valid,
        nx * ny,
        100.0 * valid / (nx * ny) if nx * ny else 0.0,
        z_min,
        heights[has_material].min() if valid else np.nan,
        heights[has_material].max() if valid else np.nan,
    )

    return heights
