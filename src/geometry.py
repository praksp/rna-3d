"""
RNA structure geometry utilities.

Uses backend (NumPy or CuPy) for array ops so runs on CPU or GPU.
"""

import numpy as np
from . import backend

xp = backend.xp


def _to_xp(a):
    """Ensure array is on backend (xp)."""
    if a is None:
        return None
    if hasattr(a, "get"):  # CuPy
        return a
    return xp.asarray(a)


def _to_numpy(a):
    return backend.asnumpy(a)


RISE_PER_RESIDUE = 2.81
ROTATION_PER_RESIDUE = 32.7 * 3.141592653589793 / 180.0  # radians
HELIX_RADIUS = 9.4


def generate_aform_helix(sequence_length: int, offset: int = 0):
    """Generate idealized A-form RNA helix C1' coordinates. Returns (sequence_length, 3)."""
    indices = xp.arange(sequence_length, dtype=xp.float32) + offset
    angles = indices * ROTATION_PER_RESIDUE
    coords = xp.zeros((sequence_length, 3), dtype=xp.float32)
    coords[:, 0] = HELIX_RADIUS * xp.cos(angles)
    coords[:, 1] = HELIX_RADIUS * xp.sin(angles)
    coords[:, 2] = indices * RISE_PER_RESIDUE
    return coords


def _sanitize_coords(coords):
    """Replace NaN/Inf with interpolated or zero. Uses NumPy for interp then returns xp."""
    arr = _to_numpy(coords)
    result = arr.copy()
    for col_idx in range(3):
        col = result[:, col_idx]
        bad = ~np.isfinite(col)
        if bad.all():
            result[:, col_idx] = 0.0
            continue
        if bad.any():
            valid_idx = np.where(~bad)[0]
            bad_idx = np.where(bad)[0]
            result[bad_idx, col_idx] = np.interp(bad_idx, valid_idx, col[valid_idx])
    return _to_xp(result)


def resample_coordinates(coords, target_len: int):
    """Resample coordinate array to target_len via interpolation."""
    arr = _to_numpy(coords)
    src_len = len(arr)
    if src_len == target_len:
        return _to_xp(arr.copy())
    result = np.zeros((target_len, 3), dtype=np.float32)
    for dim in range(3):
        result[:, dim] = np.interp(
            np.linspace(0, 1, target_len),
            np.linspace(0, 1, src_len),
            arr[:, dim],
        )
    return _to_xp(result)
