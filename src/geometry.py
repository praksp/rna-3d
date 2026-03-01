"""
RNA structure geometry utilities.

Generates idealized A-form RNA helix coordinates, provides coordinate
manipulation functions, and handles NaN sanitization.

A-form RNA helix parameters:
- Rise per residue: ~2.81 Å
- Rotation per residue: ~32.7°
- Radius from helix axis to C1': ~9.4 Å
"""

import numpy as np


RISE_PER_RESIDUE = 2.81
ROTATION_PER_RESIDUE = np.radians(32.7)
HELIX_RADIUS = 9.4


def generate_aform_helix(sequence_length: int, offset: int = 0) -> np.ndarray:
    """Generate idealized A-form RNA helix C1' coordinates.

    Returns array of shape (sequence_length, 3) with x, y, z coords.
    """
    indices = np.arange(sequence_length) + offset
    angles = indices * ROTATION_PER_RESIDUE
    coords = np.zeros((sequence_length, 3))
    coords[:, 0] = HELIX_RADIUS * np.cos(angles)
    coords[:, 1] = HELIX_RADIUS * np.sin(angles)
    coords[:, 2] = indices * RISE_PER_RESIDUE
    return coords


def _sanitize_coords(coords: np.ndarray) -> np.ndarray:
    """Replace any remaining NaN/Inf values with interpolated or zero values."""
    result = coords.copy()
    for col_idx in range(result.shape[1]):
        col = result[:, col_idx]
        bad = ~np.isfinite(col)
        if bad.all():
            result[:, col_idx] = 0.0
            continue
        if bad.any():
            valid_idx = np.where(~bad)[0]
            bad_idx = np.where(bad)[0]
            result[bad_idx, col_idx] = np.interp(bad_idx, valid_idx, col[valid_idx])
    return result


def resample_coordinates(coords: np.ndarray, target_len: int) -> np.ndarray:
    """Resample coordinate array to a different length using interpolation.

    Preserves overall shape while changing the number of points.
    """
    src_len = len(coords)
    if src_len == target_len:
        return coords.copy()

    result = np.zeros((target_len, 3), dtype=np.float32)
    old_x = np.linspace(0, 1, src_len)
    new_x = np.linspace(0, 1, target_len)

    for dim in range(3):
        result[:, dim] = np.interp(new_x, old_x, coords[:, dim])

    return result
