"""
Kabsch algorithm for optimal rigid-body superposition of 3D structures.

Given two sets of corresponding 3D points, finds the rotation and translation
that minimizes the RMSD between them.
"""

import numpy as np


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> tuple:
    """Compute optimal superposition of P onto Q using the Kabsch algorithm.

    Args:
        P: (N, 3) array of points to be rotated/translated (mobile)
        Q: (N, 3) array of reference points (fixed)

    Returns:
        (R, t, rmsd) where:
        - R: (3, 3) optimal rotation matrix
        - t: (3,) translation vector
        - rmsd: root mean square deviation after superposition
        Apply as: P_aligned = (P - centroid_P) @ R.T + centroid_Q
    """
    assert P.shape == Q.shape and P.shape[1] == 3

    centroid_P = P.mean(axis=0)
    centroid_Q = Q.mean(axis=0)

    p = P - centroid_P
    q = Q - centroid_Q

    H = p.T @ q

    U, S, Vt = np.linalg.svd(H)

    # Correct for reflection
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1.0, 1.0, np.sign(d)])

    R = Vt.T @ sign_matrix @ U.T

    t = centroid_Q - R @ centroid_P

    p_rotated = p @ R.T
    diff = p_rotated - q
    rmsd = np.sqrt((diff ** 2).sum() / len(P))

    return R, t, rmsd


def apply_superposition(coords: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply rotation and translation to coordinates."""
    return coords @ R.T + t


def superimpose_with_alignment(template_coords: np.ndarray,
                                target_coords: np.ndarray,
                                template_indices: np.ndarray,
                                target_indices: np.ndarray) -> tuple:
    """Superimpose template onto target using aligned residue pairs.

    Args:
        template_coords: (M, 3) full template coordinates
        target_coords: (N, 3) full target coordinates (can have placeholder values)
        template_indices: indices into template_coords for aligned residues
        target_indices: indices into target_coords for aligned residues

    Returns:
        (R, t, rmsd) transformation to apply to template_coords
    """
    P = template_coords[template_indices]
    Q = target_coords[target_indices]

    valid = np.isfinite(P).all(axis=1) & np.isfinite(Q).all(axis=1)
    P = P[valid]
    Q = Q[valid]

    if len(P) < 3:
        return np.eye(3), np.zeros(3), float('inf')

    return kabsch_rmsd(P, Q)


def compute_tm_score(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Compute approximate TM-score between two structures.

    TM-score = (1/L) * Σ 1/(1 + (d_i/d_0)^2)
    where d_0 = 1.24 * (L - 15)^(1/3) - 1.8 for proteins,
    adapted for RNA with slightly different constants.
    """
    L = len(coords1)
    if L == 0:
        return 0.0

    # First superimpose
    valid = np.isfinite(coords1).all(axis=1) & np.isfinite(coords2).all(axis=1)
    if valid.sum() < 3:
        return 0.0

    c1_valid = coords1[valid]
    c2_valid = coords2[valid]

    R, t, _ = kabsch_rmsd(c1_valid, c2_valid)
    aligned = apply_superposition(c1_valid, R, t)

    # d_0 for RNA (adapted from protein formula)
    d_0 = 1.24 * max(L - 15, 1) ** (1.0 / 3.0) - 1.8
    d_0 = max(d_0, 0.5)

    distances = np.sqrt(((aligned - c2_valid) ** 2).sum(axis=1))
    tm_sum = (1.0 / (1.0 + (distances / d_0) ** 2)).sum()

    return tm_sum / L
