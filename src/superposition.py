"""
Kabsch algorithm for optimal rigid-body superposition.

Uses backend (NumPy or CuPy) for GPU when available.
"""

from . import backend

xp = backend.xp


def kabsch_rmsd(P, Q):
    """Compute optimal superposition of P onto Q. P, Q: (N, 3). Returns (R, t, rmsd)."""
    assert P.shape == Q.shape and P.shape[1] == 3
    centroid_P = P.mean(axis=0)
    centroid_Q = Q.mean(axis=0)
    p = P - centroid_P
    q = Q - centroid_Q
    H = p.T @ q
    U, S, Vt = xp.linalg.svd(H)
    d = xp.linalg.det(Vt.T @ U.T)
    sign_matrix = xp.diag(xp.array([1.0, 1.0, float(xp.sign(d))], dtype=xp.float64))
    R = Vt.T @ sign_matrix @ U.T
    t = centroid_Q - R @ centroid_P
    p_rotated = p @ R.T
    diff = p_rotated - q
    rmsd = float(xp.sqrt((diff ** 2).sum() / len(P)))
    return R, t, rmsd


def apply_superposition(coords, R, t):
    """Apply rotation R and translation t to coords."""
    return coords @ R.T + t


def superimpose_with_alignment(template_coords, target_coords, template_indices, target_indices):
    """Superimpose using aligned residue pairs. Indices can be numpy or backend arrays."""
    P = template_coords[template_indices]
    Q = target_coords[target_indices]
    valid = xp.isfinite(P).all(axis=1) & xp.isfinite(Q).all(axis=1)
    P = P[valid]
    Q = Q[valid]
    if len(P) < 3:
        return xp.eye(3, dtype=xp.float64), xp.zeros(3, dtype=xp.float64), float("inf")
    return kabsch_rmsd(P, Q)


def compute_tm_score(coords1, coords2):
    """Approximate TM-score between two structures."""
    L = len(coords1)
    if L == 0:
        return 0.0
    valid = xp.isfinite(coords1).all(axis=1) & xp.isfinite(coords2).all(axis=1)
    if int(valid.sum()) < 3:
        return 0.0
    c1_valid = coords1[valid]
    c2_valid = coords2[valid]
    R, t, _ = kabsch_rmsd(c1_valid, c2_valid)
    aligned = apply_superposition(c1_valid, R, t)
    d_0 = max(1.24 * max(L - 15, 1) ** (1.0 / 3.0) - 1.8, 0.5)
    distances = xp.sqrt(((aligned - c2_valid) ** 2).sum(axis=1))
    tm_sum = float((1.0 / (1.0 + (distances / d_0) ** 2)).sum())
    return tm_sum / L
