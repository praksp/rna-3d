"""
Proper RNA sequence alignment using dynamic programming.

Implements Needleman-Wunsch (global) and Smith-Waterman (local) alignment
with RNA-specific scoring to find the best template match and produce
a residue-level mapping between template and target.
"""

import numpy as np
from functools import lru_cache

MATCH_SCORE = 4
MISMATCH_SCORE = -2
GAP_OPEN = -10
GAP_EXTEND = -1

# RNA transition/transversion scoring: purines (A,G) and pyrimidines (C,U)
_PURINES = set("AG")
_PYRIMIDINES = set("CU")


def _substitution_score(a: str, b: str) -> int:
    if a == b:
        return MATCH_SCORE
    # Transition (purine<->purine or pyrimidine<->pyrimidine) is less penalized
    if (a in _PURINES and b in _PURINES) or (a in _PYRIMIDINES and b in _PYRIMIDINES):
        return MISMATCH_SCORE // 2
    return MISMATCH_SCORE


def needleman_wunsch(seq1: str, seq2: str) -> tuple:
    """Global alignment using Needleman-Wunsch with affine gap penalties.

    Returns (score, alignment_mapping) where alignment_mapping is a list of
    (i, j) tuples mapping seq1[i] -> seq2[j]. Gaps are represented as
    (i, -1) or (-1, j).
    """
    n, m = len(seq1), len(seq2)

    # Score matrices: M (match), X (gap in seq2), Y (gap in seq1)
    NEG_INF = -999999
    M = np.full((n + 1, m + 1), NEG_INF, dtype=np.int32)
    X = np.full((n + 1, m + 1), NEG_INF, dtype=np.int32)
    Y = np.full((n + 1, m + 1), NEG_INF, dtype=np.int32)

    M[0, 0] = 0
    for i in range(1, n + 1):
        X[i, 0] = GAP_OPEN + (i - 1) * GAP_EXTEND
    for j in range(1, m + 1):
        Y[0, j] = GAP_OPEN + (j - 1) * GAP_EXTEND

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = _substitution_score(seq1[i - 1], seq2[j - 1])
            M[i, j] = sub + max(M[i - 1, j - 1], X[i - 1, j - 1], Y[i - 1, j - 1])
            X[i, j] = max(M[i - 1, j] + GAP_OPEN, X[i - 1, j] + GAP_EXTEND)
            Y[i, j] = max(M[i, j - 1] + GAP_OPEN, Y[i, j - 1] + GAP_EXTEND)

    # Traceback
    score = max(M[n, m], X[n, m], Y[n, m])
    mapping = _traceback_global(M, X, Y, seq1, seq2)
    return score, mapping


def _traceback_global(M, X, Y, seq1, seq2):
    """Traceback for global alignment."""
    n, m = len(seq1), len(seq2)
    mapping = []
    i, j = n, m

    # Determine which matrix we end in
    scores = [(M[n, m], 0), (X[n, m], 1), (Y[n, m], 2)]
    state = max(scores, key=lambda x: x[0])[1]

    while i > 0 or j > 0:
        if state == 0:  # Match state
            if i > 0 and j > 0:
                mapping.append((i - 1, j - 1))
                sub = _substitution_score(seq1[i - 1], seq2[j - 1])
                prev_scores = [
                    (M[i - 1, j - 1], 0),
                    (X[i - 1, j - 1], 1),
                    (Y[i - 1, j - 1], 2),
                ]
                state = max(prev_scores, key=lambda x: x[0])[1]
                i -= 1
                j -= 1
            elif i > 0:
                mapping.append((i - 1, -1))
                i -= 1
            else:
                mapping.append((-1, j - 1))
                j -= 1
        elif state == 1:  # Gap in seq2 (consuming seq1)
            mapping.append((i - 1, -1))
            if M[i - 1, j] + GAP_OPEN >= X[i - 1, j] + GAP_EXTEND:
                state = 0
            i -= 1
        else:  # Gap in seq1 (consuming seq2)
            mapping.append((-1, j - 1))
            if M[i, j - 1] + GAP_OPEN >= Y[i, j - 1] + GAP_EXTEND:
                state = 0
            j -= 1

    mapping.reverse()
    return mapping


def smith_waterman(seq1: str, seq2: str) -> tuple:
    """Local alignment using Smith-Waterman with affine gap penalties.

    Returns (score, aligned_pairs) where aligned_pairs is a list of
    (i, j) tuples for the locally aligned region.
    """
    n, m = len(seq1), len(seq2)
    NEG_INF = -999999

    H = np.zeros((n + 1, m + 1), dtype=np.int32)
    X = np.full((n + 1, m + 1), NEG_INF, dtype=np.int32)
    Y = np.full((n + 1, m + 1), NEG_INF, dtype=np.int32)

    max_score = 0
    max_pos = (0, 0)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = _substitution_score(seq1[i - 1], seq2[j - 1])
            H[i, j] = max(0, H[i - 1, j - 1] + sub,
                          X[i - 1, j - 1] + sub,
                          Y[i - 1, j - 1] + sub)
            X[i, j] = max(H[i - 1, j] + GAP_OPEN, X[i - 1, j] + GAP_EXTEND)
            Y[i, j] = max(H[i, j - 1] + GAP_OPEN, Y[i, j - 1] + GAP_EXTEND)

            if H[i, j] > max_score:
                max_score = H[i, j]
                max_pos = (i, j)

    # Traceback from max_pos
    aligned_pairs = []
    i, j = max_pos
    while i > 0 and j > 0 and H[i, j] > 0:
        if H[i, j] == H[i - 1, j - 1] + _substitution_score(seq1[i - 1], seq2[j - 1]):
            aligned_pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif X[i, j] >= Y[i, j]:
            i -= 1
        else:
            j -= 1

    aligned_pairs.reverse()
    return max_score, aligned_pairs


def compute_alignment_score(seq1: str, seq2: str, mode: str = "auto",
                            length_ratio_global_threshold: float = 0.85) -> tuple:
    """Compute alignment score between two sequences.

    Args:
        mode: "global", "local", or "auto" (picks based on length ratio)
        length_ratio_global_threshold: when ratio above this, force global alignment

    Returns:
        (normalized_score, aligned_pairs)
    """
    len1, len2 = len(seq1), len(seq2)
    if len1 == 0 or len2 == 0:
        return 0.0, []

    length_ratio = min(len1, len2) / max(len1, len2)

    if mode == "auto":
        mode = "global" if length_ratio >= length_ratio_global_threshold else "local"

    if mode == "global":
        score, mapping = needleman_wunsch(seq1, seq2)
        matched = [(i, j) for i, j in mapping if i >= 0 and j >= 0]
    else:
        score, matched = smith_waterman(seq1, seq2)

    max_possible = MATCH_SCORE * min(len1, len2)
    normalized = score / max_possible if max_possible > 0 else 0.0
    return max(0.0, normalized), matched


def get_residue_mapping(aligned_pairs: list, target_len: int,
                        template_len: int) -> np.ndarray:
    """Convert alignment pairs to a target->template residue index mapping.

    Returns array of length target_len where mapping[i] = template residue index
    for target residue i. Uses -1 for gaps (no corresponding template residue),
    then fills gaps by nearest-neighbor interpolation.
    """
    mapping = np.full(target_len, -1, dtype=np.int32)

    for t_idx, s_idx in aligned_pairs:
        if 0 <= t_idx < target_len and 0 <= s_idx < template_len:
            mapping[t_idx] = s_idx

    # Fill gaps by interpolation from nearest aligned residues
    aligned_positions = np.where(mapping >= 0)[0]
    if len(aligned_positions) == 0:
        # No alignment at all; map linearly
        for i in range(target_len):
            mapping[i] = min(int(i * template_len / target_len), template_len - 1)
    elif len(aligned_positions) < target_len:
        for i in range(target_len):
            if mapping[i] >= 0:
                continue
            # Find nearest aligned positions
            left = aligned_positions[aligned_positions < i]
            right = aligned_positions[aligned_positions > i]
            if len(left) > 0 and len(right) > 0:
                li, ri = left[-1], right[0]
                lv, rv = mapping[li], mapping[ri]
                frac = (i - li) / (ri - li)
                mapping[i] = min(int(lv + frac * (rv - lv)), template_len - 1)
            elif len(left) > 0:
                li = left[-1]
                offset = i - li
                mapping[i] = min(mapping[li] + offset, template_len - 1)
            else:
                ri = right[0]
                offset = ri - i
                mapping[i] = max(mapping[ri] - offset, 0)

    return mapping
