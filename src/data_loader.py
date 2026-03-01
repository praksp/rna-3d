"""Load and parse competition data files."""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def load_train_sequences(data_dir: str) -> pd.DataFrame:
    """Load training sequences with metadata."""
    path = Path(data_dir) / "train_sequences.csv"
    df = pd.read_csv(path, low_memory=False)
    df["seq_len"] = df["sequence"].apply(len)
    return df


def load_test_sequences(data_dir: str) -> pd.DataFrame:
    """Load test sequences with metadata."""
    path = Path(data_dir) / "test_sequences.csv"
    df = pd.read_csv(path)
    df["seq_len"] = df["sequence"].apply(len)
    return df


def load_sample_submission(data_dir: str) -> pd.DataFrame:
    """Load sample submission to get exact output format."""
    path = Path(data_dir) / "sample_submission.csv"
    return pd.read_csv(path)


def load_train_labels(data_dir: str) -> pd.DataFrame:
    """Load training labels (C1' coordinates).

    Returns DataFrame with columns: ID, resname, resid, x_1, y_1, z_1, chain, copy
    """
    path = Path(data_dir) / "train_labels.csv"
    df = pd.read_csv(path, low_memory=False)
    df["target_id"] = df["ID"].apply(lambda x: x.rsplit("_", 1)[0])
    return df


def build_train_structure_lookup(train_labels: pd.DataFrame) -> dict:
    """Build a dictionary mapping target_id -> structure data.

    Keeps ALL copies (not just copy 1) so we can use multiple conformations
    for diversity. Fills NaN coordinates via linear interpolation.
    Returns dict: target_id -> dict with:
        - coords: np.ndarray (n_residues, 3) for copy 1
        - all_copies: dict of copy_id -> np.ndarray
        - resnames, resids, chains
        - n_copies: number of copies available
    """
    print("Building training structure lookup (all copies)...")
    lookup = {}
    grouped = train_labels.groupby("target_id")

    for target_id, group in tqdm(grouped, desc="Processing structures"):
        copies_available = sorted(group["copy"].unique())
        all_copies = {}

        for copy_id in copies_available:
            copy_group = group[group["copy"] == copy_id].sort_values("resid")
            coords = copy_group[["x_1", "y_1", "z_1"]].values.astype(np.float32)
            _interpolate_nan(coords)
            all_copies[copy_id] = coords

        primary = group[group["copy"] == copies_available[0]].sort_values("resid")
        primary_coords = all_copies[copies_available[0]]

        lookup[target_id] = {
            "coords": primary_coords,
            "all_copies": all_copies,
            "resnames": primary["resname"].values.tolist(),
            "resids": primary["resid"].values.tolist(),
            "chains": primary["chain"].values.tolist(),
            "n_copies": len(copies_available),
        }

    print(f"Loaded {len(lookup)} training structures")
    return lookup


def _interpolate_nan(coords: np.ndarray):
    """In-place NaN interpolation for coordinate arrays."""
    for col_idx in range(3):
        col = coords[:, col_idx]
        nan_mask = np.isnan(col)
        if nan_mask.all():
            coords[:, col_idx] = 0.0
            continue
        if nan_mask.any():
            valid_idx = np.where(~nan_mask)[0]
            nan_idx = np.where(nan_mask)[0]
            coords[nan_idx, col_idx] = np.interp(nan_idx, valid_idx, col[valid_idx])


def parse_submission_targets(sample_sub: pd.DataFrame) -> dict:
    """Parse sample submission to understand what needs to be predicted per target.

    Returns dict: target_id -> list of dicts with resname, resid info.
    """
    targets = defaultdict(list)
    for _, row in sample_sub.iterrows():
        target_id = row["ID"].rsplit("_", 1)[0]
        resid = int(row["ID"].rsplit("_", 1)[1])
        targets[target_id].append({
            "ID": row["ID"],
            "resname": row["resname"],
            "resid": resid,
        })
    return dict(targets)
