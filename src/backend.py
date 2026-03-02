"""
Array backend for GPU (CuPy) or CPU (NumPy).

- On Kaggle/Linux with NVIDIA GPU: use CuPy for coordinate and Kabsch ops.
- On Mac (no CUDA) or when CuPy unavailable: use NumPy.
- Run locally with --data-dir/--output; on Kaggle /kaggle/input and /kaggle/working are used.
"""

import os
from pathlib import Path

import numpy as np

# Try CuPy (requires NVIDIA GPU + CUDA). Mac and CPU-only envs fall back to NumPy.
_USE_GPU = os.environ.get("RNA_USE_GPU", "auto").lower()
if _USE_GPU == "0" or _USE_GPU == "false" or _USE_GPU == "no":
    xp = np
    device = "cpu"
else:
    try:
        import cupy as cp
        # Quick test that GPU is usable (e.g. Kaggle GPU, or Linux with CUDA)
        _ = cp.array([1.0])
        xp = cp
        device = "cuda"
    except Exception:
        xp = np
        device = "cpu"


def asnumpy(arr):
    """Return a NumPy array from an array that may be NumPy or CuPy."""
    if arr is None:
        return None
    if hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)


def get_data_dir(override: str = None) -> Path:
    """Data directory: override, or Kaggle input, or local 'data'."""
    if override:
        return Path(override)
    k = Path("/kaggle/input/stanford-rna-3d-folding-2")
    if k.exists():
        if (k / "train_sequences.csv").exists():
            return k
        for d in k.iterdir():
            if d.is_dir() and (d / "train_sequences.csv").exists():
                return d
        return k
    return Path("data")


def get_output_dir(override: str = None) -> Path:
    """Output directory: override, or Kaggle working, or local 'output'."""
    if override:
        return Path(override)
    w = Path("/kaggle/working")
    if w.exists():
        return w
    return Path("output")
