"""Generate the submission CSV file in the required format."""

import pandas as pd
import numpy as np
from pathlib import Path

from . import config


def _sanity_check_coords(df: pd.DataFrame, coord_cols: list, sample_size: int = 200) -> list:
    """Check for NaN/Inf and plausible C1'-C1' distances. Returns list of warnings."""
    warnings = []
    if df[coord_cols].isna().any().any():
        warnings.append("Found NaN in coordinates")
    if not np.isfinite(df[coord_cols].values).all():
        warnings.append("Found non-finite values in coordinates")

    rng = np.random.default_rng(42)
    indices = rng.integers(0, max(1, len(df) - 1), size=min(sample_size, len(df)))
    for i in indices:
        if i + 1 >= len(df):
            continue
        row1, row2 = df.iloc[i], df.iloc[i + 1]
        if row1["ID"].rsplit("_", 1)[0] != row2["ID"].rsplit("_", 1)[0]:
            continue
        for pred_idx in range(1, 6):
            x1 = [row1[f"x_{pred_idx}"], row1[f"y_{pred_idx}"], row1[f"z_{pred_idx}"]]
            x2 = [row2[f"x_{pred_idx}"], row2[f"y_{pred_idx}"], row2[f"z_{pred_idx}"]]
            d = np.sqrt(sum((a - b) ** 2 for a, b in zip(x1, x2)))
            if d > 25:
                warnings.append(f"Very large C1'-C1' distance ({d:.1f} Å) in sample")
                break
        if warnings and "Very large" in warnings[-1]:
            break
    return warnings


def create_submission(predictions: dict, submission_targets: dict,
                      output_path: str) -> pd.DataFrame:
    """Create submission CSV matching the expected format.

    Format: ID, resname, resid, x_1, y_1, z_1, x_2, y_2, z_2, ..., x_5, y_5, z_5

    Args:
        predictions: dict mapping target_id -> list of 5 np.ndarray (n_residues, 3)
        submission_targets: dict mapping target_id -> list of residue info dicts
        output_path: path to write the submission CSV
    """
    rows = []

    for target_id, residues in submission_targets.items():
        if target_id not in predictions:
            print(f"WARNING: No prediction for {target_id}, using zeros")
            for res in residues:
                rows.append({
                    "ID": res["ID"],
                    "resname": res["resname"],
                    "resid": res["resid"],
                    **{f"x_{j+1}": 0.0 for j in range(5)},
                    **{f"y_{j+1}": 0.0 for j in range(5)},
                    **{f"z_{j+1}": 0.0 for j in range(5)},
                })
            continue

        pred_list = predictions[target_id]
        n_residues = len(residues)

        for i, res in enumerate(residues):
            row = {
                "ID": res["ID"],
                "resname": res["resname"],
                "resid": res["resid"],
            }
            decimals = getattr(config, "COORD_DECIMALS", 6)
            for j in range(5):
                if j < len(pred_list) and i < len(pred_list[j]):
                    row[f"x_{j+1}"] = round(float(pred_list[j][i, 0]), decimals)
                    row[f"y_{j+1}"] = round(float(pred_list[j][i, 1]), decimals)
                    row[f"z_{j+1}"] = round(float(pred_list[j][i, 2]), decimals)
                else:
                    row[f"x_{j+1}"] = 0.0
                    row[f"y_{j+1}"] = 0.0
                    row[f"z_{j+1}"] = 0.0
            rows.append(row)

    columns = ["ID", "resname", "resid"]
    for j in range(1, 6):
        columns.extend([f"x_{j}", f"y_{j}", f"z_{j}"])

    df = pd.DataFrame(rows, columns=columns)

    coord_cols = [c for c in columns if c.startswith(("x_", "y_", "z_"))]
    df[coord_cols] = df[coord_cols].fillna(0.0)

    for w in _sanity_check_coords(df, coord_cols):
        print(f"  Sanity: {w}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Submission saved to {output_path}")
    print(f"  Total rows: {len(df)}")
    print(f"  Unique targets: {df['ID'].apply(lambda x: x.rsplit('_', 1)[0]).nunique()}")

    return df


def validate_submission(submission_path: str, sample_submission_path: str) -> bool:
    """Validate that our submission matches the expected format."""
    sub = pd.read_csv(submission_path)
    sample = pd.read_csv(sample_submission_path)

    errors = []

    if list(sub.columns) != list(sample.columns):
        errors.append(f"Column mismatch: {list(sub.columns)} vs {list(sample.columns)}")

    if len(sub) != len(sample):
        errors.append(f"Row count mismatch: {len(sub)} vs {len(sample)}")

    if not sub["ID"].equals(sample["ID"]):
        mismatched = (sub["ID"] != sample["ID"]).sum()
        errors.append(f"ID mismatch in {mismatched} rows")

    coord_cols = [c for c in sub.columns if c.startswith(("x_", "y_", "z_"))]
    all_zero = (sub[coord_cols] == 0).all(axis=1).sum()
    if all_zero > 0:
        print(f"  WARNING: {all_zero} rows have all-zero coordinates")

    if sub[coord_cols].isna().any().any():
        errors.append("Found NaN values in coordinate columns")

    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(f"  - {e}")
        return False

    print("Submission validation PASSED")
    return True
