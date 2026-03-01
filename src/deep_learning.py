"""
Boltz-1 deep learning integration for RNA 3D structure prediction.

Uses the Boltz CLI for de novo structure prediction when template matching
gives poor results. Runs via subprocess since Boltz may be installed in
a different Python environment.
"""

import subprocess
import numpy as np
from pathlib import Path


def check_boltz_available() -> bool:
    """Check if Boltz CLI is installed and available."""
    try:
        result = subprocess.run(
            ["boltz", "predict", "--help"],
            capture_output=True, timeout=30
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def predict_with_boltz(target_id: str, sequence: str,
                       output_dir: str = "boltz_output",
                       n_samples: int = 5,
                       use_cpu: bool = True) -> list:
    """Predict RNA 3D structure using Boltz-1 CLI.

    Returns list of np.ndarray coordinate arrays (C1' positions),
    or empty list if prediction fails.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fasta_dir = output_dir / "input"
    fasta_dir.mkdir(exist_ok=True)
    fasta_path = fasta_dir / f"{target_id}.fasta"

    with open(fasta_path, "w") as f:
        f.write(f">{target_id}\n{sequence}\n")

    result_dir = output_dir / "results" / target_id
    result_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "boltz", "predict", str(fasta_path),
        "--out_dir", str(result_dir),
        "--diffusion_samples", str(n_samples),
        "--recycling_steps", "1",
        "--sampling_steps", "25",
    ]
    if use_cpu:
        cmd.extend(["--accelerator", "cpu"])

    try:
        print(f"    Running Boltz for {target_id} (len={len(sequence)})...")
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=1200,  # 20 minutes max per target
        )

        if result.returncode != 0:
            stderr_summary = result.stderr[:300] if result.stderr else "unknown error"
            print(f"    Boltz failed: {stderr_summary}")
            return []

        return _parse_boltz_output(result_dir, n_samples)

    except subprocess.TimeoutExpired:
        print(f"    Boltz timed out for {target_id}")
        return []
    except Exception as e:
        print(f"    Boltz error: {e}")
        return []


def _parse_boltz_output(output_dir: Path, n_samples: int) -> list:
    """Parse Boltz output to extract C1' atom coordinates."""
    predictions = []

    # Boltz outputs CIF files in a predictions subdirectory
    for pattern in ["**/*_model_*.cif", "**/*.cif", "**/*.pdb"]:
        files = sorted(output_dir.glob(pattern))
        for filepath in files[:n_samples]:
            coords = _extract_c1_prime(filepath)
            if coords is not None and len(coords) > 0:
                predictions.append(coords)

    return predictions


def _extract_c1_prime(filepath: Path) -> np.ndarray:
    """Extract C1' atom coordinates from PDB or mmCIF file."""
    coords = []
    suffix = filepath.suffix.lower()

    try:
        with open(filepath) as f:
            lines = f.readlines()

        if suffix == ".pdb":
            for line in lines:
                if line.startswith(("ATOM", "HETATM")):
                    atom_name = line[12:16].strip()
                    if atom_name == "C1'":
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        coords.append([x, y, z])

        elif suffix == ".cif":
            # Parse mmCIF ATOM_SITE records
            in_atom_site = False
            col_names = []
            atom_name_col = -1
            x_col = y_col = z_col = -1

            for line in lines:
                stripped = line.strip()
                if stripped.startswith("_atom_site."):
                    in_atom_site = True
                    col_name = stripped.split(".")[1].split()[0]
                    col_names.append(col_name)
                    idx = len(col_names) - 1
                    if col_name == "label_atom_id":
                        atom_name_col = idx
                    elif col_name == "Cartn_x":
                        x_col = idx
                    elif col_name == "Cartn_y":
                        y_col = idx
                    elif col_name == "Cartn_z":
                        z_col = idx
                elif in_atom_site and stripped.startswith(("ATOM", "HETATM")):
                    parts = stripped.split()
                    if (atom_name_col >= 0 and atom_name_col < len(parts) and
                            x_col >= 0 and y_col >= 0 and z_col >= 0):
                        atom_name = parts[atom_name_col].strip("'\"")
                        if atom_name == "C1'":
                            try:
                                x = float(parts[x_col])
                                y = float(parts[y_col])
                                z = float(parts[z_col])
                                coords.append([x, y, z])
                            except (ValueError, IndexError):
                                continue
                elif in_atom_site and not stripped.startswith("_") and stripped and stripped != "#":
                    if stripped.startswith("loop_") or stripped.startswith("data_"):
                        in_atom_site = False

    except Exception:
        return None

    if coords:
        return np.array(coords, dtype=np.float32)
    return None
