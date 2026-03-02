"""
Multi-chain RNA structure assembly.

Handles homo-oligomers (e.g., A:2, U:8) by predicting one chain and
replicating it with proper spatial arrangement, and hetero-oligomers
(e.g., B:1;A:1) by predicting each chain separately.
"""

import re
import numpy as np
import pandas as pd

from . import backend

xp = backend.xp


def parse_stoichiometry(stoich_str: str) -> list:
    """Parse stoichiometry string into chain definitions.

    Examples:
        "A:1" -> [("A", 1)]
        "A:2" -> [("A", 2)]
        "U:8" -> [("U", 8)]
        "B:1;A:1" -> [("B", 1), ("A", 1)]
        "AAA:2" -> [("AAA", 2)]

    Returns list of (chain_label, copy_count) tuples.
    """
    chains = []
    for part in stoich_str.split(";"):
        part = part.strip()
        if ":" in part:
            label, count = part.rsplit(":", 1)
            chains.append((label, int(count)))
        else:
            chains.append((part, 1))
    return chains


def parse_all_sequences_fasta(all_sequences_str: str) -> list:
    """Parse the all_sequences FASTA column into per-chain sequences.

    The all_sequences column contains FASTA entries like:
        >9JGM_1|Chains A[auth C], C[auth D]|...\nSEQUENCE

    Returns list of (chain_id, sequence) tuples.
    """
    if pd.isna(all_sequences_str) or not all_sequences_str.strip():
        return []

    entries = []
    current_header = None
    current_seq_parts = []

    for line in all_sequences_str.strip().split("\n"):
        line = line.strip()
        if line.startswith(">"):
            if current_header is not None and current_seq_parts:
                entries.append((current_header, "".join(current_seq_parts)))
            current_header = line[1:]
            current_seq_parts = []
        elif line:
            current_seq_parts.append(line)

    if current_header is not None and current_seq_parts:
        entries.append((current_header, "".join(current_seq_parts)))

    return entries


def get_chain_sequences(full_sequence: str, stoichiometry: list,
                        all_sequences_str: str = None) -> list:
    """Split a concatenated sequence into per-chain sequences.

    Uses the all_sequences FASTA column if available for accurate
    per-chain sequence boundaries. Falls back to length-based splitting.

    Returns list of (chain_label, copy_index, subsequence) tuples.
    """
    # Try using all_sequences FASTA first for accurate chain boundaries
    if all_sequences_str and not pd.isna(all_sequences_str):
        fasta_entries = parse_all_sequences_fasta(all_sequences_str)
        if fasta_entries:
            result = []
            for i, (header, seq) in enumerate(fasta_entries):
                # Extract chain ID from header like "9JGM_1|Chains A..."
                chain_id = header.split("|")[0] if "|" in header else f"chain_{i}"
                result.append((chain_id, i, seq))
            return result

    total_copies = sum(count for _, count in stoichiometry)
    if total_copies <= 0:
        return [("A", 0, full_sequence)]

    unique_chains = set(label for label, _ in stoichiometry)

    if len(unique_chains) == 1:
        label, n_copies = stoichiometry[0]
        chain_len = len(full_sequence) // n_copies
        result = []
        for i in range(n_copies):
            start = i * chain_len
            end = start + chain_len if i < n_copies - 1 else len(full_sequence)
            result.append((label, i, full_sequence[start:end]))
        return result

    chain_len = len(full_sequence) // total_copies
    result = []
    pos = 0
    for label, count in stoichiometry:
        for i in range(count):
            end = pos + chain_len if pos + chain_len < len(full_sequence) else len(full_sequence)
            result.append((label, i, full_sequence[pos:end]))
            pos = end
    return result


def generate_symmetric_copies(single_chain_coords, n_copies: int,
                               arrangement: str = "circular"):
    """Generate symmetrically arranged copies. Coords can be NumPy or CuPy. Returns same backend."""
    L = len(single_chain_coords)
    if n_copies == 1:
        return single_chain_coords.copy()

    centroid = single_chain_coords.mean(axis=0)
    centered = single_chain_coords - centroid

    max_extent = float(xp.max(xp.linalg.norm(centered, axis=1)))
    separation = max_extent * 2.5

    all_coords = xp.zeros((n_copies * L, 3), dtype=xp.float32)

    if arrangement == "circular" and n_copies > 2:
        radius = separation / (2.0 * float(xp.sin(3.141592653589793 / n_copies)))
        for i in range(n_copies):
            angle = 2.0 * 3.141592653589793 * i / n_copies
            rotation = _rotation_matrix_z(angle)
            offset = xp.array([radius * xp.cos(angle),
                               radius * xp.sin(angle), 0.0], dtype=xp.float32)
            rotated = centered @ rotation.T
            all_coords[i * L:(i + 1) * L] = rotated + centroid + offset
    else:
        z_extent = float(centered[:, 2].max() - centered[:, 2].min())
        stack_distance = max(z_extent * 1.3, separation)
        total_height = (n_copies - 1) * stack_distance
        for i in range(n_copies):
            offset = xp.array([0.0, 0.0, i * stack_distance - total_height / 2], dtype=xp.float32)
            angle = 2.0 * 3.141592653589793 * i / n_copies
            rotation = _rotation_matrix_z(angle)
            rotated = centered @ rotation.T
            all_coords[i * L:(i + 1) * L] = rotated + centroid + offset

    return all_coords


def _rotation_matrix_z(angle: float):
    c, s = float(xp.cos(angle)), float(xp.sin(angle))
    return xp.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=xp.float32)


def assemble_multimer(chain_coords_list: list, stoichiometry: list,
                      full_length: int):
    """Assemble multi-chain structure. Returns (full_length, 3) on same backend as inputs."""
    unique_chains = set(label for label, _ in stoichiometry)

    if len(unique_chains) == 1 and len(chain_coords_list) >= 1:
        _, n_copies = stoichiometry[0]
        single_chain = chain_coords_list[0]
        if n_copies > 1:
            result = generate_symmetric_copies(single_chain, n_copies)
            if len(result) >= full_length:
                return result[:full_length]
            padded = xp.zeros((full_length, 3), dtype=xp.float32)
            padded[:len(result)] = result
            return padded
        else:
            if len(single_chain) >= full_length:
                return single_chain[:full_length]
            padded = xp.zeros((full_length, 3), dtype=xp.float32)
            padded[:len(single_chain)] = single_chain
            return padded

    concatenated = xp.concatenate(chain_coords_list, axis=0)
    if len(concatenated) >= full_length:
        return concatenated[:full_length]
    padded = xp.zeros((full_length, 3), dtype=xp.float32)
    padded[:len(concatenated)] = concatenated
    return padded
