"""
Advanced template-based RNA structure prediction.

- Two-stage: k-mer prefilter → sequence alignment ranking
- Alignment-guided transfer + Kabsch superposition (backend: NumPy/CuPy)
- Temporal cutoff filtering; length-similarity in scoring
- Predictions ordered by quality (best first for best-of-5)
"""

import numpy as np
from tqdm import tqdm

from . import config
from . import backend
from .alignment import compute_alignment_score, get_residue_mapping
from .superposition import kabsch_rmsd, apply_superposition
from .multichain import (
    parse_stoichiometry, get_chain_sequences,
    generate_symmetric_copies, assemble_multimer,
)
from .geometry import generate_aform_helix, _sanitize_coords

xp = backend.xp


def _parse_cutoff(date_str) -> "datetime":
    """Parse temporal_cutoff string to comparable value."""
    if hasattr(date_str, "year"):
        return date_str
    try:
        from datetime import datetime
        return datetime.strptime(str(date_str).strip()[:10], "%Y-%m-%d")
    except Exception:
        return None


def kmer_prefilter(test_seq: str, train_sequences: dict,
                   train_structures: dict,
                   allowed_ids: set = None,
                   k: int = None, top_n: int = None,
                   length_ratio_min: float = None) -> list:
    """Fast k-mer prefilter; optionally restrict to allowed_ids (temporal cutoff)."""
    k = k or config.KMER_SIZE
    top_n = top_n or config.PREFILTER_TOP_N
    length_ratio_min = length_ratio_min or config.LENGTH_RATIO_MIN

    test_len = len(test_seq)
    test_kmers = set(test_seq[i:i + k] for i in range(len(test_seq) - k + 1))

    if not test_kmers:
        cands = list(train_structures.keys())
        if allowed_ids is not None:
            cands = [c for c in cands if c in allowed_ids]
        return cands[:top_n]

    scores = []
    for train_id, train_seq in train_sequences.items():
        if train_id not in train_structures:
            continue
        if allowed_ids is not None and train_id not in allowed_ids:
            continue

        train_len = len(train_seq)
        length_ratio = min(test_len, train_len) / max(test_len, train_len)
        if length_ratio < length_ratio_min:
            continue

        train_kmers = set(train_seq[i:i + k] for i in range(len(train_seq) - k + 1))
        if not train_kmers:
            continue

        inter = len(test_kmers & train_kmers)
        union = len(test_kmers | train_kmers)
        jaccard = inter / union if union > 0 else 0.0
        score = jaccard * 0.6 + length_ratio * 0.4
        scores.append((train_id, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [tid for tid, _ in scores[:top_n]]


def rank_templates_by_alignment(test_seq: str, candidate_ids: list,
                                 train_sequences: dict,
                                 max_align: int = None) -> list:
    """Rank candidates by alignment; add length-similarity and coverage."""
    max_align = max_align or config.MAX_ALIGN_CANDIDATES
    test_len = len(test_seq)
    results = []

    for train_id in candidate_ids[:max_align]:
        train_seq = train_sequences[train_id]
        train_len = len(train_seq)
        test_sub = test_seq[:2000]
        train_sub = train_seq[:2000]

        score, aligned_pairs = compute_alignment_score(
            test_sub, train_sub, mode="auto",
            length_ratio_global_threshold=config.LENGTH_RATIO_GLOBAL_THRESHOLD,
        )

        coverage = 0.0
        if aligned_pairs:
            target_covered = len(set(p[0] for p in aligned_pairs if p[0] >= 0))
            coverage = target_covered / len(test_sub)

        length_sim = min(test_len, train_len) / max(test_len, train_len)
        combined = (
            score * config.ALIGNMENT_SCORE_WEIGHT
            + coverage * config.COVERAGE_WEIGHT
            + length_sim * config.LENGTH_SIMILARITY_WEIGHT
        )
        results.append((train_id, combined, aligned_pairs))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def transfer_coords_with_alignment(template_coords, aligned_pairs: list,
                                    target_len: int, template_len: int,
                                    apply_kabsch: bool = True,
                                    extrapolation_cap_ratio: float = None):
    """Transfer coords via alignment; optionally apply Kabsch; cap extrapolation with A-form."""
    extrapolation_cap_ratio = extrapolation_cap_ratio or config.EXTRAPOLATION_CAP_RATIO
    mapping = get_residue_mapping(aligned_pairs, target_len, template_len)

    result = xp.zeros((target_len, 3), dtype=xp.float32)
    cap_len = int(template_len * extrapolation_cap_ratio) if template_len > 0 else target_len

    for i in range(target_len):
        j = int(mapping[i])
        if 0 <= j < len(template_coords):
            result[i] = template_coords[j]
        elif len(template_coords) > 0:
            result[i] = template_coords[min(j, len(template_coords) - 1)]

    if target_len > cap_len and cap_len >= 3:
        tail_len = target_len - cap_len
        p0, p1 = result[cap_len - 2], result[cap_len - 1]
        direction = p1 - p0
        norm = float(xp.linalg.norm(direction))
        if norm < 1e-6:
            direction = xp.array([1.0, 0.0, 0.0], dtype=xp.float32)
        else:
            direction = direction / norm
        aform = generate_aform_helix(tail_len, offset=0)
        aform[:, 2] *= (5.9 / 2.81)
        aform += result[cap_len - 1] - aform[0]
        result[cap_len:] = aform

    if target_len > config.SMOOTHING_WINDOW:
        w = config.SMOOTHING_WINDOW
        half = w // 2
        smoothed = result.copy()
        for i in range(half, target_len - half):
            smoothed[i] = result[i - half : i - half + w].mean(axis=0)
        result = smoothed

    if apply_kabsch and aligned_pairs and len(aligned_pairs) >= 3:
        t_indices = []
        s_indices = []
        for ti, si in aligned_pairs:
            if 0 <= ti < target_len and 0 <= si < len(template_coords):
                t_indices.append(ti)
                s_indices.append(si)
        if len(t_indices) >= 3:
            P = result[xp.array(t_indices)]
            Q = template_coords[xp.array(s_indices)]
            if bool(xp.isfinite(P).all() and xp.isfinite(Q).all()):
                R, t, _ = kabsch_rmsd(P, Q)
                result = apply_superposition(result, R, t)

    return result


def predict_single_chain(chain_seq: str, target_len: int,
                          train_seq_map: dict, train_structures: dict,
                          allowed_template_ids: set = None,
                          n_predictions: int = 5,
                          run_log: list = None) -> tuple:
    """Predict single chain; return (list of coords ordered by quality, run_log_entries)."""
    log_entries = []
    if run_log is not None:
        run_log = run_log  # append to this list

    candidates = kmer_prefilter(
        chain_seq, train_seq_map, train_structures,
        allowed_ids=allowed_template_ids,
    )
    candidates = [c for c in candidates if c in train_structures]

    if not candidates:
        helix = generate_aform_helix(target_len)
        for _ in range(n_predictions):
            log_entries.append({"source": "aform", "template_id": None, "score": 0.0})
        return [helix] * n_predictions, log_entries

    ranked = rank_templates_by_alignment(chain_seq, candidates, train_seq_map)

    if not ranked or ranked[0][1] < config.MIN_ALIGNMENT_SCORE_FOR_TEMPLATE:
        helix = generate_aform_helix(target_len)
        for _ in range(n_predictions):
            log_entries.append({"source": "aform", "template_id": None, "score": 0.0})
        return [helix] * n_predictions, log_entries

    predictions = []
    used_sources = set()

    for template_id, score, aligned_pairs in ranked:
        if len(predictions) >= n_predictions:
            break
        if template_id in used_sources:
            continue
        used_sources.add(template_id)

        template_data = train_structures[template_id]
        template_coords = template_data["coords"]

        if aligned_pairs:
            coords = transfer_coords_with_alignment(
                template_coords, aligned_pairs, target_len, len(template_coords),
                apply_kabsch=True,
            )
        else:
            coords = _fallback_transfer(template_coords, target_len)

        coords = _sanitize_coords(coords)
        predictions.append(coords)
        log_entries.append({"source": "template", "template_id": template_id, "score": float(score)})

    # Alternate copies of best template
    if len(predictions) < n_predictions and ranked:
        best_id, best_score, best_aligned = ranked[0][0], ranked[0][1], ranked[0][2]
        best_data = train_structures[best_id]
        if best_data["n_copies"] > 1:
            for copy_id, copy_coords in best_data["all_copies"].items():
                if len(predictions) >= n_predictions:
                    break
                key = f"{best_id}_copy{copy_id}"
                if key in used_sources:
                    continue
                used_sources.add(key)
                if best_aligned:
                    coords = transfer_coords_with_alignment(
                        copy_coords, best_aligned, target_len, len(copy_coords),
                        apply_kabsch=True,
                    )
                else:
                    coords = _fallback_transfer(copy_coords, target_len)
                coords = _sanitize_coords(coords)
                predictions.append(coords)
                log_entries.append({"source": "copy", "template_id": best_id, "score": float(best_score)})

    try:
        rng = xp.random.default_rng(config.SEED)
    except AttributeError:
        rng = np.random.default_rng(config.SEED)
    while len(predictions) < n_predictions:
        base = predictions[0].copy()
        noise = rng.normal(0, config.PERTURBATION_SIGMA, base.shape)
        if hasattr(noise, "get"):
            noise = noise
        else:
            noise = xp.asarray(noise)
        predictions.append(_sanitize_coords(base + noise))
        log_entries.append({"source": "perturb", "template_id": None, "score": 0.0})

    return predictions, log_entries


def _fallback_transfer(template_coords, target_len: int):
    """Simple coordinate transfer when alignment is not available."""
    template_len = len(template_coords)
    if template_len == target_len:
        return template_coords.copy()
    if template_len > target_len:
        return template_coords[:target_len].copy()
    # interp is numpy-only; do on host then return xp
    t_np = backend.asnumpy(template_coords)
    result_np = np.zeros((target_len, 3), dtype=np.float32)
    for dim in range(3):
        result_np[:, dim] = np.interp(
            np.linspace(0, 1, target_len),
            np.linspace(0, 1, template_len),
            t_np[:, dim],
        )
    return xp.asarray(result_np)


def _allowed_templates_for_cutoff(train_sequences_df, test_cutoff_str) -> set:
    """Set of train target_ids with temporal_cutoff strictly before test cutoff."""
    test_d = _parse_cutoff(test_cutoff_str)
    if test_d is None:
        return None
    allowed = set()
    for _, row in train_sequences_df.iterrows():
        t = _parse_cutoff(row.get("temporal_cutoff"))
        if t is not None and t < test_d:
            allowed.add(row["target_id"])
    return allowed if len(allowed) > 0 else None


def predict_one_target(row, train_seq_map, train_structures, submission_targets,
                       n_predictions: int, train_sequences_df) -> tuple:
    """Predict a single target. Returns (target_id, predictions_list, run_log_entry) or (None, None, None) if skip."""
    target_id = row["target_id"]
    if target_id not in submission_targets:
        return None, None, None
    test_seq = row["sequence"]
    stoich_str = row["stoichiometry"]
    all_sequences_str = row.get("all_sequences", None)
    test_cutoff = row.get("temporal_cutoff", None)
    n_residues = len(submission_targets[target_id])
    stoich = parse_stoichiometry(stoich_str)
    total_copies = sum(count for _, count in stoich)
    unique_chain_labels = set(label for label, _ in stoich)
    allowed = _allowed_templates_for_cutoff(train_sequences_df, test_cutoff)
    chains = get_chain_sequences(test_seq, stoich, all_sequences_str)

    if total_copies > 1 and len(unique_chain_labels) == 1:
        _, n_copies = stoich[0]
        chain_seq = chains[0][2]
        chain_len = len(chain_seq)
        chain_preds, log_entries = predict_single_chain(
            chain_seq, chain_len, train_seq_map, train_structures,
            allowed_template_ids=allowed, n_predictions=n_predictions,
        )
        full_preds = []
        for pred in chain_preds:
            assembled = generate_symmetric_copies(pred, n_copies)
            if len(assembled) >= n_residues:
                assembled = assembled[:n_residues]
            else:
                padded = xp.zeros((n_residues, 3), dtype=xp.float32)
                padded[:len(assembled)] = assembled
                assembled = padded
            full_preds.append(backend.asnumpy(assembled))
        log_entry = {"type": "homo-oligomer", "n_copies": n_copies, "predictions": log_entries,
                     "templates": [e.get("template_id") for e in log_entries if e.get("template_id")],
                     "scores": [e.get("score", 0) for e in log_entries],
                     "best_score": log_entries[0]["score"] if log_entries else 0.0,
                     "n_residues": n_residues, "chain_len": chain_len}
        return target_id, full_preds, log_entry

    if total_copies > 1 and len(unique_chain_labels) > 1:
        all_chain_preds = []
        for label, i, chain_seq in chains:
            chain_len = len(chain_seq)
            chain_pred, _ = predict_single_chain(
                chain_seq, chain_len, train_seq_map, train_structures,
                allowed_template_ids=allowed, n_predictions=n_predictions,
            )
            all_chain_preds.append(chain_pred)
        full_preds = []
        for pred_idx in range(n_predictions):
            chain_coords_list = [c[pred_idx] for c in all_chain_preds]
            assembled = assemble_multimer(chain_coords_list, stoich, n_residues)
            full_preds.append(backend.asnumpy(assembled))
        log_entry = {"type": "hetero-oligomer", "n_residues": n_residues,
                     "templates": [], "scores": [], "best_score": 0.0}
        return target_id, full_preds, log_entry

    preds, log_entries = predict_single_chain(
        test_seq, n_residues, train_seq_map, train_structures,
        allowed_template_ids=allowed, n_predictions=n_predictions,
    )
    log_entry = {"type": "single", "predictions": log_entries,
                 "templates": [e.get("template_id") for e in log_entries if e.get("template_id")],
                 "scores": [e.get("score", 0) for e in log_entries],
                 "best_score": log_entries[0]["score"] if log_entries else 0.0,
                 "n_residues": n_residues}
    return target_id, [backend.asnumpy(p) for p in preds], log_entry


def predict_with_templates(test_sequences_df, train_sequences_df,
                           train_structures: dict, submission_targets: dict,
                           n_predictions: int = 5, workers: int = 1) -> tuple:
    """Generate predictions; return (predictions_dict, run_log_dict). Use workers>1 for CPU parallel."""
    train_seq_map = dict(zip(
        train_sequences_df["target_id"],
        train_sequences_df["sequence"],
    ))
    predictions = {}
    run_log = {}

    if workers > 1 and backend.device == "cpu":
        from concurrent.futures import ProcessPoolExecutor, as_completed
        rows = [row for _, row in test_sequences_df.iterrows() if row["target_id"] in submission_targets]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    predict_one_target, row, train_seq_map, train_structures,
                    submission_targets, n_predictions, train_sequences_df,
                ): row for row in rows
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Predicting"):
                target_id, preds, log_entry = fut.result()
                if target_id is not None:
                    predictions[target_id] = preds
                    run_log[target_id] = log_entry
        return predictions, run_log

    for _, row in tqdm(test_sequences_df.iterrows(), total=len(test_sequences_df),
                       desc="Predicting"):
        target_id = row["target_id"]
        test_seq = row["sequence"]
        stoich_str = row["stoichiometry"]
        all_sequences_str = row.get("all_sequences", None)
        test_cutoff = row.get("temporal_cutoff", None)

        if target_id not in submission_targets:
            continue

        n_residues = len(submission_targets[target_id])
        stoich = parse_stoichiometry(stoich_str)
        total_copies = sum(count for _, count in stoich)
        unique_chain_labels = set(label for label, _ in stoich)

        allowed = _allowed_templates_for_cutoff(train_sequences_df, test_cutoff)

        chains = get_chain_sequences(test_seq, stoich, all_sequences_str)

        if total_copies > 1 and len(unique_chain_labels) == 1:
            _, n_copies = stoich[0]
            chain_seq = chains[0][2]
            chain_len = len(chain_seq)

            chain_preds, log_entries = predict_single_chain(
                chain_seq, chain_len, train_seq_map, train_structures,
                allowed_template_ids=allowed,
                n_predictions=n_predictions,
            )
            run_log[target_id] = {"type": "homo-oligomer", "n_copies": n_copies, "predictions": log_entries}

            full_preds = []
            for pred in chain_preds:
                assembled = generate_symmetric_copies(pred, n_copies)
                if len(assembled) >= n_residues:
                    assembled = assembled[:n_residues]
                else:
                    padded = xp.zeros((n_residues, 3), dtype=xp.float32)
                    padded[:len(assembled)] = assembled
                    assembled = padded
                full_preds.append(backend.asnumpy(assembled))
            run_log[target_id]["templates"] = [e.get("template_id") for e in log_entries if e.get("template_id")]
            run_log[target_id]["scores"] = [e.get("score", 0) for e in log_entries]
            run_log[target_id]["best_score"] = log_entries[0]["score"] if log_entries else 0.0
            run_log[target_id]["n_residues"] = n_residues
            run_log[target_id]["chain_len"] = chain_len
            predictions[target_id] = full_preds

        elif total_copies > 1 and len(unique_chain_labels) > 1:
            all_chain_preds = []
            for label, i, chain_seq in chains:
                chain_len = len(chain_seq)
                chain_pred, log_entries = predict_single_chain(
                    chain_seq, chain_len, train_seq_map, train_structures,
                    allowed_template_ids=allowed,
                    n_predictions=n_predictions,
                )
                all_chain_preds.append(chain_pred)

            full_preds = []
            for pred_idx in range(n_predictions):
                chain_coords_list = [c[pred_idx] for c in all_chain_preds]
                assembled = assemble_multimer(chain_coords_list, stoich, n_residues)
                full_preds.append(backend.asnumpy(assembled))

            run_log[target_id] = {"type": "hetero-oligomer", "n_residues": n_residues}
            run_log[target_id]["templates"] = []
            run_log[target_id]["scores"] = []
            run_log[target_id]["best_score"] = 0.0
            predictions[target_id] = full_preds

        else:
            preds, log_entries = predict_single_chain(
                test_seq, n_residues, train_seq_map, train_structures,
                allowed_template_ids=allowed,
                n_predictions=n_predictions,
            )
            run_log[target_id] = {"type": "single", "predictions": log_entries}
            run_log[target_id]["templates"] = [e.get("template_id") for e in log_entries if e.get("template_id")]
            run_log[target_id]["scores"] = [e.get("score", 0) for e in log_entries]
            run_log[target_id]["best_score"] = log_entries[0]["score"] if log_entries else 0.0
            run_log[target_id]["n_residues"] = n_residues
            predictions[target_id] = [backend.asnumpy(p) for p in preds]

    return predictions, run_log
