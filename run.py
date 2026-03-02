#!/usr/bin/env python3
"""
Stanford RNA 3D Folding Part 2 - Advanced Prediction Pipeline.

- Runs on local (Mac/Windows/Linux) or Kaggle; auto-detects data/output paths.
- GPU (CuPy) when available (e.g. Kaggle GPU); CPU (NumPy) on Mac.
- Optional parallel prediction (--workers) on CPU for faster runs.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Force CPU if requested, before backend is loaded
if "--no-gpu" in sys.argv:
    os.environ["RNA_USE_GPU"] = "0"

import numpy as np

from src import config
from src import backend
from src.data_loader import (
    load_train_sequences,
    load_test_sequences,
    load_sample_submission,
    load_train_labels,
    build_train_structure_lookup,
    parse_submission_targets,
)
from src.template_matcher import predict_with_templates
from src.submission import create_submission, validate_submission


def main():
    parser = argparse.ArgumentParser(description="RNA 3D Folding Prediction Pipeline")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Data directory (default: auto-detect Kaggle or 'data')")
    parser.add_argument("--output", type=str, default=None,
                        help="Submission CSV path (default: Kaggle working or 'output/submission.csv')")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Log directory (default: same as output dir)")
    parser.add_argument("--n-predictions", type=int, default=5,
                        help="Number of diverse predictions per target (default: 5)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers for prediction (CPU only; default 1, use 4+ for speed)")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Force CPU even if CuPy is available")
    parser.add_argument("--use-boltz", action="store_true",
                        help="Use Boltz-1 for targets with poor template matches")
    parser.add_argument("--boltz-threshold", type=float, default=0.35,
                        help="Alignment score below which Boltz is used (default: 0.35)")
    args = parser.parse_args()

    np.random.seed(config.SEED)

    data_dir = Path(args.data_dir) if args.data_dir else backend.get_data_dir()
    out_path = Path(args.output) if args.output else (backend.get_output_dir() / "submission.csv")
    log_dir = Path(args.log_dir) if args.log_dir else out_path.parent
    log_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    _print_header()
    _print_config(workers=args.workers)

    # --- Load data ---
    print("\n[1/6] Loading training sequences...")
    train_seq_df = load_train_sequences(data_dir)
    print(f"      Loaded {len(train_seq_df)} sequences")

    print("\n[2/6] Loading training labels (3D coordinates)...")
    train_labels_df = load_train_labels(data_dir)
    print(f"      {len(train_labels_df)} coordinate rows")

    print("\n[3/6] Building structure lookup...")
    train_structures = build_train_structure_lookup(train_labels_df)
    n_multi = sum(1 for v in train_structures.values() if v["n_copies"] > 1)
    print(f"      {len(train_structures)} structures ({n_multi} with multiple copies)")
    del train_labels_df

    print("\n[4/6] Loading test data...")
    test_seq_df = load_test_sequences(data_dir)
    sample_sub_df = load_sample_submission(data_dir)
    submission_targets = parse_submission_targets(sample_sub_df)
    print(f"      {len(test_seq_df)} test sequences, {len(submission_targets)} targets")
    print(f"      Total residues to predict: {len(sample_sub_df)}")

    # --- Predict ---
    workers = args.workers if backend.device == "cpu" else 1
    if workers > 1:
        print(f"\n[5/6] Generating predictions ({args.n_predictions} per target, {workers} workers)...")
    else:
        print(f"\n[5/6] Generating predictions ({args.n_predictions} per target)...")
    predictions, run_log = predict_with_templates(
        test_seq_df,
        train_seq_df,
        train_structures,
        submission_targets,
        n_predictions=args.n_predictions,
        workers=workers,
    )

    if args.use_boltz:
        _enhance_with_boltz(
            predictions, test_seq_df, submission_targets,
            args.boltz_threshold, args.n_predictions,
        )

    # --- Summary table ---
    _print_summary_table(run_log)

    # --- Write run log ---
    log_path = log_dir / "run_log.json"
    _write_run_log(log_path, run_log)
    print(f"\n      Run log: {log_path}")

    # --- Create submission ---
    print("\n[6/6] Creating submission file...")
    sub_df = create_submission(predictions, submission_targets, args.output)

    print("\nValidating submission...")
    validate_submission(args.output, str(data_dir / "sample_submission.csv"))

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"  Done in {elapsed:.1f}s  |  Submission: {out_path}")
    print("=" * 60)

    return sub_df


def _print_header():
    print()
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║  Stanford RNA 3D Folding Part 2 — Prediction Pipeline        ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")
    print()


def _print_config(workers=1):
    print("  Strategy:")
    print("    • Two-stage template matching (k-mer → alignment)")
    print("    • Kabsch superposition after coordinate transfer")
    print("    • Temporal cutoff filter; length-similarity in scoring")
    print("    • Predictions ordered by quality (best first)")
    print(f"    • Backend: {backend.device}  |  Seed: {config.SEED}")
    if workers > 1 and backend.device == "cpu":
        print(f"    • Parallel: {workers} workers")


def _print_summary_table(run_log: dict):
    """Print a compact table of target_id, type, n_residues, best_score, top template."""
    if not run_log:
        return
    print("\n  ┌──────────┬─────────────────┬──────────┬──────────┬─────────────────────┐")
    print("  │ Target   │ Type            │ Residues │ Best Scr │ Top template         │")
    print("  ├──────────┼─────────────────┼──────────┼──────────┼─────────────────────┤")
    for tid in sorted(run_log.keys()):
        r = run_log[tid]
        tpe = r.get("type", "single")[:15]
        nres = r.get("n_residues", 0)
        score = r.get("best_score", 0)
        templates = r.get("templates") or []
        top = (templates[0] if templates else "—")[:19]
        print(f"  │ {tid:<8} │ {tpe:<15} │ {nres:>8} │ {score:>8.3f} │ {top:<19} │")
    print("  └──────────┴─────────────────┴──────────┴──────────┴─────────────────────┘")


def _write_run_log(log_path: Path, run_log: dict):
    """Write run log as JSON (template IDs and scores per target)."""
    # Make values JSON-serializable
    out = {}
    for tid, data in run_log.items():
        out[tid] = {k: v for k, v in data.items() if k != "predictions"}
        if "scores" in data:
            out[tid]["scores"] = [float(x) for x in data["scores"]]
        if "best_score" in data:
            out[tid]["best_score"] = float(data["best_score"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(out, f, indent=2)


def _enhance_with_boltz(predictions, test_seq_df, submission_targets,
                        threshold, n_predictions):
    """Enhance predictions with Boltz-1 for targets with poor template scores."""
    from src.deep_learning import check_boltz_available, predict_with_boltz

    if not check_boltz_available():
        print("\n      Boltz-1 not available, skipping.")
        return

    print("\n      Boltz-1: enhancing low-confidence targets...")
    for _, row in test_seq_df.iterrows():
        target_id = row["target_id"]
        if target_id not in submission_targets or len(row["sequence"]) > 300:
            continue
        current = predictions.get(target_id, [])
        if not current:
            continue
        boltz_preds = predict_with_boltz(
            target_id, row["sequence"],
            output_dir="boltz_output", n_samples=min(3, n_predictions), use_cpu=True,
        )
        if boltz_preds:
            n_residues = len(submission_targets[target_id])
            for bp in boltz_preds:
                bp = bp[:n_residues] if len(bp) >= n_residues else np.pad(bp, ((0, n_residues - len(bp)), (0, 0)))
                if len(current) >= n_predictions:
                    current[-1] = bp
                else:
                    current.append(bp)
            predictions[target_id] = current
            print(f"        Enhanced {target_id}")


if __name__ == "__main__":
    main()
