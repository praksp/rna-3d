#!/usr/bin/env python3
"""
Submit the generated submission.csv to the Kaggle competition.
Tries the Kaggle API first; if that fails (e.g. 400), prints manual steps.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Submit solution to Kaggle")
    parser.add_argument("--file", type=str, default="output/submission.csv",
                        help="Path to submission CSV")
    parser.add_argument("--message", "-m", type=str, default="Template-based prediction",
                        help="Submission description")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"Error: submission file not found: {path}")
        sys.exit(1)

    print(f"Submitting {path} to stanford-rna-3d-folding-2...")
    result = subprocess.run(
        [
            "kaggle", "competitions", "submit",
            "-c", "stanford-rna-3d-folding-2",
            "-f", str(path.resolve()),
            "-m", args.message,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        print(result.stdout or "Submission accepted.")
        return

    print(result.stdout or "")
    print(result.stderr or "")
    if "400" in result.stderr or "Bad Request" in result.stderr:
        print("\n" + "=" * 60)
        print("API returned 400. Submit manually on Kaggle:")
        print("=" * 60)
        print("1. Open: https://www.kaggle.com/competitions/stanford-rna-3d-folding-2/submit")
        print("2. Accept the rules if prompted.")
        print("3. Upload file:", path.resolve())
        print("4. Add a short description and click Submit.")
        print("=" * 60)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
