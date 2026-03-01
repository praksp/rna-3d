# Stanford RNA 3D Folding Part 2 - Kaggle Competition Solution

Predicts 3D coordinates (C1' atom positions) of RNA molecules from their sequences.
Evaluated on **TM-score** (0-1, higher is better) with best-of-5 diverse predictions.

**Competition**: https://www.kaggle.com/competitions/stanford-rna-3d-folding-2

## Approach

**Advanced template-based prediction** with multi-strategy diversity:

1. **Two-stage template search**: Fast k-mer prefilter (5-mers) narrows ~5,700 training structures to top 50 candidates, then proper Needleman-Wunsch/Smith-Waterman alignment ranks them
2. **Alignment-guided coordinate transfer**: Residue-level mapping from alignment ensures correct structural correspondence, with smoothing to eliminate mapping artifacts
3. **Multi-chain assembly**: Parses FASTA per-chain sequences; homo-oligomers are built by predicting one chain and replicating with symmetric arrangement
4. **Diverse predictions**: 5 predictions from different sources -- distinct templates, alternate conformational copies, and perturbations
5. **Optional Boltz-1 integration**: Deep learning de novo prediction for targets without good templates

## Setup

```bash
pip install -r requirements.txt

# Set up Kaggle credentials
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Download competition data
export KAGGLE_API_TOKEN=<your-token>
kaggle competitions download -c stanford-rna-3d-folding-2 -f sample_submission.csv -p data
kaggle competitions download -c stanford-rna-3d-folding-2 -f train_sequences.csv -p data
kaggle competitions download -c stanford-rna-3d-folding-2 -f test_sequences.csv -p data
kaggle competitions download -c stanford-rna-3d-folding-2 -f train_labels.csv -p data
```

## Running

```bash
# Template-based prediction (~60 seconds)
python run.py --data-dir data --output output/submission.csv

# With Boltz-1 deep learning enhancement (slower, requires `pip install boltz`)
python run.py --data-dir data --output output/submission.csv --use-boltz
```

## Project Structure

```
.
├── run.py                      # Main pipeline entry point
├── requirements.txt
├── data/                       # Competition data (downloaded)
│   ├── train_sequences.csv     # 5,716 training RNA sequences
│   ├── train_labels.csv        # 3D C1' coordinates + multiple copies
│   ├── test_sequences.csv      # 28 test targets
│   └── sample_submission.csv
├── src/
│   ├── data_loader.py          # Data loading with multi-copy support
│   ├── alignment.py            # Needleman-Wunsch & Smith-Waterman alignment
│   ├── superposition.py        # Kabsch algorithm for structural superposition
│   ├── multichain.py           # Multi-chain parsing and symmetric assembly
│   ├── template_matcher.py     # Two-stage template search + coordinate transfer
│   ├── geometry.py             # A-form helix generation, coordinate utilities
│   ├── deep_learning.py        # Optional Boltz-1 integration
│   └── submission.py           # Submission CSV generation and validation
└── output/
    └── submission.csv          # Generated predictions
```

## Data Format

**Input**: `target_id, sequence, temporal_cutoff, description, stoichiometry, all_sequences, ligand_ids, ligand_SMILES`

**Output**: `ID, resname, resid, x_1, y_1, z_1, ..., x_5, y_5, z_5` (5 predictions of C1' atom xyz per residue)
