"""Central config: seed and hyperparameters for reproducibility and tuning."""

import os

# Reproducibility
SEED = int(os.environ.get("RNA_FOLDING_SEED", "42"))

# Template prefilter (stage 1)
KMER_SIZE = 5
PREFILTER_TOP_N = 50
LENGTH_RATIO_MIN = 0.3  # skip templates with length ratio below this

# Alignment (stage 2)
MAX_ALIGN_CANDIDATES = 30  # run full alignment on top N from prefilter
LENGTH_RATIO_GLOBAL_THRESHOLD = 0.85  # use global alignment when ratio above this
ALIGNMENT_SCORE_WEIGHT = 0.65
COVERAGE_WEIGHT = 0.25
LENGTH_SIMILARITY_WEIGHT = 0.10  # favor similar-length templates

# Coordinate transfer
EXTRAPOLATION_CAP_RATIO = 1.20  # cap extrapolation at 20% beyond template; use A-form beyond
SMOOTHING_WINDOW = 5  # moving average window for transferred coords

# Diversity
MIN_ALIGNMENT_SCORE_FOR_TEMPLATE = 0.05  # below this use A-form helix
PERTURBATION_SIGMA = 0.3  # Gaussian noise for last-resort diversity

# Output precision (Kaggle uses full-precision for scoring; 6 decimals for coordinates)
COORD_DECIMALS = 6
COORD_DTYPE = "float64"  # full precision in pipeline, then round in CSV
