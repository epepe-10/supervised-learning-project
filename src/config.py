"""Project-wide constants: paths, random seed, split sizes and column lists.

Every other script imports its settings from here, so a value only has to be
changed in one place.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"                 # OpenML download cache (not versioned)
OUTPUT_DIR = ROOT_DIR / "outputs"            # everything the scripts produce
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"

# ---------------------------------------------------------------------------
# Reproducibility and validation protocol (S1)
# ---------------------------------------------------------------------------
RANDOM_STATE = 42    # same seed everywhere -> same split, same folds, same results
TEST_SIZE = 0.20     # 80% development set / 20% held-out test set
N_FOLDS = 5          # 5-fold StratifiedKFold on the development set

# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------
TARGET_NAME = "income_above_50k"
POSITIVE_LABEL = ">50K"   # original label that becomes 1; "<=50K" becomes 0

# ---------------------------------------------------------------------------
# Columns used by the model
# ---------------------------------------------------------------------------
# Numerical features: real measurements, so distances between values mean something.
NUMERIC_FEATURES = [
    "age",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

# Categorical features: labels, not measurements -> one-hot encoded (S5).
CATEGORICAL_FEATURES = [
    "workclass",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Columns we deliberately do NOT give to the model (justified in eda.py / report):
#   - "education": same information as "education-num" (one-to-one mapping), and
#                  correlated predictors make coefficients unstable (S2).
#   - "fnlwgt":    census sampling weight, describes the survey design,
#                  not the person whose income we predict.
DROPPED_FEATURES = ["education", "fnlwgt"]

# Sensitive attributes, kept for the fairness audit later in the project.
SENSITIVE_FEATURES = ["sex", "race"]
