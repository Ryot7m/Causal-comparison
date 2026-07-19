"""
Application Configuration
"""

# ==========================
# Dataset
# ==========================

OUTCOME_COL = "Outcome"
TREATMENT_COL = "Treatment"

# 説明変数から除外する列
EXCLUDE_COLUMNS = [
    OUTCOME_COL,
    TREATMENT_COL,
]

# ==========================
# Machine Learning
# ==========================

N_SPLITS = 5
RANDOM_STATE = 42

# ==========================
# Path
# ==========================

DATA_DIR = "data"
OUTPUT_DIR = "results"

# ==========================
# API
# ==========================

API_TITLE = "Causal Inference Platform"
API_VERSION = "1.0.0"