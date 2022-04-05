from pathlib import Path

PROJECT_NAME = "Project_Template"
DESCRIPTION = """
Distilling a model through an API.
"""

PROJECT_DIR = Path.home() / "project_template"
DATA_DIR = PROJECT_DIR / "data"
EXPERIMENTS_DIR = PROJECT_DIR / "experiments"

# BRANCHES
MAIN = "main"

# SCRIPT PATHS
MAIN_PATH = PROJECT_DIR / "main.py"

# GENERAL
IMDB = "imdb"
AUTO = "auto"

# WANDB
WANDB = "wandb"
WANDB_PROJECT = "WANDB_PROJECT"

# SPLITS
TRAIN = "train"
VALIDATION = "validation"
TEST = "test"
UNLABELED = "unlabeled"
ALL = "all"
SPLITS = [TRAIN, VALIDATION, TEST, UNLABELED]

# MODEL TYPES
CLS = "cls"  # Classification
QA = "qa"  # Question Answering
TCLS = ""  # Token Classification
ALL_MODEL_TYPES = [CLS, QA, TCLS]

# TRAINER TYPES
STANDARD = "standard"
CUSTOM = "custom"
ALL_TRAINER_TYPES = [STANDARD, CUSTOM]

# METRICS
AGREEMENT = "agreement"
ACCURACY = "accuracy"
NOISE = "iteration_noise"
F1 = "f1"
MACRO = "macro"
MACRO_F1 = "macro-f1"
MATTHEWS_CORRELATION = "matthews_correlation"

# FEATURES
TEXT = "text"
LABELS = "labels"
INPUT_IDS = "input_ids"
TRAIN_SAMPLES = "train_samples"
EVAL = "eval"
EVAL_SAMPLES = "eval_samples"
