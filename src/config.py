import os
from pathlib import Path

# Base project directories
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PRED_DIR = ROOT_DIR / "data" / "predictions"

# File paths
TRAIN_DATA_PATH = DATA_RAW_DIR / "train.csv"
BLIND_TEST_PATH = DATA_RAW_DIR / "blind_test_data.csv"

# MLFlow configuration
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", f"sqlite:///{ROOT_DIR / 'mlflow.db'}"
)
MLFLOW_EXPERIMENT_NAME = "tabular-regression-pipeline"
REGISTERED_MODEL_NAME = "gradient-boosting-regressor"
CHAMPION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"

# Target variable
TARGET_COLUMN = "target"

# Feature Selection columns
TOP_FEATURES = [
    "feature_2",
    "feature_13",
    "feature_9",
    "feature_11",
    "feature_18",
]

# Training pipeline settings
RANDOM_STATE = 42 
TEST_SIZE = 0.2 
CV_FOLDS = 5
N_TRIALS = 30 

# Promotion gate
MAX_HOLDOUT_RMSE = 2.5
MIN_RELATIVE_IMPROVEMENT = 0.01