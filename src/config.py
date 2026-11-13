import os
from pathlib import Path

# Base project directories
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PRED_DIR = ROOT_DIR / "data" / "predictions"
MODELS_DIR = ROOT_DIR / "models"

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

# Optimal hyperparameters
MODEL_PARAMS = {
    "learning_rate": 0.1,
    "max_depth": 2,
    "min_samples_leaf": 10,
    "n_estimators": 300,
    "subsample": 0.8,
    "random_state": 42,
}