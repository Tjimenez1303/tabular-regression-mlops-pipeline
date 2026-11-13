import pandas as pd

from src.config import (
    TARGET_COLUMN,
    TOP_FEATURES,
    TRAIN_DATA_PATH
)

def load_train_data() -> pd.DataFrame:
    """Load raw training dataset and validate required columns"""
    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(f"Training data file not found at {TRAIN_DATA_PATH}")

    # Load the training data
    df = pd.read_csv(TRAIN_DATA_PATH)

    # Validate required columns
    required_columns = TOP_FEATURES + [TARGET_COLUMN]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in training data: {missing_columns}")

    return df