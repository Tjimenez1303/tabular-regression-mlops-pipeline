import argparse
import mlflow
import mlflow.pyfunc
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

from src.config import (
    CHAMPION_ALIAS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_PARAMS,
    MODELS_DIR,
    REGISTERED_MODEL_NAME,
    TARGET_COLUMN,
    TOP_FEATURES,
    TRAIN_DATA_PATH
)


class RegressionChampionWrapper(mlflow.pyfunc.PythonModel):
    """MLflow model wrapper for the champion regression model."""

    def __init__(self, model: GradientBoostingRegressor, features: list[str]):
        """Initialize the wrapper with the trained model."""
        self.model = model
        self.features = features

    def predict(self, context, model_input: pd.DataFrame, params=None)-> pd.DataFrame:
        """Make predictions using the trained model."""
        # Ensure the input DataFrame has the required features
        missing_columns = [col for col in self.features if col not in model_input.columns]
        if missing_columns:
            raise ValueError(f"Input data is missing required features: {missing_columns}")

        X_selected = model_input[self.features].copy()
        if X_selected.isnull().any().any():
            raise ValueError("Input features contain null/NaN values.")

        # Make predictions using the trained model
        predictions = self.model.predict(X_selected)
        return pd.DataFrame({"target_pred": predictions})


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


def evaluate_model(df: pd.DataFrame) -> dict:
    """Evaluate model generalization using an 80/20 holdout split"""
    X = df[TOP_FEATURES]
    y = df[TARGET_COLUMN]

    # Split the data into training and testing sets (80/20 split)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=MODEL_PARAMS['random_state'])

    # Train the model
    model = GradientBoostingRegressor(**MODEL_PARAMS)
    model.fit(X_train, y_train)

    # Make predictions on the validation set
    model_predictions = model.predict(X_val)

    # Return evaluation metrics
    return {
        "val_rmse": round(float(root_mean_squared_error(y_val, model_predictions)), 4),
        "val_mae": round(float(mean_absolute_error(y_val, model_predictions)), 4),
        "val_r2": round(float(r2_score(y_val, model_predictions)), 4)
    }


def train_and_register() -> None:
    """Train final model on 100% of data, wrap in PyFunc, log to MLflow, and assign @champion alias"""
    # Load training data
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_train_data()

    # Set up MLflow tracking
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # Compute evaluation metrics
    metrics = evaluate_model(df)

    # Train final model on 100% of data
    X_full = df[TOP_FEATURES]
    y_full = df[TARGET_COLUMN] 

    # Train the model
    final_model = GradientBoostingRegressor(**MODEL_PARAMS)
    final_model.fit(X_full, y_full)

    # Wrap model in a PyFunc wrapper for MLflow
    wrapped_model = RegressionChampionWrapper(model=final_model, features=TOP_FEATURES)

    # Infer signature
    sample_input = X_full.head(5)
    sample_output = wrapped_model.predict(context=None, model_input=sample_input)
    signature = infer_signature(sample_input, sample_output)

    # Save the final model to disk
    # sample_predictions = final_model.predict(X_full.head(5))
    # signature = infer_signature(X_full.head(5), sample_predictions)

    with mlflow.start_run(run_name="final_model") as run:
        # Log model and metrics to MLflow
        mlflow.log_params(MODEL_PARAMS)
        mlflow.log_param("Selected Features", ",".join(TOP_FEATURES))
        mlflow.log_param("Training Rows", len(df))
        mlflow.log_metrics(metrics)

        # Log the model to MLflow and register it
        mlflow.pyfunc.log_model(
            name="model",
            python_model=wrapped_model,
            signature=signature,
            input_example=sample_input,
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        # Initialize MLflow client
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
        latest_version = max(int(v.version) for v in versions)
        client.set_registered_model_alias(
            name=REGISTERED_MODEL_NAME,
            alias=CHAMPION_ALIAS,
            version=str(latest_version)
        )

        # Show the run ID and model version for reference
        print(f"MLflow Run ID: {run.info.run_id}")
        print(f"Registered Model '{REGISTERED_MODEL_NAME}' v{latest_version} as @{CHAMPION_ALIAS}")
        print(f"Validation Metrics: {metrics}")


def model_exists() -> bool:
    """Return True if the registered model already has the @champion alias"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    try:
        client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
        return True
    except MlflowException as e:
        if e.error_code in ("RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE"):
            return False
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--if-missing", action="store_true", help="Train and register the model only if the @champion alias does not exist")

    args = parser.parse_args()
    if args.if_missing and model_exists():
        print(f"'{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}' already exists")
    else:
        train_and_register()