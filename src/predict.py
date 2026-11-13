import argparse
from pathlib import Path

import mlflow
import mlflow.data
import pandas as pd

from src.config import (
    BLIND_TEST_PATH,
    CHAMPION_ALIAS,
    DATA_PRED_DIR,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    REGISTERED_MODEL_NAME,
    TOP_FEATURES,
)
from src.models.registry import load_champion_model


def predict_dataframe(df_input: pd.DataFrame, model) -> pd.DataFrame:
    """Run prediction using MLflow model and return a single-column DataFrame."""
    # Validate input DataFrame
    missing_columns = [col for col in TOP_FEATURES if col not in df_input.columns]
    if missing_columns:
        raise ValueError(f"Input data is missing required features: {missing_columns}")
    
    return model.predict(df_input[TOP_FEATURES])


def predict_row(features: dict, model, model_version: str) -> float:
    """Predict one observation and send the result for MLflow"""
    with mlflow.start_span(name="predict") as span:
        # Save the input features
        span.set_inputs(features)
        
        # Predict the target value using the champion model
        df_predictions = predict_dataframe(pd.DataFrame([features]), model)
        prediction = float(df_predictions["target_pred"].iloc[0])
        
        # Save the prediction result
        span.set_outputs({"target_pred": prediction})

        # Put the model name and version in the trace
        if mlflow.get_current_active_span() is not None:
            mlflow.update_current_trace(tags={"model_name": REGISTERED_MODEL_NAME, "model_version": model_version})
    
    return prediction


def log_batch_inference(df_input: pd.DataFrame, df_predictions: pd.DataFrame, source_name: str, model_version: str) -> None:
    """Log one batch-inference run to MLflow, or print a warning if it cannot be logged"""
    # Tags to MLflow run
    predictions = df_predictions["target_pred"]
    tags = {
        "stage": "inference",
        "model_name": REGISTERED_MODEL_NAME,
        "model_version": model_version,
        "model_alias": CHAMPION_ALIAS,
    }

    # Observability must never break inference, so any MLflow error is only reported
    try:
        with mlflow.start_run(run_name="batch-inference", tags=tags):
            # Log the input dataset to MLflow
            dataset = mlflow.data.from_pandas(df_input, source=source_name, name=source_name)
            mlflow.log_input(dataset, context="inference")

            # Log the predictions to MLflow
            mlflow.log_metrics(
                {
                    "rows": len(predictions),
                    "prediction_mean": float(predictions.mean()),
                    "prediction_std": float(predictions.std()),
                    "prediction_min": float(predictions.min()),
                    "prediction_max": float(predictions.max()),
                }
            )

            # Save the predictions to a CSV in MLflow artifacts
            mlflow.log_text(df_predictions.to_csv(index=False), "predictions/predictions.csv")
    except Exception as exc:
        print(f"Batch inference not logged to MLflow: {exc}")


def run_batch_prediction(input_path: Path, output_path: Path = DATA_PRED_DIR / "predictions.csv") -> None:
    """Predict a CSV file with the champion model, save the results and log the run to MLflow"""
    # Validate input file existence
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV file not found at {input_path}")

    # Set up MLflow tracking
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # Load input data and model once
    df_input = pd.read_csv(input_path)
    model, model_version = load_champion_model()

    # Make predictions and save them
    predictions_df = predict_dataframe(df_input, model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_path, index=False)

    # Log the batch-inference run to MLflow
    log_batch_inference(df_input, predictions_df, input_path.name, model_version)

    print(f"Batch inference completed ({len(predictions_df)} rows). Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Arguments for input and output CSV files
    parser.add_argument("--input", type=Path, default=BLIND_TEST_PATH, help="CSV file to predict")
    parser.add_argument("--output", type=Path, default=DATA_PRED_DIR / "predictions.csv", help="Where to save the predictions CSV",)
    args = parser.parse_args()

    # Run the batch prediction with the provided arguments
    run_batch_prediction(args.input, args.output)