from pathlib import Path
import joblib
import mlflow
import mlflow.pyfunc
import pandas as pd

from src.config import (
    BLIND_TEST_PATH,
    CHAMPION_ALIAS,
    DATA_PRED_DIR,
    MLFLOW_TRACKING_URI,
    REGISTERED_MODEL_NAME,
    MLFLOW_EXPERIMENT_NAME
)


def load_champion_model():
    """Load the champion model from MLflow or fallback to a local model."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Load the champion model from MLflow
    model_uri = f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}"
    model = mlflow.pyfunc.load_model(model_uri)
    
    print("Champion model loaded from MLflow.")
    return model, f"mlflow:{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}"


def predict_dataframe (df: pd.DataFrame, model=None) -> pd.DataFrame:
    """Run prediction using the champion wrapper and return a single-column DataFrame."""
    # Load champion model if not provided
    if model is None:
        model, _ = load_champion_model()

    # Check if the loaded object is an MLflow PyFuncModel or the raw fallback wrapper
    if isinstance(model, mlflow.pyfunc.PyFuncModel):
        return model.predict(df)
    
    return model.predict(context=None, model_input=df)



def run_batch_prediction(input_path: Path, output_path: Path = DATA_PRED_DIR / "predictions.csv", log_to_mlflow: bool = True) -> Path:
    """Run batch prediction on a CSV file and save the results."""
    # Validate input file existence
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV file not found at {input_path}")

    # Create output directory if it doesn't exist
    DATA_PRED_DIR.mkdir(parents=True, exist_ok=True)

    # Load input data and model once
    df_input = pd.read_csv(input_path)
    model, model_source = load_champion_model()

    # Make predictions
    df_predictions = predict_dataframe(df_input, model=model)
    df_predictions.to_csv(output_path, index=False)

    # Log predicions CSV to MLflow UI
    if log_to_mlflow:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        with mlflow.start_run(run_name="batch-inference"):
            mlflow.log_param("model_source", model_source)
            mlflow.log_param("input_file", input_path.name)
            mlflow.log_param("rows_predicted", len(df_predictions))
            mlflow.log_artifact(str(output_path), artifact_path="predictions")


    # Save predictions to output CSV
    df_predictions.to_csv(output_path, index=False)
    print(f"Batch inference completed ({len(df_predictions)} rows) using {model_source}. Saved to: {output_path}")

    return output_path



if __name__ == "__main__":
    run_batch_prediction(BLIND_TEST_PATH)