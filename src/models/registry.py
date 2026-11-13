import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow.entities.model_registry import ModelVersion 
from mlflow.exceptions import MlflowException
from mlflow.models import EvaluationResult, MetricThreshold, infer_signature
from mlflow.models.evaluation.validation import ModelValidationFailedException
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingRegressor 

from src.config import (
    CHALLENGER_ALIAS,
    CHAMPION_ALIAS,
    MAX_HOLDOUT_RMSE, 
    MIN_RELATIVE_IMPROVEMENT,
    MLFLOW_TRACKING_URI,
    REGISTERED_MODEL_NAME,
    TOP_FEATURES, 
)
from src.models.model import RegressionChampionWrapper

# Promotion gate metric
GATE_METRIC = "root_mean_squared_error"


def get_champion(client: MlflowClient)-> ModelVersion | None:
    """Return the current @champion model version, or None if there is none"""
    try:
        # Get the model version with the @champion alias
        return client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
    except MlflowException as e:
        if e.error_code in ("RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE"):
            return None
        raise


def model_exists() -> bool:
    """Return True if the registered model already has the @champion alias"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return get_champion(MlflowClient()) is not None


def load_champion_model(): 
    """Load the @champion model and return it with its registry version"""
    # Resovle the alias 
    champion = get_champion(MlflowClient())
    if champion is None:
        raise RuntimeError(f"No {REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS} model in the registry") 
    
    # Load the model from the registry
    model = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL_NAME}/{champion.version}") 

    # Show the loaded model version and return it
    print(f"Champion model v{champion.version} loaded from MLflow.") 
    return model, str(champion.version) 


def register_model(model: GradientBoostingRegressor, sample_input: pd.DataFrame) -> str:
    """Wrap the model, log it in the active run and register a new version"""
    # Wrap the model in a PyFunc wrapper
    wrapped_model = RegressionChampionWrapper(model=model, features=TOP_FEATURES)
    sample_output = wrapped_model.predict(context=None, model_input=sample_input)
    
    # Log the wrapped model to MLflow and register it
    info = mlflow.pyfunc.log_model(
        name="model",
        python_model=wrapped_model,
        signature=infer_signature(sample_input, sample_output),
        input_example=sample_input,
        registered_model_name=REGISTERED_MODEL_NAME,
    )
    return str(info.registered_model_version)


def promote_if_better(client: MlflowClient, version: str, candidate_result: EvaluationResult) -> bool:
    """Run the promotion gate and move @champion to the new version if it passes"""
    # Set the @challenger alias to the new version
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHALLENGER_ALIAS, version)

    # Baseline: metric already stored on the champion run (no re-evaluation on new data)
    champion = get_champion(client)
    baseline_result = None
    if champion is not None:
        champion_metrics = client.get_run(champion.run_id).data.metrics
        if GATE_METRIC in champion_metrics:
            # Get the champion's metric value and create a baseline EvaluationResult
            baseline_result = EvaluationResult(metrics={GATE_METRIC: champion_metrics[GATE_METRIC]}, artifacts={})
        else:
            print(f"Champion v{champion.version} has no {GATE_METRIC} metric and no comparable baseline, using only the absolute threshold")

    # Set the threshold for the promotion gate
    if baseline_result is None:
        threshold = MetricThreshold(threshold=MAX_HOLDOUT_RMSE, greater_is_better=False) 
    else:
        threshold = MetricThreshold(threshold=MAX_HOLDOUT_RMSE, min_relative_change=MIN_RELATIVE_IMPROVEMENT, greater_is_better=False)

    # Run the promotion gate
    candidate_rmse = candidate_result.metrics[GATE_METRIC]
    try:
        mlflow.validate_evaluation_results(
            validation_thresholds={GATE_METRIC: threshold},
            candidate_result=candidate_result,
            baseline_result=baseline_result,
        )
    except ModelValidationFailedException as e:
        client.set_model_version_tag(REGISTERED_MODEL_NAME, version, "validation_status", "failed")
        mlflow.set_tag("promotion", "rejected")
        print(f"v{version} REJECTED (holdout RMSE {candidate_rmse:.4f}): {e}")
        return False

    # If the promotion gate passed, promote the new version to @champion
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version, "validation_status", "passed")
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, version)
    client.delete_registered_model_alias(REGISTERED_MODEL_NAME, CHALLENGER_ALIAS)
    mlflow.set_tag("promotion", "promoted")

    # Show the promotion result
    print(f"v{version} PROMOTED to @{CHAMPION_ALIAS} (holdout RMSE {candidate_rmse:.4f})")
    return True
