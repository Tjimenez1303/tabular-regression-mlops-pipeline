import mlflow
import mlflow.data
import pandas as pd
from mlflow.models import EvaluationResult
from sklearn.ensemble import GradientBoostingRegressor

from src.config import (
    RANDOM_STATE,
    TARGET_COLUMN,
    TOP_FEATURES, 
    TRAIN_DATA_PATH
)


def evaluate_holdout(params: dict, train_df: pd.DataFrame, test_df: pd.DataFrame) -> EvaluationResult:
    """Evaluate the model on the holdout dataset and log the results to MLflow"""
    # Train the model with the trainging data
    model = GradientBoostingRegressor(**params, random_state=RANDOM_STATE)
    model.fit(train_df[TOP_FEATURES], train_df[TARGET_COLUMN])

    # Evaluate the model on the holdout data
    evaluate_df = test_df[TOP_FEATURES + [TARGET_COLUMN]].copy()
    evaluate_df["target_pred"] = model.predict(test_df[TOP_FEATURES])
    evaluate_dataset = mlflow.data.from_pandas(
        evaluate_df,
        source=str(TRAIN_DATA_PATH),
        name="holdout",
        targets=TARGET_COLUMN,
        predictions="target_pred",
    )

    # Log the evaluation results to MLflow
    mlflow.log_input(evaluate_dataset, context="evaluation")
    return mlflow.models.evaluate(data=evaluate_dataset, model_type="regressor")
