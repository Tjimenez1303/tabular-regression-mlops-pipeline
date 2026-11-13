import argparse
import sys

import mlflow
import mlflow.data
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from src.config import (
    CHAMPION_ALIAS,
    CV_FOLDS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    N_TRIALS,
    RANDOM_STATE,
    REGISTERED_MODEL_NAME,
    TARGET_COLUMN,
    TEST_SIZE,
    TOP_FEATURES,
    TRAIN_DATA_PATH,
)
from src.data.load_data import load_train_data
from src.models.evaluate import evaluate_holdout
from src.models.registry import model_exists, promote_if_better, register_model
from src.models.tune import tune_hyperparameters


def train_model(n_trials: int) -> None:
    """Tune, evaluate, retrain on 100%, reister and gate the model in one parent run"""
    # Set up MLflow tracking
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # Load training data
    df = load_train_data()
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Pipeline run
    with mlflow.start_run(run_name="training-pipeline") as run:
        # Tags to MLflow run
        mlflow.set_tags({
            "stage": "training",
            "model_family": "gradient-boosting",
            "optimizer_engine": "optuna"
        })

        # Set MLflow train dataset
        train_dataset = mlflow.data.from_pandas(
            train_df[TOP_FEATURES + [TARGET_COLUMN]], 
            source=str(TRAIN_DATA_PATH),
            name="train", 
            targets=TARGET_COLUMN
        )
        mlflow.log_input(train_dataset, context="training")

        # Hyperparamter search
        best_params, best_cv_rmse = tune_hyperparameters(train_df[TOP_FEATURES], train_df[TARGET_COLUMN], n_trials)
        
        # Log best hyperparameters and metrics to MLflow
        mlflow.log_params(best_params)
        mlflow.log_params(
            {
                "n_trials": n_trials,
                "cv_folds": CV_FOLDS,
                "test_size": TEST_SIZE,
                "selected_features": ",".join(TOP_FEATURES),
                "train_rows": len(train_df),
                "holdout_rows": len(test_df),
                "total_rows": len(df),
            }
        )
        mlflow.log_metric("best_cv_rmse", best_cv_rmse)

        # Holdout evaluation
        candidate_result = evaluate_holdout(best_params, train_df, test_df)

        # Retrain on 100% of the data with best hyperparameters, register a new version and run the promotion gate
        final_model = GradientBoostingRegressor(**best_params, random_state=RANDOM_STATE) 
        final_model.fit(df[TOP_FEATURES], df[TARGET_COLUMN])

        # Use 5 rows as input example
        version = register_model(final_model, sample_input=df[TOP_FEATURES].head(5))

        # Log pipeline in console
        print(f"MLflow Run ID: {run.info.run_id}")
        print(f"Registered '{REGISTERED_MODEL_NAME}' v{version}")
        print(f"Validation Metrics: {candidate_result.metrics}")

        # Promote the model if it is better than the current champion
        promote_if_better(MlflowClient(), version, candidate_result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--if-missing", action="store_true", help="Train and register the model only if the @champion alias does not exist")
    parser.add_argument("--n-trials", type=int, default=N_TRIALS, help="Number of Optuna trials")
    args = parser.parse_args()

    if args.if_missing and model_exists():
        print(f"'{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}' already exists")
    else:
        train_model(args.n_trials)

        # Boostrap to ensure that a model is registered otherwise the API can't start
        if args.if_missing and not model_exists():
            sys.exit("Bootstrap failed: no model passed the promotion gate")