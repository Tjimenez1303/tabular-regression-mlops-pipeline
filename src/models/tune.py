import mlflow
import optuna
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score

from src.config import (
    CV_FOLDS,
    RANDOM_STATE
)

def objective(trial: optuna.trial.Trial, X, y) -> float:
    """Evaluate one hyperparamter combination with cross-validation"""

    with mlflow.start_run(run_name=f"trial-{trial.number:03d}", nested=True):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 1, 5), 
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0), 
        }

        model = GradientBoostingRegressor(**params, random_state=RANDOM_STATE) 
        cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scores = -cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error")

        mlflow.log_params(params)
        mlflow.log_metrics({"cv_rmse": scores.mean(), "cv_rmse_std": scores.std()})

    return scores.mean() 


def tune_hyperparameters(X: pd.DataFrame, y: pd.Series, n_trials: int) -> tuple[dict, float]:
    """Run an Optuna study and return the best hyperparameters and CV RMSE"""
    # Suppress Optuna logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Create an Optuna study
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)

    # Log in console
    print(f"Best params: {study.best_params}")
    print(f"Best CV RMSE: {study.best_value:.4f}")

    return study.best_params, study.best_value