import mlflow.pyfunc
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


class RegressionChampionWrapper(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc wrapper that validates the input features before predicting"""

    def __init__(self, model: GradientBoostingRegressor, features: list[str]):
        """Initialize the wrapper with the trained model and its feature list"""
        self.model = model
        self.features = features

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        """Make predictions using the trained model"""
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