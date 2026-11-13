import io
import pytest
from fastapi.testclient import TestClient
import pandas as pd

from src.app.main import app
from src.config import REGISTERED_MODEL_NAME, TOP_FEATURES

@pytest.fixture(scope="module")
def client():
    """Create a TestClient for the FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Verify /health returns 200 and reports active model."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["model_name"] == REGISTERED_MODEL_NAME
    assert body["model_version"].isdigit()
    assert body["required_features"] == TOP_FEATURES


def test_predict_single_endpoint(client):
    """Verify /predict returns a valid float target_pred."""
    sample_payload = {
        "feature_2": 300.0,
        "feature_13": 150.0,
        "feature_9": 50.0,
        "feature_11": 3.5,
        "feature_18": 25.0,
    }
    response = client.post("/predict",json=sample_payload)
    assert response.status_code == 200
    assert "target_pred" in response.json()
    assert isinstance(response.json()["target_pred"], float)


def test_predict_batch_csv_endpoint(client):
    """Verify /predict-batch processes bulk CSV and outputs single-column target_pred."""
    df_mock = pd.DataFrame(
        {
            "feature_2": [100.0, 400.0], 
            "feature_13": [80.0, 220.0],
            "feature_9": [20.0, 75.0],
            "feature_11": [1.5, 6.0],
            "feature_18": [10.0, 45.0], 
        }
    )
    csv_bytes = df_mock.to_csv(index=False ).encode("utf-8") 
    files = {"file": ("test_batch.csv", io.BytesIO(csv_bytes), "text/csv")}

    response = client.post("/predict-batch", files=files)
    assert response.status_code == 200

    result_df = pd.read_csv(io.BytesIO(response.content))
    assert list(result_df.columns) == ["target_pred"] 
    assert len(result_df) == 2 