import io
from contextlib import asynccontextmanager
from typing import Annotated

import mlflow
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from src.app.schemas import HealthResponse, PredictionInput, PredictionOutput
from src.config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    REGISTERED_MODEL_NAME,
    TOP_FEATURES,
)
from src.models.registry import load_champion_model
from src.predict import log_batch_inference, predict_dataframe, predict_row

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Configure MLflow, load the champion model at startup and flush traces on shutdown"""
    # Set up MLflow tracking
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # Upload the champion model
    app.state.model, app.state.model_version = load_champion_model()
    yield
    # Flush any remaining traces to MLflow on shutdown
    mlflow.flush_trace_async_logging()


app = FastAPI(
    title="MLOps  Inference API", 
    description="Service for real-time and bulk CSV predictions", 
    version="0.1.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check(request: Request) -> HealthResponse: 
    """Verify service availability and report model health"""
    return HealthResponse(
        status="healthy",
        model_name=REGISTERED_MODEL_NAME,
        model_version=request.app.state.model_version,
        required_features=TOP_FEATURES,
    )


@app.post("/predict", response_model=PredictionOutput, tags=["Inference"])
def predict_single_row(request: Request, payload: PredictionInput) -> PredictionOutput: 
    """Predict target value for a single JSON observation"""
    try:
        prediction = predict_row(payload.model_dump(), request.app.state.model, request.app.state.model_version)
        return PredictionOutput(target_pred=prediction)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict-batch", tags=["Inference"])
def predict_batch_csv(request: Request, file: Annotated[UploadFile, File()]) -> StreamingResponse:
    """Upload a CSV file for bulk prediction and download the prediction in format CSV"""
    # Validate the uploaded file
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported")

    try:
        # Read the uploaded CSV
        df_input = pd.read_csv(file.file)
        
        # Predict using the loaded model
        predictions_df = predict_dataframe(df_input, request.app.state.model)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Batch inference failed: {exc}") from exc

    # Log the batch inference to MLflow
    log_batch_inference(df_input, predictions_df, file.filename, request.app.state.model_version)

    # Prepare the CSV output for streaming response
    output_buffer = io.StringIO()
    predictions_df.to_csv(output_buffer, index=False)
    return StreamingResponse(
        iter([output_buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"},
    )