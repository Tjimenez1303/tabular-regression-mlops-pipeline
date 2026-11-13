import io
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
import pandas as pd

from app.schemas import HealthResponse, PredictionInput, PredictionOutput
from src.config import TOP_FEATURES
from src.predict import load_champion_model, predict_dataframe

@asynccontextmanager
async def lifespan(app: FastAPI): 
    """Lifespan context manager to load the champion model at startup"""
    model, source = load_champion_model()
    app.state.model = model
    app.state.model_source = source 
    yield


app = FastAPI(
    title="MLOps  Inference API", 
    description="Service for real-time and bulk CSV predictions", 
    version="0.1.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check(request: Request) -> HealthResponse: 
    """Verify service availability and report the active model source"""
    return HealthResponse( 
        status="healthy", 
        model_source=request.app.state.model_source,
        required_features=TOP_FEATURES, 
    )


@app.post("/predict", response_model=PredictionOutput, tags=["Inference"])
def predict_single(request: Request, payload: PredictionInput) -> PredictionOutput:
    """Predict target value for a single JSON observation"""
    try:
        input_df = pd.DataFrame([payload.model_dump()])
        result_df = predict_dataframe(input_df, model=request.app.state.model)
        return PredictionOutput(target_pred=float(result_df["target_pred"].iloc[0]))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict-batch", tags=["Inference"])
async def predict_batch_csv(request: Request, file: UploadFile = File(...)) -> StreamingResponse:
    """Upload a CSV file for bulk prediction and download the prediction in format CSV"""
    # Validate the uploaded file
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported")

    try:
        # Read the uploaded CSV
        contents = await file.read()
        input_df = pd.read_csv(io.BytesIO(contents))
        
        # Predict using the loaded model
        result_df = predict_dataframe(input_df, model=request.app.state.model)

        # Prepare the CSV output for streaming response
        output_buffer = io.StringIO()
        result_df.to_csv(output_buffer, index=False)
        output_buffer.seek(0)

        return StreamingResponse(
            iter([output_buffer.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"},
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Batch inference failed: {exc}") from exc