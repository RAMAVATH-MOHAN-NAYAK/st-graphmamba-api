import os
import zipfile
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import hf_hub_download

from model import load_model
from preprocess import (
    load_speed_h5,
    load_adjacency_pickle,
    make_model_inputs,
    make_adj_bias,
    TRAIN_MEAN,
    TRAIN_STD,
)

MODEL_REPO_ID = os.getenv("MODEL_REPO_ID", "").strip()
HF_TOKEN = os.getenv("HF_TOKEN") or None

MODEL_FILENAME = os.getenv("MODEL_FILENAME", "st_graphmamba_metrla.pt")
ADJ_FILENAME = os.getenv("ADJ_FILENAME", "adj_METR-LA.pkl")
DATASET_FILENAME = os.getenv("DATASET_FILENAME", "METR-LA.h5")
DATASET_ZIP_FILENAME = os.getenv("DATASET_ZIP_FILENAME", "METR-LA.h5.zip")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / MODEL_FILENAME
ADJ_PATH = BASE_DIR / ADJ_FILENAME
DATASET_PATH = BASE_DIR / DATASET_FILENAME

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="ST-GraphMamba METR-LA Inference API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
adj_bias = None
speed = None
sensor_ids = None

class PredictionRequest(BaseModel):
    history: list[list[float]]

def download_artifact(filename: str):
    if not MODEL_REPO_ID:
        raise RuntimeError("MODEL_REPO_ID is not configured.")
    return hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=filename,
        token=HF_TOKEN,
        local_dir=str(BASE_DIR),
    )

def ensure_artifacts():
    if not MODEL_PATH.exists():
        download_artifact(MODEL_FILENAME)
    if not ADJ_PATH.exists():
        download_artifact(ADJ_FILENAME)

    if not DATASET_PATH.exists():
        zip_path = BASE_DIR / DATASET_ZIP_FILENAME
        if not zip_path.exists():
            try:
                download_artifact(DATASET_FILENAME)
            except Exception:
                download_artifact(DATASET_ZIP_FILENAME)
        if not DATASET_PATH.exists() and zip_path.exists():
            with zipfile.ZipFile(zip_path, "r") as z:
                members = [m for m in z.namelist() if m.endswith("METR-LA.h5")]
                if not members:
                    raise RuntimeError("METR-LA.h5 was not found in the zip.")
                z.extract(members[0], BASE_DIR)
                extracted = BASE_DIR / members[0]
                if extracted != DATASET_PATH:
                    extracted.replace(DATASET_PATH)

def startup():
    global model, adj_bias, speed, sensor_ids
    ensure_artifacts()

    model = load_model(str(MODEL_PATH), n_sensors=207).to(DEVICE)

    _, _, adj = load_adjacency_pickle(str(ADJ_PATH))
    if adj.shape != (207, 207):
        raise RuntimeError(f"Expected adjacency (207,207), got {adj.shape}.")
    adj_bias = make_adj_bias(adj).to(DEVICE)

    speed, sensor_ids = load_speed_h5(str(DATASET_PATH))
    print(f"Loaded ST-GraphMamba on {DEVICE}; dataset={speed.shape}")

@app.on_event("startup")
def on_startup():
    startup()

@app.get("/")
def root():
    return {
        "service": "ST-GraphMamba METR-LA",
        "status": "running",
        "device": str(DEVICE),
        "model_repo": MODEL_REPO_ID,
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "dataset_loaded": speed is not None,
        "device": str(DEVICE),
        "sensors": 207,
        "history_len": 24,
        "horizon": 12,
        "dataset_shape": list(speed.shape) if speed is not None else None,
    }

def run_prediction(history: np.ndarray, start_row: int):
    if model is None or adj_bias is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    x, tf = make_model_inputs(history, start_row=start_row)
    x, tf = x.to(DEVICE), tf.to(DEVICE)

    with torch.no_grad():
        reg_norm, cls_logits = model(x, tf, adj_bias)

    reg_raw = reg_norm.cpu().numpy()[0] * TRAIN_STD + TRAIN_MEAN
    congestion = cls_logits.argmax(-1).cpu().numpy()[0]

    return {
        "predictions_mph": reg_raw.tolist(),
        "congestion_class": congestion.tolist(),
        "history_shape": list(history.shape),
        "horizon": 12,
        "sensors": 207,
    }

@app.post("/predict")
def predict(req: PredictionRequest):
    history = np.asarray(req.history, dtype=np.float32)
    if history.shape != (24, 207):
        raise HTTPException(
            status_code=400,
            detail=f"history must be 24 x 207; received {history.shape}.",
        )
    return run_prediction(history, 0)

@app.get("/predict/latest")
def predict_latest():
    if speed is None:
        raise HTTPException(status_code=503, detail="METR-LA dataset not loaded.")

    start = len(speed) - 24
    result = run_prediction(speed[start:], start)
    result.update({
        "source": "METR-LA.h5",
        "start_row": start,
        "end_row": len(speed) - 1,
        "sensor_ids": sensor_ids,
    })
    return result

@app.post("/predict/file")
async def predict_file(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a .csv file.")

    raw = await file.read()
    with NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name

    try:
        df = pd.read_csv(tmp_path)
        numeric = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
        history = numeric.to_numpy(dtype=np.float32)

        if history.shape != (24, 207):
            raise HTTPException(
                status_code=400,
                detail=f"CSV must be 24 x 207 numeric values; received {history.shape}.",
            )
        return run_prediction(history, 0)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

@app.get("/validate")
def validate(start_row: int = 30000):
    if speed is None:
        raise HTTPException(
            status_code=503,
            detail="METR-LA dataset not loaded."
        )

    # We need:
    # 24 rows for input
    # 12 rows for ground truth
    required_rows = 24 + 12

    if start_row < 0:
        raise HTTPException(
            status_code=400,
            detail="start_row must be >= 0."
        )

    if start_row + required_rows > len(speed):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Not enough data. start_row={start_row}, "
                f"dataset length={len(speed)}. "
                f"Need {required_rows} rows."
            )
        )

    # -----------------------------------
    # 1. Get 24 historical observations
    # -----------------------------------
    history = speed[
        start_row:start_row + 24
    ]

    # -----------------------------------
    # 2. Run the actual trained model
    # -----------------------------------
    result = run_prediction(
        history,
        start_row
    )

    # -----------------------------------
    # 3. Get the real next 12 observations
    # -----------------------------------
    actual = speed[
        start_row + 24:start_row + 36
    ]

    # -----------------------------------
    # 4. Convert prediction to numpy
    # -----------------------------------
    predicted = np.asarray(
        result["predictions_mph"],
        dtype=np.float32
    )

    actual = np.asarray(
        actual,
        dtype=np.float32
    )

    # Safety check
    if predicted.shape != actual.shape:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Shape mismatch: "
                f"predicted={predicted.shape}, "
                f"actual={actual.shape}"
            )
        )

    # -----------------------------------
    # 5. Calculate errors
    # -----------------------------------
    error = predicted - actual

    mae = float(
        np.mean(np.abs(error))
    )

    rmse = float(
        np.sqrt(np.mean(error ** 2))
    )

    # -----------------------------------
    # 6. Negative predictions
    # -----------------------------------
    negative_count = int(
        np.sum(predicted < 0)
    )

    # -----------------------------------
    # 7. Count extremely large values
    # -----------------------------------
    high_count = int(
        np.sum(predicted > 100)
    )

    # -----------------------------------
    # 8. Find worst sensor
    # -----------------------------------
    sensor_mae = np.mean(
        np.abs(predicted - actual),
        axis=1
    )

    worst_sensor_index = int(
        np.argmax(sensor_mae)
    )

    worst_sensor_id = (
        sensor_ids[worst_sensor_index]
        if sensor_ids is not None
        else str(worst_sensor_index)
    )

    return {
        "status": "validation_complete",

        "start_row": start_row,

        "input_rows": [
            start_row,
            start_row + 23
        ],

        "ground_truth_rows": [
            start_row + 24,
            start_row + 35
        ],

        "prediction_shape": list(predicted.shape),

        "actual_shape": list(actual.shape),

        "mae_mph": mae,

        "rmse_mph": rmse,

        "negative_predictions": negative_count,

        "predictions_over_100_mph": high_count,

        "prediction_min_mph": float(
            np.min(predicted)
        ),

        "prediction_max_mph": float(
            np.max(predicted)
        ),

        "actual_min_mph": float(
            np.min(actual)
        ),

        "actual_max_mph": float(
            np.max(actual)
        ),

        "worst_sensor": {
            "index": worst_sensor_index,
            "sensor_id": worst_sensor_id,
            "mae_mph": float(
                sensor_mae[worst_sensor_index]
            )
        }
    }
