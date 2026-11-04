"""FastAPI service for the Bangladesh crop-yield research prototype.

Pretrained model artifacts are loaded at startup and used for inference. A model
that cannot initialize is marked unavailable; this service never fills in fake
scores or predictions.
"""

from __future__ import annotations

import json
import logging
import io
import os
import threading
from pathlib import Path
from typing import Any
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("crop_yield")

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path(os.getenv("CROP_YIELD_DATA", ROOT / "bangladesh_crop_yield_synthetic_1000.csv"))
MODEL_DIR = Path(os.getenv("CROP_YIELD_MODELS", ROOT / "crop_yield_models"))
MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"
NUMERIC_IMPUTER_PATH = MODEL_DIR / "numeric_imputer.joblib"
MODEL_PATHS = {
    "TabFM": MODEL_DIR / "tabfm_final.joblib",
    "TabPFN": MODEL_DIR / "tabpfn_final.joblib",
}
TARGET = "crop_yield_tonnes_ha"
METADATA = ["record_id", "country", "data_status"]
FEATURES = [
    "region", "year", "crop", "season", "rainfall_mm_season",
    "avg_temperature_c", "relative_humidity_pct", "soil_ph",
    "soil_moisture_m3_m3", "solar_radiation_mj_m2_day", "ndvi",
    "fertilizer_kg_ha", "irrigation_used_0_no_1_yes",
    "previous_season_yield_tonnes_ha",
]
NUMERIC = [column for column in FEATURES if column not in {"region", "crop", "season"}]
CATEGORICAL = ["region", "crop", "season"]

app = FastAPI(title="Bangladesh Crop Yield API", version="1.0.0")
models: dict[str, Any] = {}
statuses: dict[str, dict[str, Any]] = {
    name: {"ready": False, "message": "Not initialized"}
    for name in ("TabFM", "TabPFN")
}
metrics: dict[str, dict[str, float]] = {}
metadata: dict[str, Any] = {"dataset_loaded": False}
imputer: Any | None = None
_predict_lock = threading.Lock()


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    region: str
    year: int
    crop: str
    season: str
    rainfall_mm_season: float = Field(ge=0)
    avg_temperature_c: float
    relative_humidity_pct: float = Field(ge=0, le=100)
    soil_ph: float = Field(ge=0, le=14)
    soil_moisture_m3_m3: float = Field(ge=0, le=1)
    solar_radiation_mj_m2_day: float = Field(ge=0)
    ndvi: float = Field(ge=-1, le=1)
    fertilizer_kg_ha: float = Field(ge=0)
    irrigation_used_0_no_1_yes: int
    previous_season_yield_tonnes_ha: float = Field(ge=0)

    @field_validator(*[f for f in FEATURES if f not in CATEGORICAL])
    @classmethod
    def finite_values(cls, value: float | int) -> float | int:
        if not np.isfinite(value):
            raise ValueError("Numeric values must be finite")
        return value

    @field_validator("irrigation_used_0_no_1_yes")
    @classmethod
    def valid_irrigation(cls, value: int) -> int:
        if value not in (0, 1):
            raise ValueError("Irrigation must be 0 (No) or 1 (Yes)")
        return value


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    required = set(FEATURES + [TARGET])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    # Match notebook behavior: drop rows only when the target is missing.
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    if df.empty:
        raise ValueError("Dataset contains no usable target values")
    x = df[FEATURES].copy()
    for column in CATEGORICAL:
        x[column] = x[column].fillna("Unknown").astype(str)
    for column in NUMERIC:
        x[column] = pd.to_numeric(x[column], errors="coerce")
    y = df[TARGET].astype(float)
    return df, x, y


def _calculate_metrics(y_true: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, predictions))
    return {
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, predictions)),
        "R2": float(r2_score(y_true, predictions)),
    }


def _load_joblib_artifact(path: Path) -> Any:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_torch_storage(data: bytes) -> Any:
        return torch.load(io.BytesIO(data), map_location=device, weights_only=False)

    # The serialized estimators contain PyTorch storages saved on their training
    # GPU; remap them so the same artifacts can load on CPU-only machines.
    with patch("torch.storage._load_from_bytes", load_torch_storage):
        return joblib.load(path)


def _load_pretrained_models() -> dict[str, Any] | None:
    global imputer
    models.clear()
    metrics.clear()
    imputer = None

    try:
        with MODEL_METADATA_PATH.open(encoding="utf-8") as file:
            model_metadata = json.load(file)
        if model_metadata.get("target") != TARGET:
            raise ValueError("Model metadata target does not match the API target")
        if model_metadata.get("feature_columns") != FEATURES:
            raise ValueError("Model metadata feature columns do not match the API schema")
        if model_metadata.get("numeric_columns") != NUMERIC:
            raise ValueError("Model metadata numeric columns do not match the API schema")
        if model_metadata.get("categorical_columns") != CATEGORICAL:
            raise ValueError("Model metadata categorical columns do not match the API schema")

        imputer = _load_joblib_artifact(NUMERIC_IMPUTER_PATH)
        imputer_columns = list(getattr(imputer, "feature_names_in_", []))
        if imputer_columns != NUMERIC:
            raise ValueError("Saved numeric imputer columns do not match the API schema")
    except Exception as exc:
        log.exception("Pretrained model preprocessing initialization failed")
        imputer = None
        for name in statuses:
            statuses[name] = {
                "ready": False,
                "message": f"Preprocessing unavailable: {type(exc).__name__}: {exc}",
            }
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name, path in MODEL_PATHS.items():
        try:
            model = _load_joblib_artifact(path)
            if name == "TabPFN":
                model_features = list(getattr(model, "feature_names_in_", []))
                if model_features != FEATURES:
                    raise ValueError("Saved TabPFN feature columns do not match the API schema")
                model.device = str(device)
                model.devices_ = (device,)
                model.executor_.to(
                    model.devices_,
                    model.forced_inference_dtype_,
                    model.executor_.dtype_byte_size,
                )
            models[name] = model
            statuses[name] = {"ready": True, "message": f"Ready (pretrained artifact; {device.type})"}
            log.info("%s pretrained artifact loaded on %s", name, device.type)
        except Exception as exc:
            log.exception("%s pretrained artifact failed to load", name)
            statuses[name] = {
                "ready": False,
                "message": f"Artifact load failed: {type(exc).__name__}: {exc}",
            }
    return model_metadata


@app.on_event("startup")
def initialize() -> None:
    global metadata
    try:
        df, x, _ = _load_data()
    except Exception as exc:
        log.exception("Dataset initialization failed")
        metadata = {"dataset_loaded": False, "error": str(exc)}
        for name in statuses:
            statuses[name] = {"ready": False, "message": "Dataset unavailable"}
        return

    is_synthetic = (
        "synthetic" in DATA_PATH.name.lower()
        or ("data_status" in df and df["data_status"].astype(str).str.contains("SYNTHETIC", case=False, na=False).any())
    )
    categories = {column: sorted(x[column].dropna().astype(str).unique().tolist()) for column in CATEGORICAL}
    numeric_years = pd.to_numeric(x["year"], errors="coerce").dropna().unique()
    categories["year"] = sorted(int(value) for value in numeric_years)
    metadata = {
        "dataset_loaded": True,
        "rows": len(df),
        "target": TARGET,
        "features": FEATURES,
        "categories": categories,
        "synthetic": bool(is_synthetic),
        "data_status": sorted(df["data_status"].astype(str).unique().tolist()) if "data_status" in df else [],
        "evaluation": {
            "protocol": "Pretrained artifacts; no held-out evaluation performed",
            "metrics_available": False,
        },
    }
    model_metadata = _load_pretrained_models()
    if model_metadata is not None:
        metadata["model_artifacts"] = {
            "training_rows": model_metadata.get("training_rows"),
            "dataset": model_metadata.get("dataset"),
            "training_note": model_metadata.get("training_note"),
        }


@app.get("/health")
def health() -> dict[str, Any]:
    service_ready = metadata.get("dataset_loaded") and imputer is not None and any(
        status["ready"] for status in statuses.values()
    )
    return {"service": "ok" if service_ready else "degraded", "dataset": metadata, "models": statuses}


@app.get("/metadata")
def get_metadata() -> dict[str, Any]:
    if not metadata.get("dataset_loaded"):
        raise HTTPException(status_code=503, detail=metadata.get("error", "Dataset is unavailable"))
    return metadata


@app.get("/metrics")
def get_metrics() -> dict[str, Any]:
    return {"protocol": metadata.get("evaluation"), "synthetic": metadata.get("synthetic"), "models": {
        name: {"status": statuses[name], "metrics": metrics.get(name)} for name in statuses
    }}


@app.post("/predict")
def predict(payload: PredictionInput) -> dict[str, Any]:
    if not metadata.get("dataset_loaded") or imputer is None:
        raise HTTPException(status_code=503, detail="Prediction service is not ready: dataset or saved preprocessing is unavailable")
    submitted = payload.model_dump()
    categories = metadata["categories"]
    for field in CATEGORICAL + ["year"]:
        valid = categories.get(field, [])
        if submitted[field] not in valid:
            raise HTTPException(status_code=422, detail=f"{field} must be one of the dataset values")
    row = pd.DataFrame([{column: submitted[column] for column in FEATURES}], columns=FEATURES)
    for col in CATEGORICAL:
        row[col] = row[col].astype(str)
    try:
        row.loc[:, NUMERIC] = imputer.transform(row[NUMERIC])
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid numeric input: {type(exc).__name__}") from exc
    results: dict[str, Any] = {}
    with _predict_lock:
        for name, model in models.items():
            try:
                pred = np.asarray(model.predict(row), dtype=float).reshape(-1)
                if len(pred) != 1 or not np.isfinite(pred).all():
                    raise ValueError("Model returned invalid prediction")
                results[name] = {"prediction_tonnes_ha": float(pred[0]), "status": "ok"}
            except Exception as exc:
                log.exception("%s prediction failed", name)
                statuses[name] = {"ready": False, "message": f"Prediction failed: {type(exc).__name__}"}
                results[name] = {"status": "error", "message": f"Prediction failed: {type(exc).__name__}"}
    if not results:
        raise HTTPException(status_code=503, detail="Neither model is currently available")
    return {"results": results, "unit": "tonnes/ha", "synthetic": metadata.get("synthetic", True)}
