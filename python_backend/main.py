"""FastAPI service for the Bangladesh crop-yield research prototype.

Models and metrics are computed from the supplied CSV at startup. A model that
cannot initialize is marked unavailable; this service never fills in fake scores
or predictions.
"""

from __future__ import annotations

import os
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("crop_yield")

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path(os.getenv("CROP_YIELD_DATA", ROOT / "bangladesh_crop_yield_synthetic_1000.csv"))
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
train_x: pd.DataFrame | None = None
train_y: np.ndarray | None = None
imputer: SimpleImputer | None = None
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


def _fit_and_evaluate(name: str, x_train: pd.DataFrame, y_train: np.ndarray,
                      x_test: pd.DataFrame, y_test: np.ndarray) -> None:
    global train_x, train_y
    try:
        if name == "TabFM":
            try:
                from tabfm import TabFMRegressor, tabfm_v1_0_0_pytorch as backend
            except ImportError:
                # Compatibility with the import paths used in the notebook.
                from tabfm.src.classifier_and_regressor import TabFMRegressor
                from tabfm.src.pytorch import tabfm_v1_0_0 as backend
            # Mirror the notebook's GPU/FP16 attempt, then retry on CPU/FP32
            # if initialization or held-out inference fails.
            devices = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
            last_error: Exception | None = None
            for device in devices:
                try:
                    base = backend.load(
                        model_type="regression", device=device,
                        dtype=torch.float16 if device == "cuda" else torch.float32,
                    )
                    model = TabFMRegressor(
                        model=base,
                        max_num_rows=min(100, len(x_train)),
                        max_num_features=min(500, x_train.shape[1]),
                    )
                    # Preserve this model's own context for later predictions.
                    train_x, train_y = x_train, y_train
                    model.fit(x_train, y_train)
                    pred = np.asarray(model.predict(x_test), dtype=float).reshape(-1)
                    if len(pred) != len(y_test) or not np.isfinite(pred).all():
                        raise ValueError("Held-out predictions were invalid")
                    models[name] = model
                    metrics[name] = {
                        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
                        "MAE": float(mean_absolute_error(y_test, pred)),
                        "R2": float(r2_score(y_test, pred)),
                    }
                    statuses[name] = {"ready": True, "message": f"Ready ({device})"}
                    log.info("TabFM initialized on %s and evaluated on %d held-out rows", device, len(y_test))
                    return
                except Exception as exc:
                    last_error = exc
                    log.exception("TabFM %s initialization/evaluation failed", device)
            raise RuntimeError(f"TabFM failed on all available devices: {last_error}")
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            from tabpfn import TabPFNRegressor
            model = TabPFNRegressor(device=device)
            model.fit(x_train, y_train)

        pred = np.asarray(model.predict(x_test), dtype=float).reshape(-1)
        if len(pred) != len(y_test) or not np.isfinite(pred).all():
            raise ValueError("Held-out predictions had an invalid shape or values")
        models[name] = model
        metrics[name] = {
            "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
            "MAE": float(mean_absolute_error(y_test, pred)),
            "R2": float(r2_score(y_test, pred)),
        }
        statuses[name] = {"ready": True, "message": "Ready"}
        log.info("%s initialized and evaluated on %d held-out rows", name, len(y_test))
    except Exception as exc:
        # Keep details in the server log; status contains a concise message.
        log.exception("%s initialization failed", name)
        statuses[name] = {"ready": False, "message": f"Initialization failed: {type(exc).__name__}: {exc}"}


@app.on_event("startup")
def initialize() -> None:
    global imputer, metadata, train_x, train_y
    try:
        df, x, y = _load_data()
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
        "evaluation": {"protocol": "single 80/20 random holdout", "test_size": 0.2, "random_state": 42},
    }
    idx_train, idx_test = train_test_split(np.arange(len(x)), test_size=0.2, random_state=42)
    x_train, x_test = x.iloc[idx_train].copy(), x.iloc[idx_test].copy()
    y_train, y_test = y.iloc[idx_train].to_numpy(dtype=np.float32), y.iloc[idx_test].to_numpy(dtype=float)
    imputer = SimpleImputer(strategy="median")
    x_train.loc[:, NUMERIC] = imputer.fit_transform(x_train[NUMERIC])
    x_test.loc[:, NUMERIC] = imputer.transform(x_test[NUMERIC])
    for col in CATEGORICAL:
        x_train[col] = x_train[col].fillna("Unknown").astype(str)
        x_test[col] = x_test[col].fillna("Unknown").astype(str)

    # The models initialize independently, so one missing package/license does
    # not suppress the other model's genuine results.
    _fit_and_evaluate("TabFM", x_train, y_train, x_test, y_test)
    _fit_and_evaluate("TabPFN", x_train, y_train, x_test, y_test)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"service": "ok" if metadata.get("dataset_loaded") else "degraded", "dataset": metadata, "models": statuses}


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
        raise HTTPException(status_code=503, detail="Prediction service is not ready: dataset could not be loaded")
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
                # Both model contexts were established once at service startup.
                pred = np.asarray(model.predict(row), dtype=float).reshape(-1)
                if len(pred) != 1 or not np.isfinite(pred).all():
                    raise ValueError("Model returned invalid prediction")
                results[name] = {"prediction_tonnes_ha": float(pred[0]), "status": "ok"}
            except Exception as exc:
                log.exception("%s prediction failed", name)
                if name == "TabFM" and torch.cuda.is_available():
                    try:
                        try:
                            from tabfm import TabFMRegressor, tabfm_v1_0_0_pytorch as backend
                        except ImportError:
                            from tabfm.src.classifier_and_regressor import TabFMRegressor
                            from tabfm.src.pytorch import tabfm_v1_0_0 as backend
                        base = backend.load(model_type="regression", device="cpu", dtype=torch.float32)
                        cpu_model = TabFMRegressor(
                            model=base, max_num_rows=min(100, len(train_x)),
                            max_num_features=min(500, train_x.shape[1]),
                        )
                        cpu_model.fit(train_x, train_y)
                        cpu_pred = np.asarray(cpu_model.predict(row), dtype=float).reshape(-1)
                        if len(cpu_pred) != 1 or not np.isfinite(cpu_pred).all():
                            raise ValueError("CPU fallback returned invalid prediction")
                        models[name] = cpu_model
                        statuses[name] = {"ready": True, "message": "Ready (CPU fallback)"}
                        results[name] = {"prediction_tonnes_ha": float(cpu_pred[0]), "status": "ok", "device": "cpu"}
                        continue
                    except Exception as cpu_exc:
                        log.exception("TabFM CPU prediction fallback failed")
                        results[name] = {"status": "error", "message": f"Prediction failed on GPU and CPU ({type(cpu_exc).__name__})"}
                else:
                    results[name] = {"status": "error", "message": f"Prediction failed: {type(exc).__name__}"}
    if not results:
        raise HTTPException(status_code=503, detail="Neither model is currently available")
    return {"results": results, "unit": "tonnes/ha", "synthetic": metadata.get("synthetic", True)}
