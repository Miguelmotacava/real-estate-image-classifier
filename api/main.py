"""FastAPI service exposing the real-estate scene classifier."""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Make ``src`` importable when launched as `uvicorn api.main:app` from repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.inference import BUSINESS_LABELS, LoadedModel, load_model, predict_image  # noqa: E402
from api.schemas import (  # noqa: E402
    BatchPredictionResponse,
    ClassesResponse,
    ClassInfo,
    HealthResponse,
    PredictionResponse,
)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp"}
MAX_BYTES = 8 * 1024 * 1024  # 8 MB

app = FastAPI(
    title="Real Estate Image Classifier API",
    description=(
        "API de clasificaci\u00f3n autom\u00e1tica de im\u00e1genes inmobiliarias para "
        "marketplaces. Devuelve clase, top-3 con confianza y metadatos del modelo."
    ),
    version="1.0.0",
    contact={"name": "Equipo ML", "email": "ml@example.com"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_state: dict = {"model": None}


def _get_loaded() -> LoadedModel:
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado a\u00fan")
    return _state["model"]


@app.on_event("startup")
def _startup() -> None:
    try:
        _state["model"] = load_model()
        print(
            f"[startup] modelo {_state['model'].model_name} cargado desde "
            f"{_state['model'].checkpoint_path} en {_state['model'].device}"
        )
    except FileNotFoundError as exc:
        # Service still starts so /health can report degraded state.
        print(f"[startup] no hay modelo entrenado disponible: {exc}")
        _state["model"] = None


def _validate_upload(file: UploadFile, contents: bytes) -> None:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Tipo de contenido no soportado: {file.content_type}",
        )
    if len(contents) > MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Imagen demasiado grande (max 8 MB)",
        )


@app.get("/", include_in_schema=False)
def root():
    return {"service": "real-estate-image-classifier", "docs": "/docs"}


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Estado del servicio y del modelo cargado",
)
def health():
    loaded = _state["model"]
    if loaded is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "model": "none",
                "classes": [],
                "device": "n/a",
                "image_size": 0,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
        )
    return HealthResponse(
        status="ok",
        model=loaded.model_name,
        classes=loaded.classes,
        device=str(loaded.device),
        image_size=loaded.image_size,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@app.get(
    "/classes",
    response_model=ClassesResponse,
    summary="Cat\u00e1logo de clases con su mapeo de negocio",
)
def classes():
    loaded = _get_loaded()
    items: List[ClassInfo] = []
    for cls in loaded.classes:
        family, business_label = BUSINESS_LABELS.get(cls, ("(sin mapear)", cls))
        items.append(ClassInfo(name=cls, family=family, business_label=business_label))
    return ClassesResponse(classes=items, total=len(items))


@app.post(
    "/predict",
    response_model=PredictionResponse,
    response_model_by_alias=True,
    summary="Clasifica una imagen individual",
)
async def predict(file: UploadFile = File(..., description="Imagen JPG/PNG/WEBP")):
    contents = await file.read()
    _validate_upload(file, contents)
    loaded = _get_loaded()
    try:
        result = predict_image(loaded, contents)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PredictionResponse(filename=file.filename, **result)


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Clasifica varias im\u00e1genes en una sola llamada",
)
async def predict_batch(files: List[UploadFile] = File(...)):
    loaded = _get_loaded()
    results: list[PredictionResponse] = []
    errors: list[str] = []
    failed = 0
    for file in files:
        contents = await file.read()
        try:
            _validate_upload(file, contents)
            payload = predict_image(loaded, contents)
            results.append(PredictionResponse(filename=file.filename, **payload))
        except HTTPException as exc:
            failed += 1
            errors.append(f"{file.filename}: {exc.detail}")
        except ValueError as exc:
            failed += 1
            errors.append(f"{file.filename}: {exc}")
    return BatchPredictionResponse(results=results, total=len(files), failed=failed, errors=errors)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
