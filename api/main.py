"""FastAPI service exposing the real-estate scene classifier."""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Make ``src`` importable when launched as `uvicorn api.main:app` from repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.inference import (  # noqa: E402
    BUSINESS_LABELS,
    EnsembleModel,
    ModelRegistry,
    SingleModel,
    load_registry,
    predict_image,
)
from api.schemas import (  # noqa: E402
    BatchPredictionResponse,
    ClassesResponse,
    ClassInfo,
    HealthResponse,
    ModelOption,
    ModelsResponse,
    PredictionResponse,
)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp"}
MAX_BYTES = 8 * 1024 * 1024  # 8 MB

app = FastAPI(
    title="Real Estate Image Classifier API",
    description=(
        "API de clasificación automática de imágenes inmobiliarias para "
        "marketplaces. Ofrece varios modelos seleccionables vía `?model=`: "
        "FINAL (Swin-Large 384 single, recomendado), modelos individuales "
        "F3/F4/F8 y un ensemble 4-way de máxima accuracy."
    ),
    version="2.0.0",
    contact={"name": "Equipo ML", "email": "ml@example.com"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_state: dict = {"registry": None}


def _get_registry() -> ModelRegistry:
    reg = _state["registry"]
    if reg is None:
        raise HTTPException(status_code=503, detail="Registro de modelos no cargado aún")
    return reg


@app.on_event("startup")
def _startup() -> None:
    try:
        reg = load_registry()
        _state["registry"] = reg
        print(
            f"[startup] {len(reg.singles)} modelo(s) single + ensemble={reg.ensemble is not None}, "
            f"default='{reg.default_name}'"
        )
    except FileNotFoundError as exc:
        print(f"[startup] no hay modelos disponibles: {exc}")
        _state["registry"] = None


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
    return {"service": "real-estate-image-classifier", "docs": "/docs", "models": "/models"}


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Estado del servicio y del modelo cargado",
)
def health():
    reg = _state["registry"]
    if reg is None:
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
    default = reg.get(None)
    if isinstance(default, EnsembleModel):
        device = str(default.device)
        img_size = max(m.image_size for m in default.members)
        backbone_label = "ensemble"
    else:
        device = str(default.device)
        img_size = default.image_size
        backbone_label = default.backbone
    return HealthResponse(
        status="ok",
        model=backbone_label,
        classes=reg.classes,
        device=device,
        image_size=img_size,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@app.get(
    "/classes",
    response_model=ClassesResponse,
    summary="Catálogo de clases con su mapeo de negocio",
)
def classes():
    reg = _get_registry()
    items: List[ClassInfo] = []
    for cls in reg.classes:
        family, business_label = BUSINESS_LABELS.get(cls, ("(sin mapear)", cls))
        items.append(ClassInfo(name=cls, family=family, business_label=business_label))
    return ClassesResponse(classes=items, total=len(items))


@app.get(
    "/models",
    response_model=ModelsResponse,
    summary="Lista de modelos seleccionables vía ?model=",
)
def list_models():
    reg = _get_registry()
    opts = [ModelOption(**o) for o in reg.list_options()]
    return ModelsResponse(models=opts, default=reg.default_name, total=len(opts))


@app.post(
    "/predict",
    response_model=PredictionResponse,
    response_model_by_alias=True,
    summary="Clasifica una imagen. Selecciona modelo con ?model=",
)
async def predict(
    file: UploadFile = File(..., description="Imagen JPG/PNG/WEBP"),
    model: Optional[str] = Query(
        None,
        description="Alias del modelo: FINAL (default), F3, F4, F8, ensemble. "
                    "Ver /models para la lista completa.",
    ),
):
    contents = await file.read()
    _validate_upload(file, contents)
    reg = _get_registry()
    try:
        result = predict_image(reg, contents, model_name=model)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PredictionResponse(filename=file.filename, **result)


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Clasifica varias imágenes con el mismo modelo seleccionado.",
)
async def predict_batch(
    files: List[UploadFile] = File(...),
    model: Optional[str] = Query(None, description="Alias del modelo (ver /models)."),
):
    reg = _get_registry()
    try:
        reg.get(model)  # validate name once before iterating
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    results: list[PredictionResponse] = []
    errors: list[str] = []
    failed = 0
    for f in files:
        contents = await f.read()
        try:
            _validate_upload(f, contents)
            payload = predict_image(reg, contents, model_name=model)
            results.append(PredictionResponse(filename=f.filename, **payload))
        except HTTPException as exc:
            failed += 1
            errors.append(f"{f.filename}: {exc.detail}")
        except ValueError as exc:
            failed += 1
            errors.append(f"{f.filename}: {exc}")
    return BatchPredictionResponse(results=results, total=len(files), failed=failed, errors=errors)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
