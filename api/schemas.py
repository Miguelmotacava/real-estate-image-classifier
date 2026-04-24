"""Pydantic schemas for the FastAPI service."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ClassInfo(BaseModel):
    name: str = Field(..., description="Etiqueta interna del modelo")
    family: str = Field(..., description="Familia de negocio (Interior, Exterior urbano, ...)")
    business_label: str = Field(..., description="Etiqueta legible para el marketplace")


class TopPrediction(BaseModel):
    label: str = Field(..., description="Clase predicha")
    business_label: str = Field(..., description="Etiqueta legible para el cliente")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Probabilidad asignada por el modelo")


class PredictionResponse(BaseModel):
    filename: str
    class_: str = Field(..., alias="class")
    business_label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    top3: List[TopPrediction]
    model_used: str
    model_alias: str = Field("FINAL", description="Alias seleccionable (FINAL, F3, F4, F8, ensemble, ...)")
    inference_device: str
    inference_time_ms: float

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "filename": "salon_madrid_001.jpg",
                "class": "Living room",
                "business_label": "Sal\u00f3n",
                "confidence": 0.93,
                "top3": [
                    {"label": "Living room", "business_label": "Sal\u00f3n", "confidence": 0.93},
                    {"label": "Bedroom", "business_label": "Dormitorio", "confidence": 0.04},
                    {"label": "Kitchen", "business_label": "Cocina", "confidence": 0.02},
                ],
                "model_used": "mobilenetv3_small_100",
                "inference_device": "cpu",
                "inference_time_ms": 78.4,
            }
        },
    }


class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]
    total: int
    failed: int = 0
    errors: List[str] = []


class HealthResponse(BaseModel):
    status: str
    model: str
    classes: List[str]
    device: str
    image_size: int
    timestamp: str


class ClassesResponse(BaseModel):
    classes: List[ClassInfo]
    total: int


class ModelOption(BaseModel):
    name: str = Field(..., description="Alias para usar en el query param ?model=")
    label: str = Field(..., description="Nombre legible para mostrar en la UI")
    type: str = Field(..., description="single | ensemble")
    backbone: str = Field(..., description="Arquitectura(s) subyacente(s)")
    image_size: int
    val_accuracy: Optional[float] = None
    description: str
    is_default: bool


class ModelsResponse(BaseModel):
    models: List[ModelOption]
    default: str
    total: int


class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None
