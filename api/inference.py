"""Model loading and inference helper used by the FastAPI app."""
from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Imports from training utilities (kept lightweight at import time)
from src.utils.data import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from src.utils.device import detect_device  # noqa: E402
from src.utils.models import build_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = ROOT / "models"

BUSINESS_LABELS = {
    "Bedroom": ("Interior", "Dormitorio"),
    "Kitchen": ("Interior", "Cocina"),
    "Living room": ("Interior", "Sal\u00f3n"),
    "Office": ("Interior", "Despacho / Oficina"),
    "Store": ("Interior", "Local comercial"),
    "Industrial": ("Interior", "Nave industrial"),
    "Inside city": ("Exterior urbano", "Vista urbana interior"),
    "Tall building": ("Exterior urbano", "Edificio en altura"),
    "Street": ("Exterior urbano", "Calle"),
    "Suburb": ("Exterior urbano", "Suburbio / Adosados"),
    "Highway": ("Exterior urbano", "Carretera / V\u00eda r\u00e1pida"),
    "Coast": ("Entorno natural", "Costa / Mar"),
    "Forest": ("Entorno natural", "Bosque"),
    "Mountain": ("Entorno natural", "Monta\u00f1a"),
    "Open country": ("Entorno natural", "Campo abierto"),
}


@dataclass
class LoadedModel:
    model: torch.nn.Module
    classes: list[str]
    image_size: int
    device: torch.device
    model_name: str
    transform: transforms.Compose
    checkpoint_path: Path


def _eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def discover_best_checkpoint(models_dir: Path = DEFAULT_MODELS_DIR) -> Path:
    """Pick the production checkpoint, falling back to the highest test_accuracy.

    An experiment whose directory name starts with ``exp_FINAL`` is treated as
    the designated production model and wins regardless of marginal accuracy
    differences vs benchmark experiments (F-series, E-series, etc.). If no
    FINAL model exists, fall back to the experiment with the highest reported
    ``test_accuracy`` that has a ``best_model.pt`` on disk.
    """
    final_ckpt: Path | None = None
    best_ckpt: Path | None = None
    best_acc = -1.0
    for summary_path in sorted(models_dir.glob("*/summary.json")):
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        ckpt = summary_path.parent / "best_model.pt"
        if not ckpt.exists():
            continue
        if summary_path.parent.name.startswith("exp_FINAL"):
            final_ckpt = ckpt
        acc = data.get("test_accuracy", -1.0)
        if acc > best_acc:
            best_acc = acc
            best_ckpt = ckpt
    if final_ckpt is not None:
        return final_ckpt
    if best_ckpt is None:
        raise FileNotFoundError(
            f"No trained model found under {models_dir}. Run an experiment first."
        )
    return best_ckpt


def load_model(checkpoint_path: Path | None = None, device: torch.device | None = None) -> LoadedModel:
    """Reconstruct the model architecture and load weights from disk."""
    device = device or detect_device(verbose=False)
    if checkpoint_path is None:
        checkpoint_path = discover_best_checkpoint()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = build_model(
        config["model_name"],
        num_classes=config["num_classes"],
        pretrained=False,
        drop_rate=config.get("dropout", 0.2),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    classes_path = ROOT / "reports" / "eda_summary.json"
    classes = json.loads(classes_path.read_text(encoding="utf-8"))["classes"]

    return LoadedModel(
        model=model,
        classes=classes,
        image_size=config.get("image_size", 224),
        device=device,
        model_name=config["model_name"],
        transform=_eval_transform(config.get("image_size", 224)),
        checkpoint_path=checkpoint_path,
    )


def predict_image(loaded: LoadedModel, image_bytes: bytes, top_k: int = 3) -> dict:
    """Run inference on a single image and return prediction payload."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Imagen inv\u00e1lida: {exc}") from exc

    tensor = loaded.transform(image).unsqueeze(0).to(loaded.device)
    started = time.perf_counter()
    with torch.no_grad():
        logits = loaded.model(tensor)
        proba = F.softmax(logits, dim=1)[0].cpu().numpy()
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    top_indices = proba.argsort()[::-1][:top_k]
    top3 = []
    for idx in top_indices:
        cls = loaded.classes[int(idx)]
        _, business_label = BUSINESS_LABELS.get(cls, ("(sin mapear)", cls))
        top3.append(
            {
                "label": cls,
                "business_label": business_label,
                "confidence": float(proba[int(idx)]),
            }
        )

    main_cls = loaded.classes[int(top_indices[0])]
    _, main_business = BUSINESS_LABELS.get(main_cls, ("(sin mapear)", main_cls))
    return {
        "class": main_cls,
        "business_label": main_business,
        "confidence": float(proba[int(top_indices[0])]),
        "top3": top3,
        "model_used": loaded.model_name,
        "inference_device": str(loaded.device),
        "inference_time_ms": round(elapsed_ms, 2),
    }
