"""Model loading and inference helper used by the FastAPI app.

Supports a registry of selectable models:
  - One or more single backbones (Swin-L FINAL, F3 ConvNeXtV2-L, F4 EVA02-B, ...).
  - One ensemble model defined by ``models/exp_FINAL_ensemble_9010/ensemble.json``
    that combines several single backbones via soft-voting with hflip TTA.

The API picks the default ("FINAL" — Swin-Large single) on startup, but the
client can switch via the ``model`` query parameter.
"""
from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
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
    "Living room": ("Interior", "Salón"),
    "Office": ("Interior", "Despacho / Oficina"),
    "Store": ("Interior", "Local comercial"),
    "Industrial": ("Interior", "Nave industrial"),
    "Inside city": ("Exterior urbano", "Vista urbana interior"),
    "Tall building": ("Exterior urbano", "Edificio en altura"),
    "Street": ("Exterior urbano", "Calle"),
    "Suburb": ("Exterior urbano", "Suburbio / Adosados"),
    "Highway": ("Exterior urbano", "Carretera / Vía rápida"),
    "Coast": ("Entorno natural", "Costa / Mar"),
    "Forest": ("Entorno natural", "Bosque"),
    "Mountain": ("Entorno natural", "Montaña"),
    "Open country": ("Entorno natural", "Campo abierto"),
}


@dataclass
class SingleModel:
    """A single backbone with its transform + metadata."""
    name: str
    label: str
    model: torch.nn.Module
    transform: transforms.Compose
    image_size: int
    backbone: str
    device: torch.device
    checkpoint_path: Path
    val_accuracy: float
    short_description: str


@dataclass
class EnsembleModel:
    """Soft-voting ensemble of multiple SingleModel instances."""
    name: str
    label: str
    members: list[SingleModel]
    weights: list[float]
    device: torch.device
    short_description: str
    tta: str = "hflip"  # "hflip" or "multiscale_30view"
    scales: tuple = (0.82, 0.88, 0.94)


@dataclass
class ModelRegistry:
    singles: dict[str, SingleModel]
    ensemble: Optional[EnsembleModel]
    default_name: str
    classes: list[str]

    def list_options(self) -> list[dict]:
        """Return a UI-friendly catalog of selectable models."""
        opts: list[dict] = []
        for name, m in self.singles.items():
            opts.append({
                "name": name,
                "label": m.label,
                "type": "single",
                "backbone": m.backbone,
                "image_size": m.image_size,
                "val_accuracy": m.val_accuracy,
                "description": m.short_description,
                "is_default": name == self.default_name,
            })
        if self.ensemble is not None:
            opts.append({
                "name": self.ensemble.name,
                "label": self.ensemble.label,
                "type": "ensemble",
                "backbone": ", ".join(m.backbone for m in self.ensemble.members),
                "image_size": max(m.image_size for m in self.ensemble.members),
                "val_accuracy": None,
                "description": self.ensemble.short_description,
                "is_default": self.ensemble.name == self.default_name,
            })
        return opts

    def get(self, name: str | None) -> SingleModel | EnsembleModel:
        target = name or self.default_name
        if target in self.singles:
            return self.singles[target]
        if self.ensemble is not None and target == self.ensemble.name:
            return self.ensemble
        raise KeyError(f"Modelo desconocido: {target}")


def _eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _load_classes() -> list[str]:
    classes_path = ROOT / "reports" / "eda_summary.json"
    return json.loads(classes_path.read_text(encoding="utf-8"))["classes"]


def _load_single_from_ckpt(
    name: str,
    label: str,
    short_description: str,
    checkpoint_path: Path,
    device: torch.device,
    val_accuracy: float = 0.0,
) -> SingleModel:
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
    img_size = int(config.get("image_size", 224))
    return SingleModel(
        name=name,
        label=label,
        model=model,
        transform=_eval_transform(img_size),
        image_size=img_size,
        backbone=config["model_name"],
        device=device,
        checkpoint_path=checkpoint_path,
        val_accuracy=val_accuracy,
        short_description=short_description,
    )


# Manifest of known production-ready models. Only entries whose checkpoint
# is present on disk are loaded. The order also defines the dropdown order.
_MANIFEST = [
    {
        "name": "FINAL",
        "exp": "exp_FINAL_swin_large_384_9010",
        "label": "Swin-Large 384 (FINAL, recomendado)",
        "description": "Single backbone Swin-L 384 reentrenado con 90/10 split. ~90 ms/imagen, val_acc 0.9866.",
    },
    {
        "name": "F3",
        "exp": "exp_FINAL_F3_9010",
        "label": "ConvNeXtV2-Large 288 (F3)",
        "description": "CNN moderno preentrenado FCMAE+IN22K. Buena precisión con menor coste.",
    },
    {
        "name": "F4",
        "exp": "exp_FINAL_F4_9010",
        "label": "EVA02-Base 448 (F4)",
        "description": "ViT con MIM, resolución 448. Discrimina texturas finas.",
    },
    {
        "name": "F8",
        "exp": "exp_FINAL_F8_9010",
        "label": "BEiT-Large 224 (F8)",
        "description": "Transformer con MIM clásico, pequeña huella de imagen.",
    },
]


def discover_best_checkpoint(models_dir: Path = DEFAULT_MODELS_DIR) -> Path:
    """Pick the production checkpoint, falling back to highest test_accuracy.

    An experiment whose directory name starts with ``exp_FINAL`` (and is NOT
    the ensemble container) is treated as the designated production model.
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
        if summary_path.parent.name == "exp_FINAL_swin_large_384_9010":
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


def load_model(checkpoint_path: Path | None = None, device: torch.device | None = None) -> SingleModel:
    """Backward-compatible single-model loader (used by tests/scripts)."""
    device = device or detect_device(verbose=False)
    if checkpoint_path is None:
        checkpoint_path = discover_best_checkpoint()
    name = checkpoint_path.parent.name.replace("exp_", "")
    return _load_single_from_ckpt(
        name=name, label=name, short_description=name,
        checkpoint_path=checkpoint_path, device=device,
    )


def load_registry(models_dir: Path = DEFAULT_MODELS_DIR) -> ModelRegistry:
    """Load every known model present on disk plus the ensemble if defined."""
    device = detect_device(verbose=False)
    classes = _load_classes()
    singles: dict[str, SingleModel] = {}
    for entry in _MANIFEST:
        ckpt_path = models_dir / entry["exp"] / "best_model.pt"
        summary_path = models_dir / entry["exp"] / "summary.json"
        if not ckpt_path.exists():
            continue
        val_acc = 0.0
        if summary_path.exists():
            try:
                data = json.loads(summary_path.read_text(encoding="utf-8"))
                val_acc = float(data.get("final_val_accuracy_tta") or data.get("test_accuracy") or 0.0)
            except (json.JSONDecodeError, ValueError):
                pass
        try:
            singles[entry["name"]] = _load_single_from_ckpt(
                name=entry["name"], label=entry["label"],
                short_description=entry["description"],
                checkpoint_path=ckpt_path, device=device, val_accuracy=val_acc,
            )
            print(f"[registry] loaded {entry['name']}: {entry['label']} (val={val_acc:.4f})")
        except Exception as exc:  # noqa: BLE001
            print(f"[registry] FAILED to load {entry['name']}: {exc}")

    ensemble: EnsembleModel | None = None
    ensemble_cfg_path = models_dir / "exp_FINAL_ensemble_9010" / "ensemble.json"
    if ensemble_cfg_path.exists():
        try:
            cfg = json.loads(ensemble_cfg_path.read_text(encoding="utf-8"))
            members: list[SingleModel] = []
            weights: list[float] = []
            for m in cfg["members"]:
                # Find matching single by exp dir name
                exp = m["exp"]
                short = next(
                    (e["name"] for e in _MANIFEST if e["exp"] == exp),
                    exp.replace("exp_FINAL_", "").replace("_9010", ""),
                )
                if short in singles:
                    members.append(singles[short])
                    weights.append(float(m["weight"]))
            if members:
                ensemble_summary = models_dir / "exp_FINAL_ensemble_9010" / "summary.json"
                val_acc = 0.0
                if ensemble_summary.exists():
                    try:
                        es = json.loads(ensemble_summary.read_text(encoding="utf-8"))
                        val_acc = float(es.get("val_accuracy") or 0.0)
                    except (json.JSONDecodeError, ValueError):
                        pass
                tta_mode = cfg.get("tta", "hflip")
                scales = tuple(cfg.get("scales", (0.82, 0.88, 0.94)))
                latency_factor = (len(members) * 30) if tta_mode == "multiscale_30view" else (len(members) * 2)
                latency_ms = latency_factor * 90  # rough: ~90ms per single forward
                ensemble = EnsembleModel(
                    name="ensemble",
                    label=(
                        f"Ensemble 4-way "
                        f"({'multi-scale 30-view' if tta_mode == 'multiscale_30view' else 'hflip'}"
                        f", ~{latency_ms/1000:.1f}s)"
                    ),
                    members=members,
                    weights=weights,
                    device=device,
                    short_description=(
                        f"Soft-voting de {len(members)} backbones (90/10), val_acc {val_acc:.4f}. "
                        f"TTA: {tta_mode}. Máxima accuracy, latencia ~{latency_factor}× vs single."
                    ),
                    tta=tta_mode,
                    scales=scales,
                )
                print(f"[registry] loaded ensemble with {len(members)} members, val={val_acc:.4f}")
        except Exception as exc:  # noqa: BLE001
            print(f"[registry] FAILED to load ensemble: {exc}")

    if not singles:
        raise FileNotFoundError(f"No trained model found under {models_dir}.")
    default = "FINAL" if "FINAL" in singles else next(iter(singles))
    return ModelRegistry(singles=singles, ensemble=ensemble, default_name=default, classes=classes)


def _predict_single(loaded: SingleModel, image: Image.Image) -> np.ndarray:
    tensor = loaded.transform(image).unsqueeze(0).to(loaded.device)
    with torch.no_grad():
        p1 = F.softmax(loaded.model(tensor), dim=1)
        p2 = F.softmax(loaded.model(torch.flip(tensor, dims=[3])), dim=1)
        proba = ((p1 + p2) / 2)[0].cpu().numpy()
    return proba


def _predict_single_multiscale(loaded: SingleModel, image: Image.Image, scales: tuple) -> np.ndarray:
    """3 scales × 5 crops × 2 flips = 30 views per image (matches the ensemble script)."""
    tensor = loaded.transform(image).unsqueeze(0).to(loaded.device)
    full = loaded.image_size
    agg = None
    n_views = 0
    with torch.no_grad():
        for scale in scales:
            crop = int(full * scale)
            pad = full - crop
            offsets = [
                (pad // 2, pad // 2),
                (0, 0), (0, pad), (pad, 0), (pad, pad),
            ]
            for oy, ox in offsets:
                patch = tensor[:, :, oy:oy + crop, ox:ox + crop]
                patch = F.interpolate(patch, size=full, mode="bilinear", align_corners=False)
                p1 = F.softmax(loaded.model(patch), dim=1)
                p2 = F.softmax(loaded.model(torch.flip(patch, dims=[3])), dim=1)
                agg = (p1 + p2) if agg is None else agg + p1 + p2
                n_views += 2
    return (agg / n_views)[0].cpu().numpy()


def _predict_ensemble(ens: EnsembleModel, image: Image.Image) -> np.ndarray:
    probas = []
    use_multiscale = ens.tta == "multiscale_30view"
    for m in ens.members:
        if use_multiscale:
            probas.append(_predict_single_multiscale(m, image, ens.scales))
        else:
            probas.append(_predict_single(m, image))
    agg = np.zeros_like(probas[0])
    for w, p in zip(ens.weights, probas):
        agg += w * p
    return agg


def predict_image(
    model_or_registry: SingleModel | EnsembleModel | ModelRegistry,
    image_bytes: bytes,
    top_k: int = 3,
    model_name: str | None = None,
    classes: list[str] | None = None,
) -> dict:
    """Run inference on a single image and return prediction payload.

    Accepts either a SingleModel/EnsembleModel directly (legacy callers)
    or a ModelRegistry + ``model_name`` (new API).
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Imagen inválida: {exc}") from exc

    if isinstance(model_or_registry, ModelRegistry):
        target = model_or_registry.get(model_name)
        cls_list = model_or_registry.classes
    else:
        target = model_or_registry
        cls_list = classes if classes is not None else _load_classes()

    started = time.perf_counter()
    if isinstance(target, EnsembleModel):
        proba = _predict_ensemble(target, image)
        model_used = f"ensemble[{','.join(m.name for m in target.members)}]"
        device_str = str(target.device)
    else:
        proba = _predict_single(target, image)
        model_used = target.backbone
        device_str = str(target.device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    top_indices = proba.argsort()[::-1][:top_k]
    top3 = []
    for idx in top_indices:
        cls = cls_list[int(idx)]
        _, business_label = BUSINESS_LABELS.get(cls, ("(sin mapear)", cls))
        top3.append(
            {
                "label": cls,
                "business_label": business_label,
                "confidence": float(proba[int(idx)]),
            }
        )

    main_cls = cls_list[int(top_indices[0])]
    _, main_business = BUSINESS_LABELS.get(main_cls, ("(sin mapear)", main_cls))
    return {
        "class": main_cls,
        "business_label": main_business,
        "confidence": float(proba[int(top_indices[0])]),
        "top3": top3,
        "model_used": model_used,
        "model_alias": (target.name if hasattr(target, "name") else "n/a"),
        "inference_device": device_str,
        "inference_time_ms": round(elapsed_ms, 2),
    }


# Backward-compat alias used by older imports
LoadedModel = SingleModel
