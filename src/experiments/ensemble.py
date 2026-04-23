"""Soft-voting ensemble (Exp D1) over the best trained models."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from dotenv import load_dotenv

from api.inference import load_model
from src.utils.data import build_loaders, discover_dataset, stratified_split
from src.utils.metrics import (
    confusion_matrix_figure,
    per_class_report,
    roc_curves_figure,
)

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
ENSEMBLE_DIR = MODELS_DIR / "ensemble"
ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)


def _ranked_summaries() -> list[tuple[float, Path]]:
    pairs = []
    for summary_path in MODELS_DIR.glob("*/summary.json"):
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        ckpt = summary_path.parent / "best_model.pt"
        if ckpt.exists():
            pairs.append((data.get("test_accuracy", 0.0), ckpt))
    pairs.sort(key=lambda x: x[0], reverse=True)
    return pairs


@torch.no_grad()
def _model_proba(loaded, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Return probabilities and ground truth from a loader."""
    loaded.model.eval()
    all_proba: list[np.ndarray] = []
    all_y: list[int] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        proba = F.softmax(loaded.model(x), dim=1).cpu().numpy()
        all_proba.append(proba)
        all_y.extend(y.numpy().tolist())
    return np.concatenate(all_proba, axis=0), np.array(all_y)


def main(top_k: int = 3) -> dict:
    load_dotenv()
    pairs = _ranked_summaries()
    if not pairs:
        raise FileNotFoundError("No trained models found. Run experiments first.")
    selected = pairs[:top_k]
    print("Ensemble members:")
    for acc, ckpt in selected:
        print(f"  {acc:.3f}  {ckpt.parent.name}")

    paths, labels, classes = discover_dataset()
    split = stratified_split(paths, labels, classes)

    proba_sum: np.ndarray | None = None
    y_true: np.ndarray | None = None
    members = []

    for _, ckpt in selected:
        loaded = load_model(checkpoint_path=ckpt)
        members.append(loaded.model_name)
        _, _, test_loader = build_loaders(
            split,
            image_size=loaded.image_size,
            batch_size=32,
            num_workers=0,
            pin_memory=False,
        )
        proba, y = _model_proba(loaded, test_loader, loaded.device)
        proba_sum = proba if proba_sum is None else proba_sum + proba
        y_true = y if y_true is None else y_true

    ensemble_proba = proba_sum / len(selected)
    ensemble_pred = ensemble_proba.argmax(axis=1)
    test_acc = float((y_true == ensemble_pred).mean())
    report = per_class_report(y_true, ensemble_pred, classes)

    cm_fig = confusion_matrix_figure(y_true, ensemble_pred, classes)
    cm_path = ENSEMBLE_DIR / "confusion_matrix.png"
    cm_fig.savefig(cm_path, dpi=140, bbox_inches="tight")
    roc_fig = roc_curves_figure(y_true, ensemble_proba, classes)
    roc_path = ENSEMBLE_DIR / "roc_curves.png"
    roc_fig.savefig(roc_path, dpi=140, bbox_inches="tight")

    summary = {
        "experiment": "exp_D1_soft_voting",
        "members": members,
        "test_accuracy": test_acc,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
    }
    (ENSEMBLE_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (ENSEMBLE_DIR / "test_metrics.json").write_text(
        json.dumps({"test_accuracy": test_acc, "report": report}, indent=2), encoding="utf-8"
    )

    if os.getenv("WANDB_API_KEY"):
        run = wandb.init(
            entity=os.getenv("WANDB_ENTITY"),
            project=os.getenv("WANDB_PROJECT"),
            name="exp_D1_soft_voting",
            config={"members": members, "method": "soft-voting"},
            tags=["ensemble", "real-estate"],
            reinit=True,
        )
        run.summary["test_accuracy"] = test_acc
        run.summary["macro_f1"] = summary["macro_f1"]
        run.log({"confusion_matrix": wandb.Image(str(cm_path))})
        run.log({"roc_curves": wandb.Image(str(roc_path))})
        run.finish()

    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
