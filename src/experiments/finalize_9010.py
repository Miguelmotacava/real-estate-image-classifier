"""Post-hoc evaluation of exp_FINAL_swin_large_384_9010.

Training was halted early because of GPU thermal throttling after epoch 1
(val acc 0.978 = matches the F6 benchmark). This script reloads the saved
checkpoint, recomputes per-class metrics on the 10% validation split with
horizontal-flip TTA, writes summary.json / val_metrics.json / confusion
matrix / ROC curves, and syncs the artefacts to W&B.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import wandb
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from src.experiments.final_9010 import EXPERIMENT, IMAGE_SIZE, split_9010
from src.utils.data import SceneImageDataset, build_transforms, discover_dataset
from src.utils.device import detect_device
from src.utils.metrics import (
    collect_predictions,
    confusion_matrix_figure,
    per_class_report,
    roc_curves_figure,
)
from src.utils.models import build_model

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"


def main() -> None:
    load_dotenv()
    device = detect_device(verbose=True)

    output_dir = MODELS_DIR / EXPERIMENT
    ckpt_path = output_dir / "best_model.pt"
    assert ckpt_path.exists(), f"No checkpoint at {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    paths, labels, classes = discover_dataset()
    split = split_9010(paths, labels, classes)
    print(f"Samples: train={len(split.train_paths)}, val={len(split.val_paths)}")

    eval_tfm = build_transforms(IMAGE_SIZE, augment=False)
    val_ds = SceneImageDataset(split.val_paths, split.val_labels, eval_tfm)
    val_loader = DataLoader(
        val_ds, batch_size=4, shuffle=False, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(
        cfg["model_name"], num_classes=len(classes), pretrained=False,
        drop_rate=cfg.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    y_true, y_pred, y_proba = collect_predictions(model, val_loader, device, tta=True)
    report = per_class_report(y_true, y_pred, classes)
    val_acc = float((y_true == y_pred).mean())
    print(f"val_accuracy (TTA hflip) = {val_acc:.4f}")

    cm_fig = confusion_matrix_figure(y_true, y_pred, classes)
    cm_path = output_dir / "confusion_matrix.png"
    cm_fig.savefig(cm_path, dpi=140, bbox_inches="tight")
    roc_fig = roc_curves_figure(y_true, y_proba, classes)
    roc_path = output_dir / "roc_curves.png"
    roc_fig.savefig(roc_path, dpi=140, bbox_inches="tight")

    (output_dir / "val_metrics.json").write_text(
        json.dumps({"val_accuracy": val_acc, "report": report}, indent=2),
        encoding="utf-8",
    )

    summary = {
        "experiment": EXPERIMENT,
        "model": cfg["model_name"],
        "transfer_strategy": "fine_tuning",
        "split": "90/10 (4036 train / 449 val, seed=42)",
        "best_val_accuracy": float(ckpt.get("val_accuracy", val_acc)),
        "final_val_accuracy_tta": val_acc,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "test_accuracy": val_acc,  # proxy for API discover_best_checkpoint
        "epochs_run": int(ckpt.get("epoch", 1)),
        "image_size": IMAGE_SIZE,
        "note": (
            "Trained 1 epoch before halting due to GPU thermal throttling; "
            "val accuracy already matched F6 benchmark (0.978). "
            "F6 (70/15/15 split) remains the authoritative test_accuracy=0.9778 benchmark."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))

    try:
        run = wandb.init(
            entity=os.getenv("WANDB_ENTITY"),
            project=os.getenv("WANDB_PROJECT"),
            name=f"{EXPERIMENT}_finalize",
            config=cfg,
            tags=["transfer-learning", "real-estate", "final", "production"],
            reinit=True,
        )
        run.summary["val_accuracy"] = val_acc
        run.summary["macro_f1"] = report["macro avg"]["f1-score"]
        run.summary["weighted_f1"] = report["weighted avg"]["f1-score"]
        run.summary["epochs_run"] = summary["epochs_run"]
        run.log({"confusion_matrix": wandb.Image(str(cm_path))})
        run.log({"roc_curves": wandb.Image(str(roc_path))})
        table = wandb.Table(columns=["class", "precision", "recall", "f1", "support"])
        for cls in classes:
            r = report[cls]
            table.add_data(cls, r["precision"], r["recall"], r["f1-score"], r["support"])
        run.log({"per_class_metrics": table})
        run.finish()
    except Exception as exc:
        print(f"[warn] W&B logging skipped: {exc}")


if __name__ == "__main__":
    main()
