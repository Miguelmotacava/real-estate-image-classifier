"""Train one ensemble member on the 90/10 split.

Usage:
    python -m src.experiments.final_member_9010 --member F3
    python -m src.experiments.final_member_9010 --member F4
    python -m src.experiments.final_member_9010 --member F8

Each preset uses the same recipe as the original F-series benchmark
(F3, F4, F8) but with the 90/10 split — keeps train+val accuracy aligned
with the FINAL Swin-L model so they can be ensembled cleanly.

Caps at 8 epochs with patience 3 to fit within a 4h sequential budget.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import wandb
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from src.experiments.final_9010 import split_9010, set_seeds
from src.utils.data import (
    SceneImageDataset,
    build_transforms,
    discover_dataset,
)
from src.utils.device import detect_device
from src.utils.metrics import (
    collect_predictions,
    confusion_matrix_figure,
    per_class_report,
    roc_curves_figure,
)
from src.utils.models import build_model, differential_lr_param_groups
from src.utils.train_loop import TrainConfig, fit

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"

PRESETS = {
    "F3": {
        "model_name": "convnextv2_large_22k",
        "image_size": 288,
        "batch_size": 8,
        "lr": 7e-4,
        "backbone_lr_factor": 0.05,
        "drop_path": 0.1,
    },
    "F4": {
        "model_name": "eva02_base_448",
        "image_size": 448,
        "batch_size": 4,
        "lr": 5e-4,
        "backbone_lr_factor": 0.05,
        "drop_path": 0.1,
    },
    "F8": {
        "model_name": "beit_large_224",
        "image_size": 224,
        "batch_size": 8,
        "lr": 5e-4,
        "backbone_lr_factor": 0.05,
        "drop_path": 0.1,
    },
}


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--member", required=True, choices=list(PRESETS))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    args = parser.parse_args()

    preset = PRESETS[args.member]
    EXPERIMENT = f"exp_FINAL_{args.member}_9010"
    set_seeds(42)
    device = detect_device(verbose=True)

    paths, labels, classes = discover_dataset()
    split = split_9010(paths, labels, classes)
    print(f"[{args.member}] Samples: train={len(split.train_paths)}, val={len(split.val_paths)}", flush=True)

    img_size = preset["image_size"]
    bs = preset["batch_size"]
    train_tfm = build_transforms(img_size, augment=True, strong=False)
    eval_tfm = build_transforms(img_size, augment=False)
    train_ds = SceneImageDataset(split.train_paths, split.train_labels, train_tfm)
    val_ds = SceneImageDataset(split.val_paths, split.val_labels, eval_tfm)
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, drop_last=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(
        preset["model_name"], num_classes=len(classes), pretrained=True,
        drop_rate=0.1, drop_path_rate=preset["drop_path"],
    ).to(device)

    cfg = TrainConfig(
        experiment=EXPERIMENT,
        model_name=preset["model_name"],
        num_classes=len(classes),
        epochs=args.epochs,
        batch_size=bs,
        image_size=img_size,
        learning_rate=preset["lr"],
        backbone_lr_factor=preset["backbone_lr_factor"],
        weight_decay=1e-4,
        optimizer="adamw",
        label_smoothing=0.1,
        dropout=0.1,
        early_stopping_patience=args.patience,
        scheduler="cosine",
        cosine_warmup_epochs=2,
        use_ema=True,
        ema_decay=0.999,
        transfer_strategy="fine_tuning",
    )

    param_groups = differential_lr_param_groups(
        model, cfg.learning_rate, cfg.backbone_lr_factor,
    )

    output_dir = MODELS_DIR / EXPERIMENT
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        wandb.init(
            entity=os.getenv("WANDB_ENTITY"),
            project=os.getenv("WANDB_PROJECT"),
            name=EXPERIMENT,
            config={**asdict(cfg), "split_strategy": "90/10",
                    "n_train": len(split.train_paths), "n_val": len(split.val_paths)},
            tags=["transfer-learning", "real-estate", "final", "ensemble-member", args.member],
            reinit=True,
        )
    except Exception as exc:
        print(f"[{args.member}] W&B init failed (continuing without): {exc}", flush=True)

    best_val_acc, best_path, history = fit(
        model, train_loader, val_loader, cfg, device, output_dir,
        param_groups=param_groups, use_wandb=(wandb.run is not None),
    )

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    y_true, y_pred, y_proba = collect_predictions(model, val_loader, device, tta=True)
    report = per_class_report(y_true, y_pred, classes)
    val_acc = float((y_true == y_pred).mean())

    cm_path = output_dir / "confusion_matrix.png"
    confusion_matrix_figure(y_true, y_pred, classes).savefig(cm_path, dpi=140, bbox_inches="tight")
    roc_path = output_dir / "roc_curves.png"
    roc_curves_figure(y_true, y_proba, classes).savefig(roc_path, dpi=140, bbox_inches="tight")

    (output_dir / "val_metrics.json").write_text(
        json.dumps({"val_accuracy": val_acc, "report": report}, indent=2),
        encoding="utf-8",
    )
    summary = {
        "experiment": EXPERIMENT,
        "model": preset["model_name"],
        "transfer_strategy": "fine_tuning",
        "split": "90/10",
        "best_val_accuracy": best_val_acc,
        "final_val_accuracy_tta": val_acc,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "test_accuracy": val_acc,  # proxy
        "epochs_run": len(history),
        "image_size": img_size,
        "ensemble_member": args.member,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[{args.member}] DONE val_acc={val_acc:.4f} F1={report['macro avg']['f1-score']:.4f}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)

    if wandb.run is not None:
        wandb.summary["val_accuracy"] = val_acc
        wandb.summary["macro_f1"] = report["macro avg"]["f1-score"]
        wandb.log({"confusion_matrix": wandb.Image(str(cm_path))})
        wandb.log({"roc_curves": wandb.Image(str(roc_path))})
        wandb.finish()


if __name__ == "__main__":
    main()
