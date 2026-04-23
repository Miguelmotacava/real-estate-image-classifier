"""Final production retrain of the best single model on a 90/10 split.

Chooses Swin-Large 384 (F6: best individual test_accuracy in the F-series) and
retrains it on 90% train / 10% val — no test holdout, since the F-series
already benchmarked this architecture against a proper 70/15/15 split.

Logs to W&B as ``exp_FINAL_swin_large_384_9010``. This is the model the
FastAPI service loads in production.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import wandb
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.utils.data import (
    SceneImageDataset,
    SplitIndex,
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

EXPERIMENT = "exp_FINAL_swin_large_384_9010"
MODEL_NAME = "swin_large_384"
IMAGE_SIZE = 384
BATCH_SIZE = 4
EPOCHS = 15
TRAIN_RATIO = 0.90
SEED = 42


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_9010(paths, labels, classes):
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_idx[label] for label in labels])
    paths_arr = np.array([str(p) for p in paths])
    train_paths, val_paths, train_y, val_y = train_test_split(
        paths_arr, y, test_size=1.0 - TRAIN_RATIO, stratify=y, random_state=SEED,
    )
    return SplitIndex(
        train_paths=[Path(p) for p in train_paths],
        train_labels=train_y.tolist(),
        val_paths=[Path(p) for p in val_paths],
        val_labels=val_y.tolist(),
        test_paths=[],
        test_labels=[],
        classes=list(classes),
    )


def main() -> None:
    load_dotenv()
    set_seeds(SEED)
    device = detect_device(verbose=True)

    paths, labels, classes = discover_dataset()
    split = split_9010(paths, labels, classes)
    print(f"Samples: train={len(split.train_paths)}, val={len(split.val_paths)}")

    train_tfm = build_transforms(IMAGE_SIZE, augment=True, strong=False)
    eval_tfm = build_transforms(IMAGE_SIZE, augment=False)
    train_ds = SceneImageDataset(split.train_paths, split.train_labels, train_tfm)
    val_ds = SceneImageDataset(split.val_paths, split.val_labels, eval_tfm)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    model = build_model(
        MODEL_NAME, num_classes=len(classes), pretrained=True,
        drop_rate=0.1, drop_path_rate=0.1,
    ).to(device)

    cfg = TrainConfig(
        experiment=EXPERIMENT,
        model_name=MODEL_NAME,
        num_classes=len(classes),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        learning_rate=7e-4,
        backbone_lr_factor=0.05,
        weight_decay=1e-4,
        optimizer="adamw",
        label_smoothing=0.1,
        dropout=0.1,
        early_stopping_patience=6,
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

    wandb.init(
        entity=os.getenv("WANDB_ENTITY"),
        project=os.getenv("WANDB_PROJECT"),
        name=EXPERIMENT,
        config={**asdict(cfg), "split_strategy": "90/10", "n_train": len(split.train_paths), "n_val": len(split.val_paths)},
        tags=["transfer-learning", "real-estate", "final", "production", "F6-recipe"],
        reinit=True,
    )

    best_val_acc, best_path, history = fit(
        model, train_loader, val_loader, cfg, device, output_dir,
        param_groups=param_groups, use_wandb=True,
    )

    # Reload best and compute per-class metrics on val + train (no aug)
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    y_true, y_pred, y_proba = collect_predictions(model, val_loader, device, tta=True)
    report = per_class_report(y_true, y_pred, classes)
    val_acc = float((y_true == y_pred).mean())

    # Evaluate on the train split with no augmentation (real generalization signal)
    train_eval_ds = SceneImageDataset(split.train_paths, split.train_labels, eval_tfm)
    train_eval_loader = DataLoader(
        train_eval_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    yt_true, yt_pred, _ = collect_predictions(model, train_eval_loader, device, tta=False)
    train_eval_acc = float((yt_true == yt_pred).mean())
    train_report = per_class_report(yt_true, yt_pred, classes)

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
    (output_dir / "train_metrics.json").write_text(
        json.dumps({"train_accuracy_eval": train_eval_acc, "report": train_report}, indent=2),
        encoding="utf-8",
    )

    summary = {
        "experiment": EXPERIMENT,
        "model": MODEL_NAME,
        "transfer_strategy": "fine_tuning",
        "split": "90/10 (no test holdout — see F6 for 70/15/15 benchmark)",
        "best_val_accuracy": best_val_acc,
        "final_val_accuracy_tta": val_acc,
        "train_accuracy_eval": train_eval_acc,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        # Legacy key so the API's discover_best_checkpoint picks this one:
        "test_accuracy": val_acc,
        "epochs_run": len(history),
        "image_size": IMAGE_SIZE,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))

    if wandb.run is not None:
        wandb.summary["val_accuracy"] = val_acc
        wandb.summary["macro_f1"] = report["macro avg"]["f1-score"]
        wandb.summary["weighted_f1"] = report["weighted avg"]["f1-score"]
        wandb.log({"confusion_matrix": wandb.Image(str(cm_path))})
        wandb.log({"roc_curves": wandb.Image(str(roc_path))})
        table = wandb.Table(columns=["class", "precision", "recall", "f1", "support"])
        for cls in classes:
            r = report[cls]
            table.add_data(cls, r["precision"], r["recall"], r["f1-score"], r["support"])
        wandb.log({"per_class_metrics": table})
        wandb.finish()


if __name__ == "__main__":
    main()
