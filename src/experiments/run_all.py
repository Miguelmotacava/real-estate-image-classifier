"""Run a curated CPU-friendly experiment list back-to-back.

Each call mirrors what ``run_experiment`` does so we can keep configuration
co-located. Image size is reduced to 160x160 and the training subset is capped
to keep epochs under ~3 minutes each on the available CPU.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import wandb
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.data import build_loaders, discover_dataset, stratified_split
from src.utils.device import detect_device
from src.utils.metrics import (
    collect_predictions,
    confusion_matrix_figure,
    misclassified_samples,
    per_class_report,
    roc_curves_figure,
)
from src.utils.models import (
    build_model,
    differential_lr_param_groups,
    freeze_backbone,
    unfreeze_last_n,
)
from src.utils.train_loop import TrainConfig, fit

MODELS_DIR = ROOT / "models"
SEED = 42
IMAGE_SIZE = 160  # reduced for CPU
TRAIN_SUBSET = 1600  # cap training samples to keep wall time reasonable
EPOCHS = 2

EXPERIMENTS = [
    {
        "experiment": "exp_A1_scratch_cnn",
        "model": "scratch_cnn",
        "transfer": "fine_tuning",
        "lr": 1e-3,
        "pretrained": False,
    },
    {
        "experiment": "exp_A2_mobilenetv3_small",
        "model": "mobilenetv3_small_100",
        "transfer": "feature_extraction",
        "lr": 5e-4,
        "pretrained": True,
    },
    {
        "experiment": "exp_A3_efficientnet_b0",
        "model": "efficientnet_b0",
        "transfer": "fine_tuning",
        "lr": 7e-4,
        "pretrained": True,
    },
]


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _maybe_subset(split, max_train: int):
    if max_train is None or max_train >= len(split.train_paths):
        return split
    indices = list(range(len(split.train_paths)))
    random.Random(SEED).shuffle(indices)
    indices = indices[:max_train]
    return split.__class__(
        train_paths=[split.train_paths[i] for i in indices],
        train_labels=[split.train_labels[i] for i in indices],
        val_paths=split.val_paths,
        val_labels=split.val_labels,
        test_paths=split.test_paths,
        test_labels=split.test_labels,
        classes=split.classes,
    )


def _setup_transfer(model, strategy, head_lr, backbone_lr_factor=0.1):
    if strategy == "feature_extraction":
        freeze_backbone(model)
        return None
    if strategy == "partial":
        freeze_backbone(model)
        unfreeze_last_n(model, n_blocks=15)
        return None
    if strategy == "fine_tuning":
        return differential_lr_param_groups(model, head_lr, backbone_lr_factor)
    raise ValueError(strategy)


def _evaluate(model, test_loader, classes, device, output_dir, use_wandb):
    y_true, y_pred, y_proba = collect_predictions(model, test_loader, device)
    report = per_class_report(y_true, y_pred, classes)
    test_acc = float((y_true == y_pred).mean())

    cm_fig = confusion_matrix_figure(y_true, y_pred, classes)
    cm_path = output_dir / "confusion_matrix.png"
    cm_fig.savefig(cm_path, dpi=140, bbox_inches="tight")
    roc_fig = roc_curves_figure(y_true, y_proba, classes)
    roc_path = output_dir / "roc_curves.png"
    roc_fig.savefig(roc_path, dpi=140, bbox_inches="tight")

    test_paths = [Path(p) for p in test_loader.dataset.paths]
    mistakes = misclassified_samples(y_true, y_pred, test_paths, classes)
    (output_dir / "misclassified.json").write_text(
        json.dumps(mistakes, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "test_metrics.json").write_text(
        json.dumps({"test_accuracy": test_acc, "report": report}, indent=2),
        encoding="utf-8",
    )

    if use_wandb and wandb.run is not None:
        wandb.summary["test_accuracy"] = test_acc
        wandb.summary["macro_f1"] = report["macro avg"]["f1-score"]
        wandb.summary["weighted_f1"] = report["weighted avg"]["f1-score"]
        wandb.log({"confusion_matrix": wandb.Image(str(cm_path))})
        wandb.log({"roc_curves": wandb.Image(str(roc_path))})
        per_class_table = wandb.Table(columns=["class", "precision", "recall", "f1", "support"])
        for cls in classes:
            r = report[cls]
            per_class_table.add_data(cls, r["precision"], r["recall"], r["f1-score"], r["support"])
        wandb.log({"per_class_metrics": per_class_table})
    return {"test_accuracy": test_acc, "report": report}


def run_one(spec: dict, split, classes, device, use_wandb: bool = True) -> dict:
    """Execute a single experiment and return its summary."""
    _set_seeds(SEED)
    output_dir = MODELS_DIR / spec["experiment"]
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(
        spec["model"],
        num_classes=len(classes),
        pretrained=spec["pretrained"],
        drop_rate=0.2,
    ).to(device)
    param_groups = _setup_transfer(model, spec["transfer"], spec["lr"])

    cfg = TrainConfig(
        experiment=spec["experiment"],
        model_name=spec["model"],
        num_classes=len(classes),
        epochs=EPOCHS,
        batch_size=16,
        image_size=IMAGE_SIZE,
        learning_rate=spec["lr"],
        backbone_lr_factor=0.1,
        weight_decay=1e-4,
        optimizer="adamw",
        label_smoothing=0.1,
        dropout=0.2,
        early_stopping_patience=2,
        scheduler_patience=1,
        transfer_strategy=spec["transfer"],
    )

    train_loader, val_loader, test_loader = build_loaders(
        split,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=0,
        pin_memory=False,
    )

    if use_wandb:
        wandb.init(
            entity=os.getenv("WANDB_ENTITY"),
            project=os.getenv("WANDB_PROJECT"),
            name=spec["experiment"],
            config=asdict(cfg),
            tags=["transfer-learning", "real-estate", spec["transfer"]],
            reinit=True,
        )

    t0 = time.time()
    best_val_acc, best_path, history = fit(
        model, train_loader, val_loader, cfg, device, output_dir,
        param_groups=param_groups, use_wandb=use_wandb,
    )
    train_time = time.time() - t0

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    metrics = _evaluate(model, test_loader, classes, device, output_dir, use_wandb)

    summary = {
        "experiment": spec["experiment"],
        "model": spec["model"],
        "transfer_strategy": spec["transfer"],
        "best_val_accuracy": best_val_acc,
        "test_accuracy": metrics["test_accuracy"],
        "macro_f1": metrics["report"]["macro avg"]["f1-score"],
        "weighted_f1": metrics["report"]["weighted avg"]["f1-score"],
        "epochs_run": len(history),
        "train_seconds": round(train_time, 1),
        "image_size": IMAGE_SIZE,
        "train_samples_used": len(split.train_paths),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if use_wandb and wandb.run is not None:
        wandb.finish()
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    import argparse

    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--train-subset", type=int, default=TRAIN_SUBSET)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--only", default=None,
                        help="Restrict to a single experiment name")
    args = parser.parse_args()

    global EPOCHS, TRAIN_SUBSET, IMAGE_SIZE
    EPOCHS = args.epochs
    TRAIN_SUBSET = args.train_subset
    IMAGE_SIZE = args.image_size

    device = detect_device(verbose=True)
    paths, labels, classes = discover_dataset()
    full_split = stratified_split(paths, labels, classes)
    split = _maybe_subset(full_split, TRAIN_SUBSET)
    print(
        f"Using split sizes - train: {len(split.train_paths)} val: {len(split.val_paths)} test: {len(split.test_paths)}"
    )

    use_wandb = bool(os.getenv("WANDB_API_KEY"))
    summaries = []
    selected = [s for s in EXPERIMENTS if args.only is None or s["experiment"] == args.only]
    for spec in selected:
        print("\n" + "=" * 80)
        print(f">>> {spec['experiment']} ({spec['model']}, {spec['transfer']})")
        try:
            summary = run_one(spec, split, classes, device, use_wandb=use_wandb)
            summaries.append(summary)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] {spec['experiment']} failed: {exc}")
            if use_wandb and wandb.run is not None:
                wandb.finish(exit_code=1)

    out_path = ROOT / "reports" / "experiments_runtime.json"
    out_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
