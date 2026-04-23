"""Single-experiment runner: train + evaluate + W&B logging.

Usage:
    python -m src.experiments.run_experiment \
        --experiment exp_A2_mobilenetv3_small \
        --model mobilenetv3_small_100 \
        --transfer fine_tuning --epochs 6 --batch-size 16
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

from src.utils.data import build_loaders, discover_dataset, stratified_split
from src.utils.device import detect_device, recommended_batch_size
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

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_transfer(model, strategy: str, head_lr: float, backbone_lr_factor: float):
    """Configure parameter trainability and return optimizer param groups."""
    if strategy == "feature_extraction":
        freeze_backbone(model)
        return None  # plain optimizer over remaining trainable params
    if strategy == "partial":
        freeze_backbone(model)
        unfreeze_last_n(model, n_blocks=15)
        return None
    if strategy == "fine_tuning":
        return differential_lr_param_groups(model, head_lr, backbone_lr_factor)
    raise ValueError(f"Unknown transfer strategy {strategy!r}")


def evaluate_and_log(
    model,
    test_loader,
    classes: list[str],
    device,
    output_dir: Path,
    use_wandb: bool,
    tta: bool = False,
) -> dict:
    """Compute test metrics and push artefacts to W&B."""
    y_true, y_pred, y_proba = collect_predictions(model, test_loader, device, tta=tta)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--transfer", default="fine_tuning",
                        choices=["feature_extraction", "partial", "fine_tuning"])
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr-factor", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train", type=int, default=None,
                        help="Optional cap on training samples (debug/CPU)")
    parser.add_argument("--strong-augment", action="store_true",
                        help="Use TrivialAugmentWide + higher RandomErasing")
    parser.add_argument("--drop-path-rate", type=float, default=0.0,
                        help="Stochastic depth rate (0 = off)")
    parser.add_argument("--scheduler", default="plateau", choices=["plateau", "cosine"])
    parser.add_argument("--use-ema", action="store_true",
                        help="Track EMA of weights and eval with them each epoch")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--cosine-warmup-epochs", type=int, default=1)
    parser.add_argument("--tta", action="store_true",
                        help="Test-time augmentation (horizontal flip average) at eval")
    return parser.parse_args()


def maybe_subset(split, max_train: int | None):
    """Optionally truncate the training subset for fast smoke runs."""
    if max_train is None or max_train >= len(split.train_paths):
        return split
    indices = list(range(len(split.train_paths)))
    random.Random(42).shuffle(indices)
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


def main() -> None:
    load_dotenv()
    args = parse_args()
    set_seeds(42)
    device = detect_device(verbose=True)

    bs = args.batch_size or recommended_batch_size(device)
    output_dir = MODELS_DIR / args.experiment
    output_dir.mkdir(parents=True, exist_ok=True)

    paths, labels, classes = discover_dataset()
    split = stratified_split(paths, labels, classes)
    split = maybe_subset(split, args.max_train)

    pretrained = not args.no_pretrained
    model = build_model(
        args.model,
        num_classes=len(classes),
        pretrained=pretrained,
        drop_rate=args.dropout,
        drop_path_rate=args.drop_path_rate,
    ).to(device)

    param_groups = setup_transfer(model, args.transfer, args.lr, args.backbone_lr_factor)

    cfg = TrainConfig(
        experiment=args.experiment,
        model_name=args.model,
        num_classes=len(classes),
        epochs=args.epochs,
        batch_size=bs,
        image_size=args.image_size,
        learning_rate=args.lr,
        backbone_lr_factor=args.backbone_lr_factor,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        label_smoothing=args.label_smoothing,
        dropout=args.dropout,
        early_stopping_patience=args.patience,
        transfer_strategy=args.transfer,
        scheduler=args.scheduler,
        cosine_warmup_epochs=args.cosine_warmup_epochs,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
    )

    train_loader, val_loader, test_loader = build_loaders(
        split,
        image_size=args.image_size,
        batch_size=bs,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        strong_augment=args.strong_augment,
    )

    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            entity=os.getenv("WANDB_ENTITY"),
            project=os.getenv("WANDB_PROJECT"),
            name=args.experiment,
            config=asdict(cfg),
            tags=["transfer-learning", "real-estate", args.transfer],
            reinit=True,
        )

    best_val_acc, best_path, history = fit(
        model,
        train_loader,
        val_loader,
        cfg,
        device,
        output_dir,
        param_groups=param_groups,
        use_wandb=use_wandb,
    )

    # Reload best weights before test eval
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    metrics = evaluate_and_log(model, test_loader, classes, device, output_dir, use_wandb, tta=args.tta)

    summary = {
        "experiment": args.experiment,
        "model": args.model,
        "transfer_strategy": args.transfer,
        "best_val_accuracy": best_val_acc,
        "test_accuracy": metrics["test_accuracy"],
        "macro_f1": metrics["report"]["macro avg"]["f1-score"],
        "weighted_f1": metrics["report"]["weighted avg"]["f1-score"],
        "epochs_run": len(history),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if use_wandb and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
