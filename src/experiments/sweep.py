"""W&B bayesian sweep on top of MobileNetV3-Small (CPU-friendly).

The search budget is intentionally small (default 4 runs) to fit within the
project's CPU-only hardware budget, while still showcasing the bayesian
optimisation flow.
"""
from __future__ import annotations

import argparse
import os
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import wandb
from dotenv import load_dotenv

from src.utils.data import build_loaders, discover_dataset, stratified_split
from src.utils.device import detect_device
from src.utils.models import build_model, differential_lr_param_groups
from src.utils.train_loop import TrainConfig, fit

ROOT = Path(__file__).resolve().parents[2]
SWEEP_DIR = ROOT / "models" / "sweep"


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


SWEEP_IMAGE_SIZE = int(os.getenv("SWEEP_IMAGE_SIZE", "160"))
SWEEP_TRAIN_SUBSET = int(os.getenv("SWEEP_TRAIN_SUBSET", "1200"))


def _maybe_subset(split, max_train: int):
    if max_train is None or max_train >= len(split.train_paths):
        return split
    rng = random.Random(42)
    indices = list(range(len(split.train_paths)))
    rng.shuffle(indices)
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


def _train_run() -> None:
    """Function executed for each sweep agent run."""
    load_dotenv()
    run = wandb.init()
    cfg_dict = dict(wandb.config)
    _set_seeds(42)
    device = detect_device(verbose=False)

    paths, labels, classes = discover_dataset()
    split = stratified_split(paths, labels, classes)
    split = _maybe_subset(split, SWEEP_TRAIN_SUBSET)

    model = build_model(
        cfg_dict.get("model_name", "mobilenetv3_small_100"),
        num_classes=len(classes),
        pretrained=True,
        drop_rate=cfg_dict.get("dropout", 0.2),
    ).to(device)
    param_groups = differential_lr_param_groups(
        model, head_lr=cfg_dict["learning_rate"], backbone_lr_factor=0.1
    )

    cfg = TrainConfig(
        experiment=run.name,
        model_name=cfg_dict.get("model_name", "mobilenetv3_small_100"),
        num_classes=len(classes),
        epochs=cfg_dict.get("epochs", 3),
        batch_size=cfg_dict["batch_size"],
        image_size=SWEEP_IMAGE_SIZE,
        learning_rate=cfg_dict["learning_rate"],
        backbone_lr_factor=0.1,
        weight_decay=1e-4,
        optimizer=cfg_dict["optimizer"],
        label_smoothing=cfg_dict["label_smoothing"],
        dropout=cfg_dict["dropout"],
        early_stopping_patience=2,
        transfer_strategy="fine_tuning",
    )

    train_loader, val_loader, _ = build_loaders(
        split,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=0,
        pin_memory=False,
    )

    output_dir = SWEEP_DIR / run.name
    best_val_acc, _, _ = fit(
        model,
        train_loader,
        val_loader,
        cfg,
        device,
        output_dir,
        param_groups=param_groups,
        use_wandb=True,
    )
    wandb.summary["best_val_accuracy"] = best_val_acc
    wandb.finish()


def build_sweep_config(epochs: int = 3) -> dict:
    """Bayesian sweep over learning rate, batch size, dropout and label smoothing."""
    return {
        "method": "bayes",
        "metric": {"name": "val_accuracy", "goal": "maximize"},
        "parameters": {
            "learning_rate": {
                "min": 1e-5,
                "max": 5e-3,
                "distribution": "log_uniform_values",
            },
            "batch_size": {"values": [16, 32]},
            "dropout": {"values": [0.1, 0.2, 0.3]},
            "label_smoothing": {"values": [0.0, 0.1, 0.2]},
            "optimizer": {"values": ["adam", "adamw", "sgd"]},
            "model_name": {"value": "mobilenetv3_small_100"},
            "epochs": {"value": epochs},
        },
    }


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=4, help="Bayesian sweep budget (default 4 due to CPU)")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    sweep_cfg = build_sweep_config(epochs=args.epochs)
    sweep_id = wandb.sweep(
        sweep_cfg,
        entity=os.getenv("WANDB_ENTITY"),
        project=os.getenv("WANDB_PROJECT"),
    )
    print(f"Sweep ID: {sweep_id}")
    wandb.agent(sweep_id, function=_train_run, count=args.count)


if __name__ == "__main__":
    main()
