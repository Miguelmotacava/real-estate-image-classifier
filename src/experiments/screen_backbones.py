"""Screen multiple pretrained backbones to pick the best for intensive fine-tuning.

Each candidate is fine-tuned with a small budget (3 epochs, full train set,
224x224, batch size from device heuristic) on the GPU. The best by
``best_val_accuracy`` is highlighted at the end.
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import wandb
from dotenv import load_dotenv

from src.experiments.run_experiment import evaluate_and_log
from src.utils.data import build_loaders, discover_dataset, stratified_split
from src.utils.device import detect_device, recommended_batch_size
from src.utils.models import build_model, differential_lr_param_groups
from src.utils.train_loop import TrainConfig, fit

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"

CANDIDATES = [
    {"name": "efficientnet_b0",        "lr": 7e-4, "image_size": 224},
    {"name": "efficientnet_b3",        "lr": 5e-4, "image_size": 224},
    {"name": "resnet50",               "lr": 5e-4, "image_size": 224},
    {"name": "convnext_tiny",          "lr": 3e-4, "image_size": 224},
    {"name": "regnety_008",            "lr": 5e-4, "image_size": 224},
    {"name": "deit_small_patch16_224", "lr": 3e-4, "image_size": 224},
]


def _seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_one(cfg_dict: dict, device, split, classes) -> dict:
    name = cfg_dict["name"]
    image_size = cfg_dict["image_size"]
    lr = cfg_dict["lr"]
    bs = recommended_batch_size(device)
    experiment = f"exp_B_{name}"
    output_dir = MODELS_DIR / experiment
    output_dir.mkdir(parents=True, exist_ok=True)

    _seed(42)
    model = build_model(name, num_classes=len(classes), pretrained=True, drop_rate=0.2).to(device)
    param_groups = differential_lr_param_groups(model, head_lr=lr, backbone_lr_factor=0.1)

    cfg = TrainConfig(
        experiment=experiment,
        model_name=name,
        num_classes=len(classes),
        epochs=3,
        batch_size=bs,
        image_size=image_size,
        learning_rate=lr,
        backbone_lr_factor=0.1,
        weight_decay=1e-4,
        optimizer="adamw",
        label_smoothing=0.1,
        dropout=0.2,
        early_stopping_patience=5,
        transfer_strategy="fine_tuning",
    )

    train_loader, val_loader, test_loader = build_loaders(
        split, image_size=image_size, batch_size=bs,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    use_wandb = bool(os.getenv("WANDB_API_KEY"))
    if use_wandb:
        wandb.init(
            entity=os.getenv("WANDB_ENTITY"),
            project=os.getenv("WANDB_PROJECT"),
            name=experiment,
            config=asdict(cfg),
            tags=["screening", "backbone-selection", "gpu"],
            reinit=True,
        )

    t0 = time.time()
    best_val_acc, best_path, history = fit(
        model, train_loader, val_loader, cfg, device, output_dir,
        param_groups=param_groups, use_wandb=use_wandb,
    )
    elapsed = time.time() - t0

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    metrics = evaluate_and_log(model, test_loader, classes, device, output_dir, use_wandb)

    summary = {
        "experiment": experiment,
        "model": name,
        "transfer_strategy": "fine_tuning",
        "best_val_accuracy": best_val_acc,
        "test_accuracy": metrics["test_accuracy"],
        "macro_f1": metrics["report"]["macro avg"]["f1-score"],
        "weighted_f1": metrics["report"]["weighted avg"]["f1-score"],
        "epochs_run": len(history),
        "train_seconds": round(elapsed, 1),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if use_wandb and wandb.run is not None:
        wandb.summary["train_seconds"] = elapsed
        wandb.finish()
    return summary


def main() -> None:
    load_dotenv()
    device = detect_device(verbose=True)
    print(f"Recommended batch size: {recommended_batch_size(device)}")

    paths, labels, classes = discover_dataset()
    split = stratified_split(paths, labels, classes)

    results = []
    for cfg in CANDIDATES:
        print(f"\n>>> Screening {cfg['name']}")
        try:
            res = _run_one(cfg, device, split, classes)
        except Exception as exc:
            print(f"!!! {cfg['name']} failed: {exc}")
            res = {"experiment": f"exp_B_{cfg['name']}", "error": str(exc)}
        results.append(res)
        print(json.dumps(res, indent=2))

    print("\n=== Screening summary ===")
    ok = [r for r in results if "error" not in r]
    ok.sort(key=lambda r: r["best_val_accuracy"], reverse=True)
    for r in ok:
        print(f"  {r['model']:<30} val={r['best_val_accuracy']:.3f}  test={r['test_accuracy']:.3f}  "
              f"f1={r['macro_f1']:.3f}  t={r['train_seconds']:.0f}s")
    if ok:
        print(f"\n>>> Winner: {ok[0]['model']} (val={ok[0]['best_val_accuracy']:.3f})")


if __name__ == "__main__":
    main()
