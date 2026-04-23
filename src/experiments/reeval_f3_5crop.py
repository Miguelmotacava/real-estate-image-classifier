"""Re-evaluate F3 (ConvNeXtV2 Large 288) on val and test with 5-crop+flip TTA.

If either val or test crosses 0.98, persist the upgraded summary. Always logs
results back to the existing W&B run for traceability.
"""
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
EXP = os.getenv("EXP_NAME", "exp_F3_convnextv2_large_288")
MODEL_DIR = ROOT / "models" / EXP


def _probs_5crop_flip(loaded, loader, full: int) -> tuple[np.ndarray, np.ndarray]:
    loaded.model.eval()
    crop = int(full * 0.88)
    pad = full - crop
    offsets = [
        (pad // 2, pad // 2),
        (0, 0), (0, pad), (pad, 0), (pad, pad),
    ]
    out, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(loaded.device)
            agg = None
            for oy, ox in offsets:
                patch = x[:, :, oy:oy + crop, ox:ox + crop]
                patch = F.interpolate(patch, size=full, mode="bilinear", align_corners=False)
                p1 = F.softmax(loaded.model(patch), dim=1)
                p2 = F.softmax(loaded.model(torch.flip(patch, dims=[3])), dim=1)
                pm = (p1 + p2) / 2.0
                agg = pm if agg is None else agg + pm
            agg = agg / len(offsets)
            out.append(agg.cpu().numpy())
            ys.extend(y.numpy().tolist())
    return np.concatenate(out, axis=0), np.array(ys)


def main() -> None:
    load_dotenv()
    paths, labels, classes = discover_dataset()
    split = stratified_split(paths, labels, classes)

    loaded = load_model(checkpoint_path=MODEL_DIR / "best_model.pt")
    train_loader, val_loader, test_loader = build_loaders(
        split, image_size=loaded.image_size, batch_size=8, num_workers=0
    )

    print(f"Re-evaluating {EXP} with 5-crop+flip TTA at {loaded.image_size}px...")

    print("\n--> VAL set")
    val_probs, y_val = _probs_5crop_flip(loaded, val_loader, loaded.image_size)
    val_pred = val_probs.argmax(axis=1)
    val_acc = float((y_val == val_pred).mean())
    val_report = per_class_report(y_val, val_pred, classes)

    print("--> TEST set")
    test_probs, y_test = _probs_5crop_flip(loaded, test_loader, loaded.image_size)
    test_pred = test_probs.argmax(axis=1)
    test_acc = float((y_test == test_pred).mean())
    test_report = per_class_report(y_test, test_pred, classes)

    summary_orig = json.loads((MODEL_DIR / "summary.json").read_text())
    print("\n=== F3 results comparison ===")
    print(f"             | hflip TTA | 5crop+flip TTA")
    print(f"  val_acc    | {summary_orig['best_val_accuracy']:.4f}    | {val_acc:.4f}")
    print(f"  test_acc   | {summary_orig['test_accuracy']:.4f}    | {test_acc:.4f}")
    print(f"  macro_f1   | {summary_orig['macro_f1']:.4f}    | {test_report['macro avg']['f1-score']:.4f}")

    target = 0.98
    val_pass = val_acc >= target
    test_pass = test_acc >= target
    print(f"\nObjetivo {target} en val: {'PASS' if val_pass else 'FAIL'}")
    print(f"Objetivo {target} en test: {'PASS' if test_pass else 'FAIL'}")

    out_dir = MODEL_DIR
    out = {
        "experiment": EXP,
        "tta_mode": "5crop_flip",
        "val_accuracy_5crop": val_acc,
        "test_accuracy_5crop": test_acc,
        "macro_f1_5crop_test": test_report["macro avg"]["f1-score"],
        "weighted_f1_5crop_test": test_report["weighted avg"]["f1-score"],
        "val_report": val_report,
        "test_report": test_report,
    }
    (out_dir / "tta_5crop_metrics.json").write_text(json.dumps(out, indent=2))
    cm_fig = confusion_matrix_figure(y_test, test_pred, classes)
    cm_path = out_dir / "confusion_matrix_5crop.png"
    cm_fig.savefig(cm_path, dpi=140, bbox_inches="tight")
    roc_fig = roc_curves_figure(y_test, test_probs, classes)
    roc_path = out_dir / "roc_curves_5crop.png"
    roc_fig.savefig(roc_path, dpi=140, bbox_inches="tight")
    print(f"\nPersisted: {out_dir}/tta_5crop_metrics.json")

    entity = os.getenv("WANDB_ENTITY")
    project = os.getenv("WANDB_PROJECT")
    run = wandb.init(
        entity=entity, project=project,
        name=f"{EXP}_5crop_eval",
        config={"base_run": EXP, "tta": "5crop_flip"},
        tags=["reeval", "5crop", "real-estate"],
        reinit=True,
    )
    run.summary["val_acc_5crop"] = val_acc
    run.summary["test_acc_5crop"] = test_acc
    run.summary["macro_f1_5crop"] = test_report["macro avg"]["f1-score"]
    run.summary["val_acc_hflip"] = summary_orig["best_val_accuracy"]
    run.summary["test_acc_hflip"] = summary_orig["test_accuracy"]
    run.log({"confusion_matrix_5crop": wandb.Image(str(cm_path))})
    run.log({"roc_curves_5crop": wandb.Image(str(roc_path))})
    run.finish()


if __name__ == "__main__":
    main()
