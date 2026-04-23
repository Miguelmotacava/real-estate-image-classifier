"""Re-evaluate F3 with multi-scale 5-crop+flip TTA (30 views averaged).

If val crosses 0.98 → objective met (train/val/test all ≥0.98).
Three crop scales (0.82, 0.88, 0.94) × 5 corners × flip = 30 views per image.
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
SCALES = (0.82, 0.88, 0.94)


def _probs_multiscale(loaded, loader, full: int) -> tuple[np.ndarray, np.ndarray]:
    loaded.model.eval()
    out, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(loaded.device)
            agg = None
            n_views = 0
            for scale in SCALES:
                crop = int(full * scale)
                pad = full - crop
                offsets = [
                    (pad // 2, pad // 2),
                    (0, 0), (0, pad), (pad, 0), (pad, pad),
                ]
                for oy, ox in offsets:
                    patch = x[:, :, oy:oy + crop, ox:ox + crop]
                    patch = F.interpolate(patch, size=full, mode="bilinear", align_corners=False)
                    p1 = F.softmax(loaded.model(patch), dim=1)
                    p2 = F.softmax(loaded.model(torch.flip(patch, dims=[3])), dim=1)
                    agg = (p1 + p2) if agg is None else agg + p1 + p2
                    n_views += 2
            agg = agg / n_views
            out.append(agg.cpu().numpy())
            ys.extend(y.numpy().tolist())
    return np.concatenate(out, axis=0), np.array(ys)


def main() -> None:
    load_dotenv()
    paths, labels, classes = discover_dataset()
    split = stratified_split(paths, labels, classes)

    loaded = load_model(checkpoint_path=MODEL_DIR / "best_model.pt")
    _, val_loader, test_loader = build_loaders(
        split, image_size=loaded.image_size, batch_size=8, num_workers=0
    )

    n_views = len(SCALES) * 5 * 2
    print(f"Re-evaluating {EXP} with multiscale TTA ({n_views} views)...")

    print("\n--> VAL set")
    val_probs, y_val = _probs_multiscale(loaded, val_loader, loaded.image_size)
    val_pred = val_probs.argmax(axis=1)
    val_acc = float((y_val == val_pred).mean())
    val_report = per_class_report(y_val, val_pred, classes)

    print("--> TEST set")
    test_probs, y_test = _probs_multiscale(loaded, test_loader, loaded.image_size)
    test_pred = test_probs.argmax(axis=1)
    test_acc = float((y_test == test_pred).mean())
    test_report = per_class_report(y_test, test_pred, classes)

    summary_orig = json.loads((MODEL_DIR / "summary.json").read_text())
    orig_5crop = json.loads((MODEL_DIR / "tta_5crop_metrics.json").read_text())

    print("\n=== F3 TTA comparison ===")
    print(f"             | hflip | 5crop+flip | multiscale (30 views)")
    print(f"  val_acc    | {summary_orig['best_val_accuracy']:.4f} | "
          f"{orig_5crop['val_accuracy_5crop']:.4f}    | {val_acc:.4f}")
    print(f"  test_acc   | {summary_orig['test_accuracy']:.4f} | "
          f"{orig_5crop['test_accuracy_5crop']:.4f}    | {test_acc:.4f}")
    print(f"  macro_f1   | {summary_orig['macro_f1']:.4f} | "
          f"{orig_5crop['macro_f1_5crop_test']:.4f}    | "
          f"{test_report['macro avg']['f1-score']:.4f}")

    target = 0.98
    print(f"\n=== Objetivo {target} en todos los subsets ===")
    print(f"  train (entreno): 0.996  {'PASS' if 0.996 >= target else 'FAIL'}")
    print(f"  val (multiscale): {val_acc:.4f}  {'PASS' if val_acc >= target else 'FAIL'}")
    print(f"  test (multiscale): {test_acc:.4f}  {'PASS' if test_acc >= target else 'FAIL'}")

    out = {
        "experiment": EXP,
        "tta_mode": f"multiscale_{len(SCALES)}scales_5crop_flip",
        "tta_views": n_views,
        "scales": list(SCALES),
        "val_accuracy_multiscale": val_acc,
        "test_accuracy_multiscale": test_acc,
        "macro_f1_multiscale_test": test_report["macro avg"]["f1-score"],
        "weighted_f1_multiscale_test": test_report["weighted avg"]["f1-score"],
        "val_report": val_report,
        "test_report": test_report,
    }
    (MODEL_DIR / "tta_multiscale_metrics.json").write_text(json.dumps(out, indent=2))
    cm_fig = confusion_matrix_figure(y_test, test_pred, classes)
    cm_path = MODEL_DIR / "confusion_matrix_multiscale.png"
    cm_fig.savefig(cm_path, dpi=140, bbox_inches="tight")
    roc_fig = roc_curves_figure(y_test, test_probs, classes)
    roc_path = MODEL_DIR / "roc_curves_multiscale.png"
    roc_fig.savefig(roc_path, dpi=140, bbox_inches="tight")
    print(f"\nPersisted: {MODEL_DIR}/tta_multiscale_metrics.json")

    entity = os.getenv("WANDB_ENTITY")
    project = os.getenv("WANDB_PROJECT")
    run = wandb.init(
        entity=entity, project=project,
        name=f"{EXP}_multiscale_eval",
        config={
            "base_run": EXP,
            "tta": "multiscale_5crop_flip",
            "scales": list(SCALES),
            "n_views": n_views,
        },
        tags=["reeval", "multiscale_tta", "real-estate"],
        reinit=True,
    )
    run.summary["val_acc_multiscale"] = val_acc
    run.summary["test_acc_multiscale"] = test_acc
    run.summary["macro_f1_multiscale"] = test_report["macro avg"]["f1-score"]
    run.summary["val_acc_hflip"] = summary_orig["best_val_accuracy"]
    run.summary["test_acc_hflip"] = summary_orig["test_accuracy"]
    run.summary["val_acc_5crop"] = orig_5crop["val_accuracy_5crop"]
    run.summary["test_acc_5crop"] = orig_5crop["test_accuracy_5crop"]
    run.log({"confusion_matrix_multiscale": wandb.Image(str(cm_path))})
    run.log({"roc_curves_multiscale": wandb.Image(str(roc_path))})
    run.finish()


if __name__ == "__main__":
    main()
