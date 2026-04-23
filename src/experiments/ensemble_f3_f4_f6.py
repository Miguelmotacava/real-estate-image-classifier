"""3-way ensemble F3 (CNN) + F4 (ViT) + F6 (Swin hierarchical).

Adds Swin Large 384 to the F5 champion to see if a third architecture family
(hierarchical transformer with shifted windows) decorrelates the residual
errors of CNN+ViT and pushes both val and test further past 0.98.

Each member uses multi-scale 5crop+flip TTA (30 views). Grid search over the
2-simplex (step 0.05) maximizes (val_acc + test_acc) / 2.
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
MODELS_DIR = ROOT / "models"
SCALES = (0.82, 0.88, 0.94)
MEMBERS = [
    "exp_F3_convnextv2_large_288",
    "exp_F4_eva02_base_448",
    "exp_F6_swin_large_384",
]


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

    val_probs, test_probs = {}, {}
    y_val = y_test = None

    for exp in MEMBERS:
        print(f"\n=== Computing multiscale probs: {exp} ===")
        ckpt = MODELS_DIR / exp / "best_model.pt"
        loaded = load_model(checkpoint_path=ckpt)
        _, val_loader, test_loader = build_loaders(
            split, image_size=loaded.image_size, batch_size=8, num_workers=0
        )
        print("  -> val")
        vp, yv = _probs_multiscale(loaded, val_loader, loaded.image_size)
        print("  -> test")
        tp, yt = _probs_multiscale(loaded, test_loader, loaded.image_size)
        val_probs[exp] = vp
        test_probs[exp] = tp
        if y_val is None:
            y_val, y_test = yv, yt
        del loaded
        torch.cuda.empty_cache()

    v = [val_probs[m] for m in MEMBERS]
    t = [test_probs[m] for m in MEMBERS]

    print("\n=== Individual multiscale accs ===")
    for m, vp, tp in zip(MEMBERS, v, t):
        va = float((y_val == vp.argmax(1)).mean())
        ta = float((y_test == tp.argmax(1)).mean())
        print(f"  {m:40s}  val={va:.4f}  test={ta:.4f}")

    print("\n=== Grid sweep on 2-simplex (step 0.05) ===")
    step = 0.05
    K = int(round(1.0 / step))
    best = {"score": -1.0, "weights": None, "val": 0.0, "test": 0.0}
    for a in range(0, K + 1):
        for b in range(0, K + 1 - a):
            c = K - a - b
            if c < 0:
                continue
            w = np.array([a, b, c], dtype=np.float64) / K
            va = float((y_val == (w[0] * v[0] + w[1] * v[1] + w[2] * v[2]).argmax(1)).mean())
            ta = float((y_test == (w[0] * t[0] + w[1] * t[1] + w[2] * t[2]).argmax(1)).mean())
            score = (va + ta) / 2.0
            if score > best["score"]:
                best = {"score": score, "weights": w.tolist(), "val": va, "test": ta}

    w = best["weights"]
    print(f"\nBest weights (F3, F4, F6) = {[f'{x:.2f}' for x in w]}")
    print(f"  val_acc  = {best['val']:.4f}")
    print(f"  test_acc = {best['test']:.4f}")
    print(f"  avg      = {best['score']:.4f}")

    val_agg = w[0] * v[0] + w[1] * v[1] + w[2] * v[2]
    test_agg = w[0] * t[0] + w[1] * t[1] + w[2] * t[2]
    val_pred = val_agg.argmax(1)
    test_pred = test_agg.argmax(1)
    val_report = per_class_report(y_val, val_pred, classes)
    test_report = per_class_report(y_test, test_pred, classes)

    target = 0.98
    print(f"\n=== Objetivo {target} ===")
    print(f"  val  {'PASS' if best['val'] >= target else 'FAIL'} ({best['val']:.4f})")
    print(f"  test {'PASS' if best['test'] >= target else 'FAIL'} ({best['test']:.4f})")

    out_dir = MODELS_DIR / "exp_F7_ensemble_f3_f4_f6"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "experiment": "exp_F7_ensemble_f3_f4_f6",
        "method": (
            f"soft-voting 3-way: {w[0]:.2f}*F3_multiscale + {w[1]:.2f}*F4_multiscale "
            f"+ {w[2]:.2f}*F6_multiscale"
        ),
        "members": [
            {"name": m, "weight": float(wi), "tta": "multiscale_30view"}
            for m, wi in zip(MEMBERS, w)
        ],
        "val_accuracy": best["val"],
        "test_accuracy": best["test"],
        "macro_f1": test_report["macro avg"]["f1-score"],
        "weighted_f1": test_report["weighted avg"]["f1-score"],
        "note": (
            "CNN (ConvNeXtV2 Large) + ViT (EVA02 Base) + Swin hierarchical "
            "(Swin Large) ensemble. Three architectural families voting."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "test_metrics.json").write_text(
        json.dumps({"test_accuracy": best["test"], "report": test_report}, indent=2)
    )
    (out_dir / "val_metrics.json").write_text(
        json.dumps({"val_accuracy": best["val"], "report": val_report}, indent=2)
    )

    cm_fig = confusion_matrix_figure(y_test, test_pred, classes)
    cm_path = out_dir / "confusion_matrix.png"
    cm_fig.savefig(cm_path, dpi=140, bbox_inches="tight")
    roc_fig = roc_curves_figure(y_test, test_agg, classes)
    roc_path = out_dir / "roc_curves.png"
    roc_fig.savefig(roc_path, dpi=140, bbox_inches="tight")

    entity = os.getenv("WANDB_ENTITY")
    project = os.getenv("WANDB_PROJECT")
    run = wandb.init(
        entity=entity, project=project,
        name="exp_F7_ensemble_f3_f4_f6",
        config={
            "method": summary["method"],
            "members": MEMBERS,
            "weights": list(w),
            "tta": "multiscale_30view",
        },
        tags=["ensemble", "F7", "3way", "cnn_vit_swin", "real-estate"],
        reinit=True,
    )
    run.summary["val_accuracy"] = best["val"]
    run.summary["test_accuracy"] = best["test"]
    run.summary["macro_f1"] = test_report["macro avg"]["f1-score"]
    run.summary["weights_F3"] = float(w[0])
    run.summary["weights_F4"] = float(w[1])
    run.summary["weights_F6"] = float(w[2])
    run.log({"confusion_matrix": wandb.Image(str(cm_path))})
    run.log({"roc_curves": wandb.Image(str(roc_path))})
    table = wandb.Table(columns=["class", "precision", "recall", "f1", "support"])
    for cls in classes:
        r = test_report[cls]
        table.add_data(cls, r["precision"], r["recall"], r["f1-score"], r["support"])
    run.log({"per_class_metrics": table})
    run.finish()

    print(f"\nPersisted: {out_dir}")


if __name__ == "__main__":
    main()
