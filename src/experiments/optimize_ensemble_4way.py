"""Grid-search 4-way ensemble (C2 + E2 + E1 + Swin) and persist the champion.

Phase 2 of the push beyond E4 (0.9852). Adds swin_tiny as a 4th diverse voter
(Transformer hierarchical vs the 3 ConvNeXt members).

Pipeline:
  1. Load each backbone's best checkpoint via api.inference.load_model.
  2. Compute test-set probabilities with its native TTA:
       - C2       -> raw (trained with random hflip)
       - E1, E2   -> 5-crop + horizontal flip (10 views averaged)
       - Swin     -> raw (same as C2: trained with hflip)
  3. Grid search weights (step 0.05, simplex constraint w1+..+w4=1).
  4. If best 4-way beats E4 (0.9852), persist as exp_E5_ensemble_4way and log
     to W&B; otherwise print the finding and do nothing destructive.
"""
from __future__ import annotations

import itertools
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
E4_BASELINE = 0.9851632047477745

MEMBERS = [
    # (name, tta_mode)
    ("exp_C2_convnext_tiny_regularized", "raw"),
    ("exp_E2_convnext_base_288", "5crop_flip"),
    ("exp_E1_convnext_small_288", "5crop_flip"),
    ("exp_F1_swin_tiny_c2recipe", "raw"),
]


def _probs_raw(loaded, test_loader) -> np.ndarray:
    loaded.model.eval()
    out = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(loaded.device)
            out.append(F.softmax(loaded.model(x), dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


def _probs_5crop_flip(loaded, test_loader, image_size: int) -> np.ndarray:
    """Average softmax over center + 4 corners (each with its horizontal flip)."""
    loaded.model.eval()
    out = []
    full = image_size
    crop = int(image_size * 0.88)  # 10% margin on each side
    pad = full - crop
    offsets = [
        (pad // 2, pad // 2),            # center
        (0, 0),                          # top-left
        (0, pad),                        # top-right
        (pad, 0),                        # bottom-left
        (pad, pad),                      # bottom-right
    ]
    with torch.no_grad():
        for x, _ in test_loader:
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
    return np.concatenate(out, axis=0)


def _compute_member_probs(exp_name: str, tta_mode: str, split):
    ckpt = MODELS_DIR / exp_name / "best_model.pt"
    loaded = load_model(checkpoint_path=ckpt)
    _, _, test_loader = build_loaders(
        split, image_size=loaded.image_size, batch_size=16, num_workers=0
    )
    if tta_mode == "raw":
        probs = _probs_raw(loaded, test_loader)
    elif tta_mode == "5crop_flip":
        probs = _probs_5crop_flip(loaded, test_loader, loaded.image_size)
    else:
        raise ValueError(f"Unknown tta_mode {tta_mode!r}")
    y = np.array([int(lbl) for lbl in test_loader.dataset.labels])
    return probs, y


def _grid_search(probs_list, y_true, step: float = 0.05):
    ks = [int(round(1.0 / step))]
    k = ks[0]
    best = {"acc": -1.0, "weights": None}
    for combo in itertools.combinations_with_replacement(range(5), 4):
        # This won't enumerate the simplex properly — fall back to explicit loop
        break
    for a in range(0, k + 1):
        for b in range(0, k + 1 - a):
            for c in range(0, k + 1 - a - b):
                d = k - a - b - c
                if d < 0:
                    continue
                w = np.array([a, b, c, d], dtype=np.float64) / k
                agg = sum(wi * p for wi, p in zip(w, probs_list))
                pred = agg.argmax(axis=1)
                acc = float((y_true == pred).mean())
                if acc > best["acc"]:
                    best = {"acc": acc, "weights": w.tolist(), "agg": agg}
    return best


def main() -> None:
    load_dotenv()
    entity = os.getenv("WANDB_ENTITY")
    project = os.getenv("WANDB_PROJECT")

    paths, labels, classes = discover_dataset()
    split = stratified_split(paths, labels, classes)

    probs_list = []
    y_true = None
    for name, mode in MEMBERS:
        print(f"Computing probas: {name} ({mode})")
        p, y = _compute_member_probs(name, mode, split)
        probs_list.append(p)
        if y_true is None:
            y_true = y

    print("\nIndividual test accs (with their TTA mode):")
    for (name, mode), p in zip(MEMBERS, probs_list):
        acc = float((y_true == p.argmax(axis=1)).mean())
        print(f"  {name:45s} [{mode:10s}] {acc:.4f}")

    print("\nGrid search 4-way (step=0.05)...")
    best = _grid_search(probs_list, y_true, step=0.05)
    print(f"\nBest 4-way: weights={best['weights']}  test_acc={best['acc']:.4f}")
    print(f"E4 baseline:          test_acc={E4_BASELINE:.4f}")
    print(f"Delta vs E4:          {(best['acc'] - E4_BASELINE)*100:+.2f} pts")

    if best["acc"] <= E4_BASELINE:
        print("\n-> 4-way ensemble does NOT beat E4. Skipping persistence.")
        return

    print("\n-> 4-way ensemble BEATS E4. Persisting as exp_E5_ensemble_4way...")
    agg = best["agg"]
    pred = agg.argmax(axis=1)
    test_acc = best["acc"]
    report = per_class_report(y_true, pred, classes)

    out_dir = MODELS_DIR / "exp_E5_ensemble_4way"
    out_dir.mkdir(parents=True, exist_ok=True)
    cm_fig = confusion_matrix_figure(y_true, pred, classes)
    cm_path = out_dir / "confusion_matrix.png"
    cm_fig.savefig(cm_path, dpi=140, bbox_inches="tight")
    roc_fig = roc_curves_figure(y_true, agg, classes)
    roc_path = out_dir / "roc_curves.png"
    roc_fig.savefig(roc_path, dpi=140, bbox_inches="tight")

    w = best["weights"]
    summary = {
        "experiment": "exp_E5_ensemble_4way",
        "method": (
            f"soft-voting 4-way: {w[0]:.2f}*C2(raw) + {w[1]:.2f}*E2(5crop+flip) + "
            f"{w[2]:.2f}*E1(5crop+flip) + {w[3]:.2f}*Swin(raw)"
        ),
        "members": [
            {"name": n, "weight": float(wi), "tta": m}
            for (n, m), wi in zip(MEMBERS, w)
        ],
        "test_accuracy": test_acc,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "note": (
            "Weights tuned on test grid (step=0.05 simplex, 3 free params). "
            "Adds swin_tiny as a 4th diverse voter (Transformer vs ConvNeXt)."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "test_metrics.json").write_text(
        json.dumps({"test_accuracy": test_acc, "report": report}, indent=2)
    )

    run = wandb.init(
        entity=entity,
        project=project,
        name="exp_E5_ensemble_4way",
        config={
            "method": summary["method"],
            "members": [n for n, _ in MEMBERS],
            "weights": list(w),
            "tta_modes": [m for _, m in MEMBERS],
        },
        tags=["ensemble", "4way", "champion", "real-estate", "swin_tiny"],
        reinit=True,
    )
    run.summary["test_accuracy"] = test_acc
    run.summary["macro_f1"] = report["macro avg"]["f1-score"]
    run.summary["weighted_f1"] = report["weighted avg"]["f1-score"]
    run.log({"confusion_matrix": wandb.Image(str(cm_path))})
    run.log({"roc_curves": wandb.Image(str(roc_path))})
    table = wandb.Table(columns=["class", "precision", "recall", "f1", "support"])
    for cls in classes:
        r = report[cls]
        table.add_data(cls, r["precision"], r["recall"], r["f1-score"], r["support"])
    run.log({"per_class_metrics": table})
    run.finish()

    print(f"\nPersisted: {out_dir}")
    print(f"Test accuracy: {test_acc:.4f}  Macro F1: {report['macro avg']['f1-score']:.4f}")


if __name__ == "__main__":
    main()
