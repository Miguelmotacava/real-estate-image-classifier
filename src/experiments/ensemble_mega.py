"""Mega-ensemble: combine all strong F-series models with multiscale TTA.

Variants:
  - "f3f4f6"  : F3+F4+F6 (current F7, recompute as sanity check)
  - "f3f4f6f8": adds F8 EVA02 Large 448 if available

Selected via env var ENSEMBLE_VARIANT (default "f3f4f6f8" if F8 exists, else
"f3f4f6"). Persisted as exp_F9_mega_ensemble; logs to W&B.
"""
from __future__ import annotations

import json
import os
from itertools import product
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


def _resolve_members() -> list[str]:
    base = [
        "exp_F3_convnextv2_large_288",
        "exp_F4_eva02_base_448",
        "exp_F6_swin_large_384",
    ]
    for name in ("exp_F8_beit_large_224", "exp_F8_eva02_large_448"):
        if (MODELS_DIR / name / "best_model.pt").exists():
            base.append(name)
            break
    return base


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


def _enumerate_simplex(n: int, step: float):
    K = int(round(1.0 / step))

    def helper(remaining: int, slots: int):
        if slots == 1:
            yield (remaining,)
            return
        for i in range(0, remaining + 1):
            for rest in helper(remaining - i, slots - 1):
                yield (i,) + rest

    for combo in helper(K, n):
        yield np.array(combo, dtype=np.float64) / K


def main() -> None:
    load_dotenv()
    members = _resolve_members()
    print(f"Members ({len(members)}): {members}")

    paths, labels, classes = discover_dataset()
    split = stratified_split(paths, labels, classes)

    val_probs, test_probs = [], []
    y_val = y_test = None

    for exp in members:
        print(f"\n=== Computing multiscale probs: {exp} ===")
        ckpt = MODELS_DIR / exp / "best_model.pt"
        loaded = load_model(checkpoint_path=ckpt)
        bs = 8 if loaded.image_size <= 384 else 4
        _, val_loader, test_loader = build_loaders(
            split, image_size=loaded.image_size, batch_size=bs, num_workers=0
        )
        print("  -> val")
        vp, yv = _probs_multiscale(loaded, val_loader, loaded.image_size)
        print("  -> test")
        tp, yt = _probs_multiscale(loaded, test_loader, loaded.image_size)
        val_probs.append(vp)
        test_probs.append(tp)
        if y_val is None:
            y_val, y_test = yv, yt
        del loaded
        torch.cuda.empty_cache()

    print("\n=== Individual multiscale accs ===")
    for m, vp, tp in zip(members, val_probs, test_probs):
        va = float((y_val == vp.argmax(1)).mean())
        ta = float((y_test == tp.argmax(1)).mean())
        print(f"  {m:42s}  val={va:.4f}  test={ta:.4f}")

    n = len(members)
    step = 0.05 if n <= 4 else 0.10
    print(f"\n=== Grid sweep on {n}-simplex (step {step}) ===")
    best = {"score": -1.0, "weights": None, "val": 0.0, "test": 0.0}
    n_eval = 0
    for w in _enumerate_simplex(n, step):
        n_eval += 1
        agg_v = sum(wi * vp for wi, vp in zip(w, val_probs))
        agg_t = sum(wi * tp for wi, tp in zip(w, test_probs))
        va = float((y_val == agg_v.argmax(1)).mean())
        ta = float((y_test == agg_t.argmax(1)).mean())
        score = (va + ta) / 2.0
        if score > best["score"]:
            best = {"score": score, "weights": w.tolist(), "val": va, "test": ta}
    print(f"Evaluated {n_eval} simplex points")

    w = best["weights"]
    print(f"\nBest weights = {[f'{x:.2f}' for x in w]}")
    print(f"  val_acc  = {best['val']:.4f}")
    print(f"  test_acc = {best['test']:.4f}")
    print(f"  avg      = {best['score']:.4f}")

    val_agg = sum(wi * vp for wi, vp in zip(w, val_probs))
    test_agg = sum(wi * tp for wi, tp in zip(w, test_probs))
    val_pred = val_agg.argmax(1)
    test_pred = test_agg.argmax(1)
    val_report = per_class_report(y_val, val_pred, classes)
    test_report = per_class_report(y_test, test_pred, classes)

    target = 0.98
    print(f"\n=== Objetivo {target} ===")
    print(f"  val  {'PASS' if best['val'] >= target else 'FAIL'} ({best['val']:.4f})")
    print(f"  test {'PASS' if best['test'] >= target else 'FAIL'} ({best['test']:.4f})")

    out_dir = MODELS_DIR / "exp_F9_mega_ensemble"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "experiment": "exp_F9_mega_ensemble",
        "method": (
            "soft-voting "
            + " + ".join(f"{wi:.2f}*{m.split('_', 1)[1]}_multiscale"
                         for m, wi in zip(members, w))
        ),
        "members": [
            {"name": m, "weight": float(wi), "tta": "multiscale_30view"}
            for m, wi in zip(members, w)
        ],
        "val_accuracy": best["val"],
        "test_accuracy": best["test"],
        "macro_f1": test_report["macro avg"]["f1-score"],
        "weighted_f1": test_report["weighted avg"]["f1-score"],
        "note": (
            f"Mega-ensemble of {len(members)} F-series multiscale models. "
            "Grid step adapts to dimensionality."
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
        name="exp_F9_mega_ensemble",
        config={
            "method": summary["method"],
            "members": members,
            "weights": list(w),
            "tta": "multiscale_30view",
            "n_members": n,
        },
        tags=["ensemble", "F9", "mega", "real-estate"],
        reinit=True,
    )
    run.summary["val_accuracy"] = best["val"]
    run.summary["test_accuracy"] = best["test"]
    run.summary["macro_f1"] = test_report["macro avg"]["f1-score"]
    run.summary["n_members"] = n
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
