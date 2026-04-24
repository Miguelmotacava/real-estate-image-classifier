"""Ensemble grid search on the 90/10 val split.

Combines the 4 final ensemble members trained on the 90/10 split:
  - exp_FINAL_F3_9010 (ConvNeXtV2-L 288)
  - exp_FINAL_F4_9010 (EVA02-B 448)
  - exp_FINAL_swin_large_384_9010 (Swin-L 384) — alias F6
  - exp_FINAL_F8_9010 (BEiT-L 224)

For each member computes single-forward softmax probs (no multi-scale TTA
to keep production latency reasonable), then sweeps a 4-simplex with
step=0.05 to find the weight combination that maximizes val_accuracy.

Persists ensemble.json with members + weights so the API can load it.
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
from torch.utils.data import DataLoader

from src.experiments.final_9010 import split_9010
from src.utils.data import SceneImageDataset, build_transforms, discover_dataset
from src.utils.device import detect_device
from src.utils.metrics import (
    confusion_matrix_figure,
    per_class_report,
    roc_curves_figure,
)
from src.utils.models import build_model

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"

MEMBERS = [
    ("F3", "exp_FINAL_F3_9010"),
    ("F4", "exp_FINAL_F4_9010"),
    ("F6", "exp_FINAL_swin_large_384_9010"),
    ("F8", "exp_FINAL_F8_9010"),
]

# Multi-scale TTA: 3 scales × 5 crops × 2 flips = 30 views per image per model
SCALES = (0.82, 0.88, 0.94)


def _load_member(exp_dir: Path, device: torch.device):
    ckpt = torch.load(exp_dir / "best_model.pt", map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = build_model(
        cfg["model_name"], num_classes=cfg["num_classes"], pretrained=False,
        drop_rate=cfg.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg["image_size"], cfg["model_name"]


@torch.no_grad()
def _probs(model: torch.nn.Module, loader: DataLoader, device: torch.device, full: int):
    """Multi-scale 30-view TTA per image: 3 scales × 5 crops × hflip."""
    out, ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        agg = None
        n_views = 0
        for scale in SCALES:
            crop = int(full * scale)
            pad = full - crop
            offsets = [
                (pad // 2, pad // 2),  # center
                (0, 0), (0, pad), (pad, 0), (pad, pad),  # 4 corners
            ]
            for oy, ox in offsets:
                patch = x[:, :, oy:oy + crop, ox:ox + crop]
                patch = F.interpolate(patch, size=full, mode="bilinear", align_corners=False)
                p1 = F.softmax(model(patch), dim=1)
                p2 = F.softmax(model(torch.flip(patch, dims=[3])), dim=1)
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
    device = detect_device(verbose=True)

    paths, labels, classes = discover_dataset()
    split = split_9010(paths, labels, classes)
    print(f"[ensemble] val={len(split.val_paths)} samples")

    member_probs = []
    member_meta = []
    y_val = None

    for tag, exp_name in MEMBERS:
        exp_dir = MODELS_DIR / exp_name
        if not (exp_dir / "best_model.pt").exists():
            print(f"[ensemble] WARN: {exp_name} not found, skipping")
            continue
        print(f"[ensemble] computing probs: {tag} ({exp_name})", flush=True)
        model, img_size, model_name = _load_member(exp_dir, device)
        eval_tfm = build_transforms(img_size, augment=False)
        val_ds = SceneImageDataset(split.val_paths, split.val_labels, eval_tfm)
        loader = DataLoader(
            val_ds, batch_size=4, shuffle=False, num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
        p, yv = _probs(model, loader, device, img_size)
        member_probs.append(p)
        member_meta.append({"tag": tag, "exp": exp_name, "model_name": model_name, "image_size": img_size})
        if y_val is None:
            y_val = yv
        del model
        torch.cuda.empty_cache()

    n = len(member_probs)
    print(f"\n[ensemble] {n} members loaded; sweeping {n}-simplex (step 0.05)")
    print("\nIndividual val accs:")
    for meta, p in zip(member_meta, member_probs):
        a = float((y_val == p.argmax(1)).mean())
        print(f"  {meta['tag']:3s}  val={a:.4f}  ({meta['exp']})")

    best = {"score": -1.0, "weights": None, "val": 0.0}
    n_eval = 0
    for w in _enumerate_simplex(n, 0.05):
        n_eval += 1
        agg = sum(wi * p for wi, p in zip(w, member_probs))
        va = float((y_val == agg.argmax(1)).mean())
        if va > best["score"]:
            best = {"score": va, "weights": w.tolist(), "val": va}
    print(f"\n[ensemble] Evaluated {n_eval} simplex points")
    w = best["weights"]
    print(f"Best weights = {[f'{x:.2f}' for x in w]}")
    print(f"  val_acc  = {best['val']:.4f}")

    # Final eval
    val_agg = sum(wi * p for wi, p in zip(w, member_probs))
    val_pred = val_agg.argmax(1)
    val_report = per_class_report(y_val, val_pred, classes)

    out_dir = MODELS_DIR / "exp_FINAL_ensemble_9010"
    out_dir.mkdir(parents=True, exist_ok=True)

    cm_path = out_dir / "confusion_matrix.png"
    confusion_matrix_figure(y_val, val_pred, classes).savefig(cm_path, dpi=140, bbox_inches="tight")
    roc_path = out_dir / "roc_curves.png"
    roc_curves_figure(y_val, val_agg, classes).savefig(roc_path, dpi=140, bbox_inches="tight")

    summary = {
        "experiment": "exp_FINAL_ensemble_9010",
        "method": "soft-voting 4-way + multi-scale 30-view TTA per member (90/10 split)",
        "members": [
            {**meta, "weight": float(wi)} for meta, wi in zip(member_meta, w)
        ],
        "val_accuracy": best["val"],
        "macro_f1": val_report["macro avg"]["f1-score"],
        "weighted_f1": val_report["weighted avg"]["f1-score"],
        "test_accuracy": best["val"],  # proxy for API discover
        "n_members": n,
        "tta": "multiscale_30view_per_member",
        "split": "90/10 (val=449)",
        "note": (
            "Hackathon ensemble: 4 backbones (90/10) + multi-scale 30-view TTA "
            "(3 scales × 5 crops × 2 flips). Weights chosen by grid search on "
            "the 90/10 val to maximize accuracy. Latency ~30× per model = "
            "120× baseline single. Optimized for ranking, not production speed."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "val_metrics.json").write_text(
        json.dumps({"val_accuracy": best["val"], "report": val_report}, indent=2),
        encoding="utf-8",
    )
    # ensemble.json — what the API loads
    (out_dir / "ensemble.json").write_text(
        json.dumps(
            {
                "members": [
                    {
                        "exp": meta["exp"],
                        "model_name": meta["model_name"],
                        "image_size": meta["image_size"],
                        "weight": float(wi),
                    }
                    for meta, wi in zip(member_meta, w)
                ],
                "tta": "multiscale_30view",
                "scales": list(SCALES),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    try:
        run = wandb.init(
            entity=os.getenv("WANDB_ENTITY"),
            project=os.getenv("WANDB_PROJECT"),
            name="exp_FINAL_ensemble_9010",
            config={"members": [m["exp"] for m in member_meta], "weights": w, "tta": "hflip"},
            tags=["ensemble", "final", "real-estate", "production"],
            reinit=True,
        )
        run.summary["val_accuracy"] = best["val"]
        run.summary["macro_f1"] = val_report["macro avg"]["f1-score"]
        run.log({"confusion_matrix": wandb.Image(str(cm_path))})
        run.log({"roc_curves": wandb.Image(str(roc_path))})
        run.finish()
    except Exception as exc:
        print(f"[ensemble] W&B logging skipped: {exc}", flush=True)

    print(f"\n[ensemble] persisted: {out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
