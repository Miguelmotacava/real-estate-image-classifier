"""Upload missing ensemble runs (D1b, D1c, E3 + variants) to W&B.

Training runs (A/B/C/E) log automatically; these soft-voting ensembles were
computed inline so we persist each combination as its own W&B run here.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
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


@dataclass
class EnsembleSpec:
    name: str
    members: list[tuple[str, float]]  # (experiment_name, weight)
    tta_flip: bool = True
    tags: tuple[str, ...] = ("ensemble", "real-estate")


ENSEMBLES: list[EnsembleSpec] = [
    EnsembleSpec(
        "exp_D1b_soft_voting_diverse",
        [
            ("exp_C2_convnext_tiny_regularized", 1.0),
            ("exp_B_deit_small_patch16_224", 1.0),
            ("exp_B_convnext_tiny", 1.0),
        ],
        tta_flip=False,
        tags=("ensemble", "diverse", "real-estate"),
    ),
    EnsembleSpec(
        "exp_D1c_soft_voting_c2c1",
        [
            ("exp_C2_convnext_tiny_regularized", 1.0),
            ("exp_C1_convnext_tiny_intensive", 1.0),
        ],
        tta_flip=False,
        tags=("ensemble", "pair", "real-estate"),
    ),
    EnsembleSpec(
        "exp_E3_ensemble_equal_C2_E2",
        [
            ("exp_C2_convnext_tiny_regularized", 0.5),
            ("exp_E2_convnext_base_288", 0.5),
        ],
        tags=("ensemble", "weighted", "real-estate"),
    ),
    EnsembleSpec(
        "exp_E3_ensemble_C2heavy_C2_E2",
        [
            ("exp_C2_convnext_tiny_regularized", 0.6),
            ("exp_E2_convnext_base_288", 0.4),
        ],
        tags=("ensemble", "weighted", "real-estate"),
    ),
    EnsembleSpec(
        "exp_E3_ensemble_Bheavy_C2_E2",  # CHAMPION
        [
            ("exp_C2_convnext_tiny_regularized", 0.4),
            ("exp_E2_convnext_base_288", 0.6),
        ],
        tags=("ensemble", "weighted", "champion", "real-estate"),
    ),
    EnsembleSpec(
        "exp_E3_ensemble_trio_C2_E1_E2",
        [
            ("exp_C2_convnext_tiny_regularized", 1.0),
            ("exp_E1_convnext_small_288", 1.0),
            ("exp_E2_convnext_base_288", 1.0),
        ],
        tags=("ensemble", "trio", "real-estate"),
    ),
]


def _compute_proba(exp_name: str, split, tta_flip: bool) -> tuple[np.ndarray, np.ndarray]:
    ckpt = MODELS_DIR / exp_name / "best_model.pt"
    loaded = load_model(checkpoint_path=ckpt)
    _, _, test_loader = build_loaders(split, image_size=loaded.image_size, batch_size=16, num_workers=0)
    loaded.model.eval()
    all_probs, all_y = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(loaded.device)
            p = F.softmax(loaded.model(x), dim=1)
            if tta_flip:
                p = (p + F.softmax(loaded.model(torch.flip(x, dims=[3])), dim=1)) / 2.0
            all_probs.append(p.cpu().numpy())
            all_y.extend(y.numpy().tolist())
    return np.concatenate(all_probs, axis=0), np.array(all_y)


def main() -> None:
    load_dotenv()
    entity = os.getenv("WANDB_ENTITY")
    project = os.getenv("WANDB_PROJECT")
    if not os.getenv("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY not set — check your .env")

    paths, labels, classes = discover_dataset()
    split = stratified_split(paths, labels, classes)

    # Cache probs per (model, tta) pair
    proba_cache: dict[tuple[str, bool], np.ndarray] = {}
    y_true: np.ndarray | None = None
    needed = {(m, spec.tta_flip) for spec in ENSEMBLES for m, _ in spec.members}
    for exp_name, tta_flip in sorted(needed):
        print(f"Computing probas: {exp_name}  (tta_flip={tta_flip})")
        probs, y = _compute_proba(exp_name, split, tta_flip)
        proba_cache[(exp_name, tta_flip)] = probs
        y_true = y if y_true is None else y_true

    for spec in ENSEMBLES:
        total_w = sum(w for _, w in spec.members)
        agg = np.zeros_like(proba_cache[(spec.members[0][0], spec.tta_flip)])
        for name, w in spec.members:
            agg += w * proba_cache[(name, spec.tta_flip)]
        agg /= total_w
        pred = agg.argmax(axis=1)
        test_acc = float((y_true == pred).mean())
        report = per_class_report(y_true, pred, classes)

        out_dir = MODELS_DIR / spec.name
        out_dir.mkdir(parents=True, exist_ok=True)
        cm_fig = confusion_matrix_figure(y_true, pred, classes)
        cm_path = out_dir / "confusion_matrix.png"
        cm_fig.savefig(cm_path, dpi=140, bbox_inches="tight")
        roc_fig = roc_curves_figure(y_true, agg, classes)
        roc_path = out_dir / "roc_curves.png"
        roc_fig.savefig(roc_path, dpi=140, bbox_inches="tight")

        summary = {
            "experiment": spec.name,
            "method": "soft-voting" + (" + TTA hflip" if spec.tta_flip else ""),
            "members": [{"name": n, "weight": w} for n, w in spec.members],
            "test_accuracy": test_acc,
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_f1": report["weighted avg"]["f1-score"],
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        (out_dir / "test_metrics.json").write_text(
            json.dumps({"test_accuracy": test_acc, "report": report}, indent=2)
        )

        run = wandb.init(
            entity=entity,
            project=project,
            name=spec.name,
            config={
                "method": summary["method"],
                "members": [n for n, _ in spec.members],
                "weights": [w for _, w in spec.members],
                "tta_flip": spec.tta_flip,
            },
            tags=list(spec.tags),
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
        print(f"  -> {spec.name}: test_acc={test_acc:.4f}  macro_f1={report['macro avg']['f1-score']:.4f}")


if __name__ == "__main__":
    main()
