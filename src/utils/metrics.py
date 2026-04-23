"""Evaluation metrics and W&B-friendly artifact builders."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    tta: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference and return (y_true, y_pred, y_proba).

    If ``tta=True``, average softmax over the original batch and its
    horizontal flip — cheap +0.5-1% usually.
    """
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    y_proba: list[np.ndarray] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        proba = F.softmax(model(x), dim=1)
        if tta:
            proba_flip = F.softmax(model(torch.flip(x, dims=[3])), dim=1)
            proba = (proba + proba_flip) / 2.0
        proba_np = proba.cpu().numpy()
        y_true.extend(y.numpy().tolist())
        y_pred.extend(proba_np.argmax(axis=1).tolist())
        y_proba.append(proba_np)
    return np.array(y_true), np.array(y_pred), np.concatenate(y_proba, axis=0)


def per_class_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Sequence[str],
) -> dict:
    """Return sklearn classification report as a dict."""
    return classification_report(
        y_true, y_pred, target_names=list(classes), output_dict=True, zero_division=0
    )


def confusion_matrix_figure(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Sequence[str],
    normalize: bool = True,
) -> plt.Figure:
    """Return a seaborn heatmap with the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    if normalize:
        cm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=list(classes),
        yticklabels=list(classes),
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix" + (" (row-normalized)" if normalize else ""))
    fig.tight_layout()
    return fig


def roc_curves_figure(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: Sequence[str],
) -> plt.Figure:
    """Per-class ROC curves with macro AUC annotation."""
    fig, ax = plt.subplots(figsize=(8, 6))
    aucs = []
    for idx, cls in enumerate(classes):
        binary = (y_true == idx).astype(int)
        if binary.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(binary, y_proba[:, idx])
        try:
            auc = roc_auc_score(binary, y_proba[:, idx])
        except ValueError:
            auc = float("nan")
        aucs.append(auc)
        ax.plot(fpr, tpr, label=f"{cls} (AUC={auc:.2f})", linewidth=1.0)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC curves (macro AUC={np.nanmean(aucs):.3f})")
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    fig.tight_layout()
    return fig


def misclassified_samples(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    paths: Sequence[Path],
    classes: Sequence[str],
    n: int = 12,
) -> list[dict]:
    """Pick a small set of misclassified examples for qualitative inspection."""
    wrong = np.where(y_true != y_pred)[0]
    rng = np.random.default_rng(42)
    rng.shuffle(wrong)
    picked = wrong[:n]
    return [
        {
            "path": str(paths[i]),
            "true": classes[y_true[i]],
            "pred": classes[y_pred[i]],
        }
        for i in picked
    ]
