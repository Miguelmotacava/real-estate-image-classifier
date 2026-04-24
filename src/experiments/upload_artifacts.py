"""Upload already-trained FINAL/ensemble checkpoints to W&B as artifacts.

The scripts in final_9010.py / final_member_9010.py now call
``wandb.log_artifact`` at the end of each training, but the models we
already have on disk were trained before that change. This script
creates a one-shot W&B run per experiment, attaches the .pt checkpoint
and metadata, and finishes the run — giving us versioned, downloadable
artifacts for reproducibility and teammate onboarding.

Usage:
    python -m src.experiments.upload_artifacts
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import wandb
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"

# (experiment_dir, artifact_type, extra_files)
TARGETS = [
    ("exp_FINAL_swin_large_384_9010", "model", []),
    ("exp_FINAL_F3_9010", "model", []),
    ("exp_FINAL_F4_9010", "model", []),
    ("exp_FINAL_F8_9010", "model", []),
    ("exp_FINAL_ensemble_9010", "ensemble-config", ["ensemble.json"]),
]


def _upload(exp: str, atype: str, extras: list[str]) -> None:
    exp_dir = MODELS_DIR / exp
    if not exp_dir.exists():
        print(f"[skip] {exp} — directory missing")
        return
    summary_path = exp_dir / "summary.json"
    if not summary_path.exists():
        print(f"[skip] {exp} — no summary.json")
        return

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    run = wandb.init(
        entity=os.getenv("WANDB_ENTITY"),
        project=os.getenv("WANDB_PROJECT"),
        name=f"{exp}_upload_artifact",
        job_type="upload-artifact",
        tags=["artifact-upload", "reproducibility"],
        config=summary,
        reinit=True,
    )
    try:
        val_acc = float(summary.get("val_accuracy") or summary.get("test_accuracy") or 0.0)
        artifact = wandb.Artifact(
            name=f"model-{exp}", type=atype,
            description=f"Checkpoint/config for {exp} (val_acc={val_acc:.4f})",
            metadata=summary,
        )
        ckpt = exp_dir / "best_model.pt"
        if ckpt.exists():
            size_mb = ckpt.stat().st_size / 1024 / 1024
            print(f"[{exp}] adding best_model.pt ({size_mb:.1f} MB)")
            artifact.add_file(str(ckpt))
        artifact.add_file(str(summary_path))
        for extra in extras + ["val_metrics.json"]:
            p = exp_dir / extra
            if p.exists():
                artifact.add_file(str(p))
                print(f"[{exp}] + {extra}")
        run.log_artifact(artifact)
        print(f"[{exp}] logged artifact OK")
    finally:
        run.finish()


def main() -> None:
    load_dotenv()
    if not os.getenv("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY missing in environment / .env")
    for exp, atype, extras in TARGETS:
        _upload(exp, atype, extras)


if __name__ == "__main__":
    main()
