"""Aggregate per-experiment summaries into a single markdown table."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"


def collect() -> pd.DataFrame:
    rows = []
    for summary_path in sorted(MODELS_DIR.glob("*/summary.json")):
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        rows.append(data)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    keep = [
        "experiment", "model", "transfer_strategy",
        "best_val_accuracy", "test_accuracy", "macro_f1", "weighted_f1", "epochs_run",
    ]
    for col in keep:
        if col not in df.columns:
            df[col] = None
    df = df[keep].sort_values("test_accuracy", ascending=False)
    return df


def main() -> Path:
    df = collect()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = REPORTS_DIR / "experiments_summary.csv"
    out_md = REPORTS_DIR / "experiments_summary.md"
    if df.empty:
        out_md.write_text("# Experimentos\n\nA\u00fan no hay resultados.\n", encoding="utf-8")
        return out_md
    df.to_csv(out_csv, index=False)
    md = ["# Comparativa de experimentos\n\n"]
    md.append(df.round(4).to_markdown(index=False))
    md.append("\n")
    out_md.write_text("".join(md), encoding="utf-8")
    return out_md


if __name__ == "__main__":
    print(main())
