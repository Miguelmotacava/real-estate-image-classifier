"""Inject the actual experiment numbers into ``reports/final_report.md``.

Looks at every ``models/*/summary.json``, builds a comparison table and
substitutes the ``<!-- RESULTS_TABLE -->`` placeholder if present. Otherwise
appends the table at the end of the report.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"


def _load_summaries() -> pd.DataFrame:
    rows = []
    for summary_path in sorted(MODELS_DIR.glob("**/summary.json")):
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
        "best_val_accuracy", "test_accuracy", "macro_f1", "weighted_f1",
        "epochs_run", "train_seconds",
    ]
    for col in keep:
        if col not in df.columns:
            df[col] = None
    return df[keep].sort_values("test_accuracy", ascending=False).round(4)


def main() -> None:
    df = _load_summaries()
    if df.empty:
        print("No summaries to update")
        return
    table_md = df.to_markdown(index=False)
    summary_md = REPORTS_DIR / "experiments_summary.md"
    summary_md.write_text(
        "# Comparativa de experimentos\n\n" + table_md + "\n", encoding="utf-8"
    )
    print(f"Updated {summary_md}")
    df.to_csv(REPORTS_DIR / "experiments_summary.csv", index=False)


if __name__ == "__main__":
    main()
