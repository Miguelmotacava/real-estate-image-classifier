"""Exploratory Data Analysis for the real-estate scene dataset.

Generates ``reports/eda.md`` with quantitative summaries plus a
``reports/sample_grid.png`` figure showcasing one example per class.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "dataset"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Business mapping: real-estate marketplace relevance
BUSINESS_MAPPING = {
    "Bedroom": ("Interior", "Dormitorio"),
    "Kitchen": ("Interior", "Cocina"),
    "Living room": ("Interior", "Sal\u00f3n"),
    "Office": ("Interior", "Despacho / Oficina"),
    "Store": ("Interior", "Local comercial"),
    "Industrial": ("Interior", "Nave industrial"),
    "Inside city": ("Exterior urbano", "Vista urbana interior"),
    "Tall building": ("Exterior urbano", "Edificio en altura"),
    "Street": ("Exterior urbano", "Calle"),
    "Suburb": ("Exterior urbano", "Suburbio / Adosados"),
    "Highway": ("Exterior urbano", "Carretera / V\u00eda r\u00e1pida"),
    "Coast": ("Entorno natural", "Costa / Mar"),
    "Forest": ("Entorno natural", "Bosque"),
    "Mountain": ("Entorno natural", "Monta\u00f1a"),
    "Open country": ("Entorno natural", "Campo abierto"),
}


def collect_split_stats(split_dir: Path) -> pd.DataFrame:
    """Return per-image metadata (class, size, mode) for the given split."""
    rows = []
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            try:
                with Image.open(img_path) as im:
                    rows.append(
                        {
                            "split": split_dir.name,
                            "class": class_dir.name,
                            "path": str(img_path.relative_to(ROOT)),
                            "width": im.width,
                            "height": im.height,
                            "mode": im.mode,
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] cannot open {img_path}: {exc}")
    return pd.DataFrame(rows)


def build_sample_grid(df: pd.DataFrame, output: Path, n_per_class: int = 1) -> None:
    """Save a compact grid image with ``n_per_class`` samples per class."""
    classes = sorted(df["class"].unique())
    cols = 5
    rows = int(np.ceil(len(classes) * n_per_class / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.6))
    axes = np.array(axes).reshape(-1)

    for idx, cls in enumerate(classes):
        sample = df[df["class"] == cls].sample(n_per_class, random_state=SEED)
        for j, (_, row) in enumerate(sample.iterrows()):
            ax = axes[idx * n_per_class + j]
            with Image.open(ROOT / row["path"]).convert("RGB") as im:
                ax.imshow(im)
            ax.set_title(cls, fontsize=9)
            ax.axis("off")

    for k in range(len(classes) * n_per_class, len(axes)):
        axes[k].axis("off")

    fig.suptitle("Muestra por clase (15-Scene Dataset)", fontsize=12)
    fig.tight_layout()
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_distribution(df: pd.DataFrame, output: Path) -> None:
    """Bar plot of class frequencies per split."""
    counts = df.groupby(["class", "split"]).size().unstack(fill_value=0)
    counts = counts.sort_values("training", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    counts.plot(kind="bar", stacked=False, ax=ax, color=["#1f77b4", "#ff7f0e"])
    ax.set_title("Distribuci\u00f3n de im\u00e1genes por clase y split")
    ax.set_ylabel("N\u00famero de im\u00e1genes")
    ax.set_xlabel("Clase")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def write_report(df: pd.DataFrame, sample_grid: Path, dist_plot: Path) -> Path:
    """Render the markdown EDA report."""
    train = df[df["split"] == "training"]
    val = df[df["split"] == "validation"]

    per_class = (
        df.groupby(["class", "split"]).size().unstack(fill_value=0).sort_index()
    )
    per_class["total"] = per_class.sum(axis=1)
    per_class = per_class.sort_values("total", ascending=False)

    sizes = df.assign(area=df["width"] * df["height"])
    width_q = sizes["width"].describe(percentiles=[0.1, 0.5, 0.9]).round(1)
    height_q = sizes["height"].describe(percentiles=[0.1, 0.5, 0.9]).round(1)
    mode_counts = Counter(df["mode"])

    cv_train = train.groupby("class").size()
    imbalance_ratio = round(cv_train.max() / cv_train.min(), 2)

    md = []
    md.append("# Exploratory Data Analysis - Real Estate Image Classifier\n")
    md.append("## 1. Visi\u00f3n general del dataset\n")
    md.append(
        f"- **Splits provistos**: training ({len(train)} im\u00e1genes), "
        f"validation ({len(val)} im\u00e1genes)\n"
        f"- **Total de im\u00e1genes**: {len(df)}\n"
        f"- **N\u00famero de clases**: {df['class'].nunique()}\n"
        f"- **Modo de color**: {dict(mode_counts)}\n"
    )

    md.append("## 2. Mapeo a categor\u00edas de negocio (marketplace inmobiliario)\n")
    md.append("| Clase original | Familia de negocio | Etiqueta para el cliente |\n")
    md.append("|---|---|---|\n")
    for cls in sorted(df["class"].unique()):
        family, label = BUSINESS_MAPPING.get(cls, ("(sin mapear)", cls))
        md.append(f"| {cls} | {family} | {label} |\n")
    md.append("\n")

    md.append("## 3. Distribuci\u00f3n por clase\n")
    md.append("| Clase | training | validation | total |\n|---|---|---|---|\n")
    for cls, row in per_class.iterrows():
        md.append(
            f"| {cls} | {int(row.get('training', 0))} | "
            f"{int(row.get('validation', 0))} | {int(row['total'])} |\n"
        )
    md.append(
        f"\n- Ratio de desbalance en training (max/min): **{imbalance_ratio}x** - "
        "moderado, las clases minoritarias (Kitchen=110, Office=115, Bedroom=116) "
        "frente a las mayoritarias (Open country=310, Mountain=274) requieren "
        "estratificaci\u00f3n y, opcionalmente, sample weights.\n"
    )

    md.append("## 4. Resoluci\u00f3n de las im\u00e1genes\n")
    md.append("| Estad\u00edstico | Width | Height |\n|---|---|---|\n")
    for stat in ["mean", "std", "min", "10%", "50%", "90%", "max"]:
        md.append(
            f"| {stat} | {width_q.get(stat, 'NA')} | {height_q.get(stat, 'NA')} |\n"
        )
    md.append(
        "\nLas im\u00e1genes son t\u00edpicas del 15-Scene benchmark: en torno a "
        "256x256, en su mayor\u00eda escala de grises. Para transfer learning con "
        "backbones ImageNet conviene **convertir a RGB** y reescalar a 224x224.\n\n"
    )

    md.append("## 5. Estrategia de transfer learning recomendada\n")
    md.append(
        "Tama\u00f1o moderado (~3.000 im\u00e1genes de training) y dominio "
        "razonablemente alineado con ImageNet (escenas naturales y objetos "
        "comunes). Recomendaciones:\n\n"
        "1. **Backbones preentrenados en ImageNet** son la mejor relaci\u00f3n "
        "coste/beneficio: MobileNetV3-Small, EfficientNetB0/B3, ResNet50.\n"
        "2. Pipeline en dos fases: (a) **feature extraction** con backbone "
        "congelado durante 3-5 \u00e9pocas para estabilizar el clasificador; "
        "(b) **fine-tuning parcial** de las \u00faltimas capas con LR diferencial "
        "(LR backbone = LR head / 10).\n"
        "3. Aumentado fuerte: flips horizontales, rotaci\u00f3n \u00b115\u00b0, jitter "
        "de color, RandomResizedCrop(0.7-1.0), Cutout/RandomErasing. La mezcla "
        "mixup/cutmix solo en el modelo grande dado el coste en CPU.\n"
        "4. Loss con **label smoothing 0.1** y **stratified split 70/15/15** "
        "(reagrupando train+val originales para construir los tres conjuntos).\n"
        "5. Evaluaci\u00f3n: matriz de confusi\u00f3n, F1 por clase y ROC-AUC OvR. "
        "Atenci\u00f3n especial a parejas confundibles desde una perspectiva de "
        "negocio (Living room \u2194 Bedroom, Inside city \u2194 Tall building, "
        "Open country \u2194 Mountain).\n\n"
    )

    md.append("## 6. Restricciones de hardware detectadas\n")
    md.append(
        "El entorno actual es **CPU-only** (PyTorch 2.7.0+cpu). Esto:\n\n"
        "- Bloquea el Bloque C (ViT/Swin/DeiT) por coste prohibitivo.\n"
        "- Limita el Bloque B a backbones ligeros (EfficientNetB0/B3) con "
        "pocas \u00e9pocas y batch size reducido.\n"
        "- Aplicamos `torch.set_num_threads(os.cpu_count())`, "
        "`pin_memory=False`, `num_workers` bajo (riesgo de cuelgues en "
        "Windows/Python 3.13) y entrenamiento determinista con SEED=42.\n\n"
    )

    md.append("## 7. Visualizaciones\n")
    md.append(
        f"![Distribuci\u00f3n por clase]({dist_plot.name})\n\n"
        f"![Sample grid]({sample_grid.name})\n"
    )
    out = REPORTS_DIR / "eda.md"
    out.write_text("".join(md), encoding="utf-8")
    return out


def main() -> None:
    print(f"Reading dataset from {DATA_DIR}")
    df_train = collect_split_stats(DATA_DIR / "training")
    df_val = collect_split_stats(DATA_DIR / "validation")
    df = pd.concat([df_train, df_val], ignore_index=True)
    df.to_csv(REPORTS_DIR / "eda_metadata.csv", index=False)

    summary = {
        "n_training": int(len(df_train)),
        "n_validation": int(len(df_val)),
        "n_classes": int(df["class"].nunique()),
        "classes": sorted(df["class"].unique().tolist()),
        "mean_width": float(df["width"].mean()),
        "mean_height": float(df["height"].mean()),
    }
    (REPORTS_DIR / "eda_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    sample_grid = REPORTS_DIR / "sample_grid.png"
    dist_plot = REPORTS_DIR / "class_distribution.png"
    build_sample_grid(df, sample_grid)
    plot_distribution(df, dist_plot)
    report = write_report(df, sample_grid, dist_plot)
    print(f"EDA report written to {report}")


if __name__ == "__main__":
    main()
