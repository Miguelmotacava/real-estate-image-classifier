# Guía rápida — comparar runs y leer accuracy en Weights & Biases

> Workspace: <https://wandb.ai/jumipe_meflipapesos/real-estate-classifier>
> Entity: `jumipe_meflipapesos` · Project: `real-estate-classifier`

Este documento explica **dónde mirar en W&B** para responder las dos preguntas
operativas del proyecto:

1. ¿Cómo comparo varios experimentos entre sí?
2. ¿Dónde se ve la *accuracy* en train/val/test de cada modelo?

---

## 1. Estructura de lo que se loguea

Cada `python -m src.experiments.run_experiment ...` crea **un run** en W&B con
estas piezas:

| Pieza | Dónde se ve | Qué contiene |
|---|---|---|
| **Config** | pestaña *Overview* del run | hyperparámetros (lr, batch, dropout, optimizer, transfer_strategy, image_size, epochs…) — capturados desde `TrainConfig` |
| **Métricas por época** | pestaña *Charts* | `train_loss`, `train_acc`, `val_loss`, `val_accuracy`, `learning_rate`, `epoch_time_seconds` (uno por epoch) |
| **Test summary** | pestaña *Overview → Summary* | `test_accuracy`, `macro_f1`, `weighted_f1` (escalares finales) |
| **Artefactos** | pestaña *Files* y *Media* | `confusion_matrix.png`, `roc_curves.png`, tabla `per_class_metrics` |
| **Tags** | columna *Tags* en la tabla de runs | `transfer-learning`, `real-estate`, `fine_tuning|feature_extraction`, etc. |

> El nombre del run = `--experiment` que pasas en CLI (p. ej.
> `exp_A3_efficientnet_b0`). Los runs del sweep heredan nombres aleatorios
> generados por W&B (`stellar-sweep-1`, `iconic-sweep-2`, …).

---

## 2. Cómo comparar varios experimentos en W&B

### 2.1 Vista de tabla (Runs table)

1. Abrir el proyecto: <https://wandb.ai/jumipe_meflipapesos/real-estate-classifier>
2. La pestaña por defecto **Runs** muestra una tabla con todos los experimentos
   (filas) y columnas. Por defecto solo aparecen *Name* y *State*.
3. Pulsar **Columns** → marcar:
   - `config:model_name`, `config:transfer_strategy`, `config:learning_rate`,
     `config:batch_size`, `config:image_size`, `config:epochs`
   - `summary:test_accuracy`, `summary:macro_f1`, `summary:weighted_f1`,
     `summary:best_val_accuracy`
4. Ordenar la tabla por `summary:test_accuracy` (descendente). Arriba debe
   aparecer `exp_A3_efficientnet_b0` (≈ 0.669).
5. Usar **Filter** para acotar: ej. `Tags = real-estate AND model_name =
   efficientnet_b0`.

### 2.2 Vista de gráficos (Workspace / Charts)

1. Pulsar la pestaña **Workspace** (icono de gráfico de barras).
2. Seleccionar varios runs en el panel izquierdo (checkboxes). Las líneas se
   pintarán en colores distintos en cada gráfico.
3. Las métricas que registramos (`train_acc`, `val_accuracy`, `train_loss`,
   `val_loss`, `learning_rate`, `epoch_time_seconds`) aparecen como gráficos
   automáticos. El eje X es `step` (= epoch).
4. Útil:
   - *Smoothing* a 0 cuando hay solo 2-3 epochs (suavizar deforma).
   - **Mediana de Y axis** = "fixed" y misma escala para comparar tamaños
     de modelo de forma honesta.

### 2.3 Vista de **Sweep**

1. En el panel lateral del proyecto pulsar **Sweeps** → seleccionar
   `9w6fed9y` (real-estate-classifier).
2. La pestaña **Sweep** trae:
   - **Parallel coordinates plot**: cada línea es un run, eje X = cada
     hyperparam, color por `val_accuracy`. Útil para ver qué combinaciones
     funcionan.
   - **Hyperparameter importance**: tabla con la correlación de cada hp con la
     métrica objetivo (`val_accuracy`).
   - **Bayesian search history**: gráfico de scatter con métrica vs run.

### 2.4 Crear un **Report** público

1. Botón **Create Report** (esquina superior derecha del proyecto).
2. Insertar bloques: tabla de runs, gráficos, parallel-coords del sweep, texto.
3. Compartir con el flag *Public link* (los profesores ya tienen acceso vía
   *Members*).

---

## 3. Dónde está la accuracy en train, val y test

| Pregunta | Dónde mirar exactamente |
|---|---|
| **Accuracy de train por época** | run → *Charts* → panel `train_acc`. Cada punto es una época completa. Se loguea con `wandb.log({"train_acc": ...})` desde `src/utils/train_loop.py`. |
| **Accuracy de validación por época** | run → *Charts* → panel `val_accuracy`. Mismo logueo, métrica objetivo del sweep. |
| **Mejor val accuracy del run** | run → *Overview → Summary* → `best_val_accuracy` (escalar = mejor val acc observado en cualquier época). |
| **Accuracy en test set** | run → *Overview → Summary* → `test_accuracy`. Se calcula **una sola vez** tras el entrenamiento, sobre el split test (15 %), recargando el `best_model.pt`. Implementado en `src/experiments/run_experiment.py::evaluate_and_log`. |
| **F1 macro / weighted en test** | *Summary* → `macro_f1`, `weighted_f1`. |
| **Métricas por clase en test** | run → *Tables* → `per_class_metrics` (precision/recall/F1/support por cada una de las 15 clases). |
| **Matriz de confusión y ROC** | run → *Media* → `confusion_matrix`, `roc_curves` (PNG). |

> En la tabla de runs (sección 2.1) las mismas métricas aparecen como columnas
> seleccionables: `summary:train_acc` (último epoch), `summary:val_accuracy`
> (último epoch) y `summary:test_accuracy` (test final).

---

## 4. Localmente (sin abrir W&B)

Cada experimento deja una huella idéntica en disco bajo `models/<exp>/`:

| Fichero | Contenido |
|---|---|
| `summary.json` | `best_val_accuracy`, `test_accuracy`, `macro_f1`, `weighted_f1`, `epochs_run` |
| `history.json` | lista de dicts por época con `train_loss`, `train_acc`, `val_loss`, `val_accuracy`, `learning_rate`, `epoch_time_seconds` |
| `test_metrics.json` | `test_accuracy` + `report` por clase |
| `confusion_matrix.png` / `roc_curves.png` | versiones locales de los artefactos |
| `misclassified.json` | top errores cualitativos (path, true, pred, confidence) |

Comando para listar accuracies locales:

```bash
python - <<'PY'
import json, glob
for s in sorted(glob.glob("models/*/summary.json")):
    d = json.load(open(s))
    print(f"{d['experiment']:<35}  val={d['best_val_accuracy']:.3f}  test={d['test_accuracy']:.3f}  f1={d['macro_f1']:.3f}")
PY
```

O directamente la comparativa Markdown: `python -m src.experiments.update_report`
genera [`experiments_summary.md`](experiments_summary.md) y `.csv`.

---

## 5. Resumen visual rápido (paso a paso)

1. <https://wandb.ai/jumipe_meflipapesos/real-estate-classifier> → **Runs**.
2. *Columns* → añadir `summary:test_accuracy` y `config:model_name`.
3. Ordenar por `test_accuracy` ↓ → el ganador es **exp_A3_efficientnet_b0**.
4. Pulsar el run → *Overview* (config + summary) y *Charts* (curvas por época).
5. Para el sweep: *Sweeps* → `9w6fed9y` → *Parallel coordinates* + *Hp importance*.
6. Para comparar dos runs en el mismo gráfico: marcar ambos con el checkbox
   en *Workspace* y elegir el panel que interese (`val_accuracy`, `train_loss`,
   etc.).
