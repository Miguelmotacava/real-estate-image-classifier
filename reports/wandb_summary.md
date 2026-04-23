# Weights & Biases - Resumen del workspace

- **Entity**: `jumipe_meflipapesos`
- **Project**: `real-estate-classifier`
- **URL**: <https://wandb.ai/jumipe_meflipapesos/real-estate-classifier>
- **Run de prueba de conexión**: `connection-test`

## Runs registrados

| Run | Tipo | Descripción |
|---|---|---|
| `connection-test` | smoke | Verificación de credenciales W&B antes del entrenamiento. |
| `exp_A1_scratch_cnn` | baseline | CNN ligera entrenada desde cero (referencia inferior). |
| `exp_A2_mobilenetv3_small` | feature extraction → fine-tune | MobileNetV3-Small con backbone congelado y posterior LR diferencial. |
| `exp_A3_efficientnet_b0` | fine-tuning completo | EfficientNetB0 con LR diferencial (head=1e-3, backbone=1e-4). |
| `exp_D1_soft_voting` | ensemble | Soft voting de los mejores backbones. |
| `sweep_*` | bayes sweep | 4 runs sobre MobileNetV3-Small variando LR, batch, dropout, label-smoothing y optimizer. |

## Métricas registradas por época

`train_loss`, `train_acc`, `val_loss`, `val_accuracy`, `learning_rate`, `epoch_time_seconds`.

## Métricas finales (resumen del run)

`test_accuracy`, `macro_f1`, `weighted_f1`, además de:

- Imagen de **matriz de confusión** (normalizada por filas).
- Imagen con **curvas ROC** una vs resto + macro AUC.
- **Tabla** `per_class_metrics` con precision/recall/F1/support por clase.
- Lista de **misclassified examples** (`misclassified.json` por experimento).

## Sweep bayesiano

- **Sweep ID**: `9w6fed9y` · <https://wandb.ai/jumipe_meflipapesos/real-estate-classifier/sweeps/9w6fed9y>
- **Search method**: bayes · **Métrica objetivo**: `val_accuracy` (maximizar)
- **Hiperparámetros explorados**:
  - `learning_rate` log-uniform en [1e-5, 5e-3]
  - `batch_size` ∈ {16, 32}
  - `dropout` ∈ {0.1, 0.2, 0.3}
  - `label_smoothing` ∈ {0.0, 0.1, 0.2}
  - `optimizer` ∈ {adam, adamw, sgd}
- **Presupuesto efectivo**: 3 runs completados + 1 abortado (limitado por CPU,
  el código está preparado para 15+ con GPU).
- **Resultado por run** (orden cronológico):

| Run | LR (head) | BS | Dropout | LSmooth | Optim | Best val_acc |
|---|---:|---:|---:|---:|---|---:|
| `stellar-sweep-1` | 1.74e-5 | 16 | 0.3 | 0.0 | adamw | 0.089 |
| `iconic-sweep-2` | 8.60e-5 | 16 | 0.1 | 0.2 | adam  | **0.235** |
| `clean-sweep-3` | (abortado, sin métrica) | | | | | — |

> Lectura: el sampler bayesiano comenzó explorando la zona baja del espacio
> log-uniform (1e-5 → 1e-4), muy por debajo del LR productivo descubierto
> manualmente para EfficientNetB0 (head=7e-4, backbone=7e-5). Con tan pocos
> runs el sweep no llegó a esa banda — se documenta como evidencia de que
> 3-4 muestras son insuficientes para una búsqueda bayesiana fiable y de la
> necesidad de presupuesto 15+ cuando se disponga de GPU.

## Acceso para los profesores

El proyecto W&B está abierto a los siguientes correos invitados al workspace:

- `agascon@comillas.edu`
- `rkramer@comillas.edu`

(Las invitaciones se gestionan en el menú **Settings → Members** del proyecto.)
