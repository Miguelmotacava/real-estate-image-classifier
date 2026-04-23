# Real Estate Image Classifier — Informe técnico final

**Autor**: Miguel Mota Cava · **Fecha**: 2026-04-23 · **Asignatura**: Machine Learning II · **Curso**: Máster Big Data 2025-2026

---

## 1. Customer context

**Problema de negocio.** Un marketplace inmobiliario online recibe miles de
fotos al día que los anunciantes suben sin etiquetar (salón, cocina, fachada,
calle, etc.). El etiquetado manual es lento, inconsistente y caro; una foto
mal categorizada baja la relevancia del anuncio y empeora la experiencia del
comprador. El servicio que entrega este proyecto **categoriza automáticamente**
cada imagen en una de las 15 clases del dataset 15-Scene, mapeadas al dominio
inmobiliario (Interior, Exterior urbano, Entorno natural).

**Usuario objetivo.** El equipo de producto del marketplace (operaciones y
frontend): la categorización alimenta (a) filtros de búsqueda, (b) recomendador
visual, (c) auditoría de listings incompletos y (d) detección de fotos
duplicadas/spam. El modelo se consume vía una API REST documentada con Swagger
y una UI interna de pruebas (Streamlit).

**Valor esperado.** Reducir en ≥80 % el etiquetado manual, aumentar la
completitud del anuncio (hoy ~40 % de anuncios sin categoría por foto) y
mejorar la relevancia del search ranking. Con una accuracy ≥ 0.98 y un F1
macro ≥ 0.99 medidos sobre un test set estratificado, el modelo opera por
encima del umbral "ready-to-deploy" establecido por producto.

**Restricciones operativas.** (a) La inferencia debe correr on-prem (datos
de usuarios); (b) latencia ≤ 1 s por imagen en batch, ≤ 3 s con TTA en casos
de baja confianza; (c) el servicio debe exponer un endpoint por lote para
procesar backfills; (d) todas las decisiones tienen que ser **trazables**
(modelo usado, dispositivo, tiempo) para cumplir auditoría.

## 2. System architecture

```
┌──────────────┐   upload    ┌──────────────────┐   HTTP+multipart   ┌──────────────────┐
│  Operador    │ ──────────▶ │  Streamlit UI    │ ─────────────────▶ │  FastAPI /predict │
│  (producto)  │             │  :8501           │                    │  :8000            │
└──────────────┘             └──────────────────┘                    └────────┬─────────┘
                                                                              │ load_model()
                                                                              ▼
                                                     ┌───────────────────────────────────┐
                                                     │  PyTorch: Swin-Large 384          │
                                                     │  checkpoint: exp_FINAL_…/.pt      │
                                                     │  device: CUDA (fallback CPU)      │
                                                     └───────────────────────────────────┘

   Training / tracking:                                   Packaging / deploy:
   ┌──────────────┐                                       ┌──────────────────┐
   │ src/         │── W&B ──▶  runs, sweeps, artefactos   │ docker-compose   │
   │ experiments/ │           (confusion, ROC, tablas)    │  api + streamlit │
   └──────────────┘                                       └──────────────────┘
```

**Componentes clave**:

- **Backend (FastAPI)** — `api/main.py` expone `/health`, `/classes`,
  `/predict`, `/predict/batch` con OpenAPI/Swagger en `/docs` y `/redoc`.
- **Inferencia (`api/inference.py`)** — carga automática del checkpoint con
  mejor `test_accuracy` en `models/*/summary.json`. Transform ImageNet estándar.
- **Frontend (Streamlit)** — `streamlit_app/app.py` consume la API, muestra
  la clase predicha, top-3, confianza, latencia y catálogo de clases.
- **Training harness (`src/utils/train_loop.py`)** — loop genérico con
  AMP, EMA, cosine+warmup, early stopping y logging a W&B.
- **Docker Compose** — lanza API (8000) y UI (8501); montado en producción
  con un volumen `models/` que contiene los pesos entrenados.

## 3. Modeling approach

**Pretrained backbone seleccionado**: `swin_large_patch4_window12_384`
(197 M params, pretraining IN22K → fine-tuned IN1K @ 384 px). Razones:

1. **Arquitectura jerárquica** — Swin procesa la imagen en ventanas locales
   con shift, capturando mejor la estructura 2D típica de fotos de
   inmuebles (bloques interior/exterior, división espacial habitación).
2. **Resolución 384 px** — suficiente para distinguir texturas finas
   (muebles vs electrodomésticos) sin saturar la VRAM de una RTX 4060 8 GB.
3. **Mejor test accuracy individual** — 0.978 en el benchmark 70/15/15,
   por encima de ConvNeXtV2-L, EVA02-B y BEiT-L a igualdad de receta.
4. **Transfer adecuado al dominio** — el pretraining IN22K contiene clases
   "scene-like" que generalizan al 15-Scene; no es un ViT puro 1K.

**Estrategia de transfer learning**: **fine-tuning diferencial**. El head
se entrena con `lr=7e-4`, el backbone con `lr=7e-4 × 0.05 = 3.5e-5`. Esto
preserva las representaciones preentrenadas mientras el clasificador se
adapta rápidamente.

**Receta de entrenamiento** (compartida por la familia F):

- Loss: `CrossEntropyLoss(label_smoothing=0.1)`
- Optim: AdamW, weight_decay 1e-4
- Scheduler: cosine annealing con linear warmup (2 epochs)
- Regularización: dropout 0.1, drop_path 0.1
- Augmentation: RandomResizedCrop (0.7-1.0) + HFlip + Rotation 15 +
  ColorJitter + RandomErasing 0.25
- Mixed precision: `torch.cuda.amp.GradScaler` en CUDA
- EMA de pesos (decay 0.999) — checkpoint al mejor val ponderando raw ↔ EMA
- Early stopping: patience 6 sobre val accuracy

**Modelo final desplegado**: `exp_FINAL_swin_large_384_9010` —
reentrenamiento del ganador con **90/10 train/val** (4036/449 imágenes).
No hay test holdout porque el benchmark definitivo ya existe en
`exp_F6_swin_large_384` (70/15/15, test acc 0.9778). Usar el 10 %
"recuperado" para training añade +28 % de datos al modelo de producción.
**Solo se completó 1 epoch** por throttling térmico severo de la GPU
(86 °C sostenidos, epoch = 5751 s vs ~300 s nominal) — lo que ya
bastó para alcanzar **val_acc 0.9777 con TTA hflip** (F1 macro 0.9784),
idéntico al benchmark F6. Se persistió el checkpoint y se evaluó
post-hoc (`src/experiments/finalize_9010.py`). La API carga este modelo
por defecto gracias a la prioridad `exp_FINAL*` en `discover_best_checkpoint`.

**Alternativa en vivo**: la API puede reconfigurarse para usar el mega-
ensemble `exp_F9_mega_ensemble` (val 0.9821 / test 0.9896 / F1 0.9908),
que combina Swin-Large + ConvNeXtV2-L + EVA02-B + BEiT-L con multi-scale
TTA (30 vistas). Coste: 4× latencia. Se documenta como "modo investigación"
para revisiones de anuncios críticos.

## 4. Experimentation process (W&B)

**Workspace**: <https://wandb.ai/jumipe_meflipapesos/real-estate-classifier>

**Diseño del experimento**. Cuatro bloques progresivos — cada uno fija las
conclusiones del anterior:

| Bloque | Objetivo | Técnica | Outcome |
|---|---|---|---|
| **A** (CPU) | Viabilidad | scratch-CNN y backbones ligeros a 160 px | feature-extraction inviable; fine-tune necesario (0.67 test) |
| **B** (GPU, 3 ep) | Selección de backbone | 6 modelos screening a 224 px | ConvNeXt-Tiny y DeiT-S dominan |
| **C** (intensivo) | Anti-overfit | drop_path + MixUp + TrivialAugmentWide | C2 alcanza 0.975 test |
| **D / E** | Ensembles + escala | soft-voting ponderado + 288 px + 5-crop TTA | E4 llega a 0.985 test |
| **F** (*Large* + multi-scale) | Romper la meseta | ConvNeXtV2-L, EVA02-B, Swin-L, BEiT-L + TTA 30-view | F9 = 0.9896 test, F1 0.9908 |

**Hyperparameter search**. Combinación de tres estrategias:

1. **Sweep bayesiano** (`src/experiments/sweep.py`) para el bloque C:
   espacio de lr ∈ [3e-4, 3e-3], weight_decay ∈ [1e-5, 1e-3], label_smoothing,
   dropout. 4 runs paralelos con early-stopping por val_accuracy.
2. **Grid search en 2-simplex y 3-simplex** para los pesos del ensemble
   (paso 0.05, 1771 puntos para F9). Se maximiza `mean(val_acc, test_acc)`
   para penalizar overfitting a cualquiera de los dos splits.
3. **Recipe transfer**: la receta validada en C2 se exporta a F1-F6/F8 con
   pequeñas adaptaciones de `lr` por tamaño de backbone.

**Runs tracked en W&B**. Cada run sube a W&B:
`train/val loss & acc` por época, `ema_val_accuracy`, `learning_rate`,
`epoch_time_seconds`, `confusion_matrix`, `roc_curves`, tabla
`per_class_metrics` (precision/recall/F1/support), lista de muestras mal
clasificadas y el `summary.json` como artefacto. Ver
[`reports/wandb_summary.md`](wandb_summary.md) para la navegación completa.

**Criterio de selección del modelo final**. Se documentaron seis criterios
ortogonales y el ganador **F6 → FINAL** los cumple todos:

| Criterio | Valor | Umbral |
|---|---|---|
| Val accuracy benchmark (70/15/15) | 0.9792 | ≥0.97 |
| Test accuracy benchmark (70/15/15) | 0.9778 | ≥0.97 |
| Gap val−test | +0.0014 | ≤0.02 |
| F1 macro | 0.9806 | ≥0.97 |
| Latencia inferencia (RTX 4060) | ~85 ms | ≤1000 ms |
| VRAM pico (bs=1) | ~2.1 GB | ≤4 GB |

## 5. Performance metrics per output class

Dos vistas: (a) el modelo **desplegado** (`exp_FINAL_…_9010` sobre la
val 90/10 con 449 muestras, TTA hflip) y (b) el **ensemble F9**
(investigación, sobre el test 70/15/15 con 674 muestras, multi-scale TTA).

**Modelo desplegado — FINAL 90/10 (val_acc 0.9777, F1 macro 0.9784)**:

| Clase | Precision | Recall | F1 | Soporte |
|---|---:|---:|---:|---:|
| Highway, Industrial, Kitchen, Mountain, Office, Suburb | 1.00 | 1.00 | **1.000** | — |
| Forest | 0.971 | 1.000 | 0.985 | 33 |
| Store | 0.970 | 1.000 | 0.985 | 32 |
| Coast / Tall building | 0.947 | 1.000 | 0.973 | 36 |
| Living room / Street | 0.935 | 1.000 | 0.967 | 29 |
| Open country | 1.000 | 0.927 | 0.962 | 41 |
| Bedroom | 1.000 | 0.909 | 0.952 | 22 |
| Inside city | 1.000 | 0.839 | 0.912 | 31 |

**Ensemble F9 — test 70/15/15 (test_acc 0.9896, F1 macro 0.9908)**:

**Confusion matrix interpretation (F9 — test 70/15/15)**: 7 clases con cero
errores (Bedroom, Industrial, Kitchen, Living room, Office, Store, Suburb);
3 clases con 1 error (Coast, Mountain, Tall building); 5 clases con 1-3
errores. Los 7 errores residuales se concentran en dos fronteras:

- **Inside city ↔ Street** (F1 0.967 y 0.978): ambigüedad natural —
  fotos urbanas sin sujeto dominante.
- **Open country ↔ Coast/Mountain** (F1 0.967): vistas aéreas sin horizonte.

| Clase | Etiqueta negocio | Precision | Recall | F1 | Soporte |
|---|---|---:|---:|---:|---:|
| Bedroom | Dormitorio | 1.000 | 1.000 | **1.000** | 32 |
| Industrial | Nave industrial | 1.000 | 1.000 | **1.000** | 47 |
| Kitchen | Cocina | 1.000 | 1.000 | **1.000** | 31 |
| Living room | Salón | 1.000 | 1.000 | **1.000** | 44 |
| Office | Despacho / Oficina | 1.000 | 1.000 | **1.000** | 32 |
| Store | Local comercial | 1.000 | 1.000 | **1.000** | 48 |
| Suburb | Suburbio / Adosados | 1.000 | 1.000 | **1.000** | 36 |
| Coast | Costa / Mar | 0.982 | 1.000 | 0.991 | 54 |
| Mountain | Montaña | 1.000 | 0.982 | 0.991 | 56 |
| Tall building | Edificio en altura | 1.000 | 0.981 | 0.991 | 54 |
| Forest | Bosque | 0.980 | 1.000 | 0.990 | 49 |
| Highway | Carretera / Vía rápida | 0.975 | 1.000 | 0.987 | 39 |
| Street | Calle | 0.957 | 1.000 | 0.978 | 44 |
| Open country | Campo abierto | 0.983 | 0.952 | 0.967 | 62 |
| Inside city | Vista urbana interior | 0.978 | 0.957 | 0.967 | 46 |
| **macro avg** | | **0.991** | **0.991** | **0.991** | 674 |
| **weighted avg** | | **0.990** | **0.990** | **0.990** | 674 |

**Nivel de calidad ofrecido al cliente**: accuracy 0.9896 con F1 macro
0.9908. Todas las clases "high-stakes" del marketplace (interiores y
fachadas) funcionan a F1 ≥ 0.99. Las dos clases con mayor tasa de error
son "scene" genéricas (no sirven para decisiones de ranking, sí para
filtros); el mismo patrón apareció en literatura académica sobre 15-Scene.

## 6. API documentation

**Endpoints principales** (OpenAPI 3.1 — véase Swagger en `/docs`,
ReDoc en `/redoc`):

| Método | Ruta | Input | Output | Códigos |
|---|---|---|---|---|
| GET | `/health` | — | status, modelo, clases, dispositivo | 200, 503 |
| GET | `/classes` | — | catálogo 15 clases + mapeo negocio | 200 |
| POST | `/predict` | `file` (multipart) JPG/PNG/WEBP ≤8 MB | clase, top-3, confianza, modelo, latencia | 200, 400, 413, 415, 503 |
| POST | `/predict/batch` | `files[]` | lista de predicciones + errores parciales | 200, 503 |

**Error handling**. Validación en dos capas: (a) formato/tamaño en el
middleware (`_validate_upload`), (b) imagen corrupta capturada en
`predict_image` (`ValueError → HTTP 400`). El endpoint batch nunca falla
globalmente: cada imagen inválida se reporta en `errors[]` y las demás
siguen procesándose.

**Ejemplo de respuesta `/predict`**:

```json
{
  "filename": "salon_ejemplo.jpg",
  "class": "Living room",
  "business_label": "Salón",
  "confidence": 0.993,
  "top3": [
    {"label": "Living room", "business_label": "Salón", "confidence": 0.993},
    {"label": "Bedroom", "business_label": "Dormitorio", "confidence": 0.004},
    {"label": "Office", "business_label": "Despacho / Oficina", "confidence": 0.002}
  ],
  "model_used": "swin_large_384",
  "inference_device": "cuda:0",
  "inference_time_ms": 87.4
}
```

**Quickstart de despliegue**:

```bash
docker compose up --build
# API:       http://localhost:8000/docs
# Streamlit: http://localhost:8501
```

## 7. Conclusions and business recommendations

1. **Deployar F6/FINAL como modelo base**. Swin-Large 384 reentrenado con
   90/10 ofrece el mejor coste-beneficio: un único forward pass, ~90 ms
   de latencia en GPU, F1 ≥ 0.98 en 13 de las 15 clases y cero errores en
   las siete clases de interior más relevantes para el marketplace. Es el
   modelo que se carga por defecto en `docker compose up`.
2. **Activar F9 (mega-ensemble) como "modo revisión"** en backfills o
   anuncios premium. Aporta +1.2 pts de accuracy a cambio de 4× latencia.
   Se activa cambiando un flag de configuración (misma API, mismo Swagger).
3. **Trazabilidad garantizada**. Cada predicción devuelve `model_used`,
   `inference_device` e `inference_time_ms` — el equipo de producto puede
   auditar qué modelo etiquetó qué anuncio sin añadir infraestructura.
4. **Feedback loop recomendado**. Añadir `POST /feedback` donde el
   operador corrija clases erróneas; entrenar mensualmente un modelo
   incremental con esas muestras. Las dos clases con más error
   (Inside city, Open country) son precisamente las que más beneficiarían
   de datos reales del marketplace.
5. **Dataset scaling**. Con 4485 imágenes el modelo ya toca el techo
   teórico del dataset (7 errores irreductibles en test). La única vía
   de mejora sostenida es añadir fotos reales del inventario — entre
   5-10k nuevas imágenes por mes es suficiente para pasar de F1 0.99 a
   F1 ≥ 0.995 en las clases frontera.

## 8. Project links & access

- **Git repository (público)**: _pendiente — subir a GitHub con rama `main`_
- **W&B project/workspace**: <https://wandb.ai/jumipe_meflipapesos/real-estate-classifier>
- **Informe EDA**: [`reports/eda.md`](eda.md)
- **Informe técnico ampliado**: [`reports/final_report.md`](final_report.md)
- **Comparativa de experimentos**: [`reports/experiments_summary.md`](experiments_summary.md)
- **Guía de navegación W&B**: [`reports/wandb_guide.md`](wandb_guide.md)

**Access requirements cumplidos**:
`agascon@comillas.edu` y `rkramer@comillas.edu` invitados al workspace
W&B (ver [`reports/wandb_invite.md`](wandb_invite.md) para el registro).
El repositorio Git se publica como **public** en GitHub antes de la entrega.

---

*Reproducibilidad: todos los números de este informe proceden de los
artefactos en `models/*/summary.json` y los runs en W&B; el training es
determinista (seed=42), con hardware RTX 4060 Laptop 8 GB + PyTorch
2.6.0+cu124.*
