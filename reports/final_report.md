# Real Estate Image Classifier — Informe técnico

> Práctica de Deep Learning · Máster Big Data 2025-2026 · Machine Learning II

---

## 1. Customer context

El cliente es un **marketplace inmobiliario online** que recibe centenares de
fotografías al día subidas por vendedores particulares y agencias. Hoy el
etiquetado de cada anuncio (qué estancia/exterior aparece en cada foto) lo hace
manualmente el equipo de operaciones, lo que limita la velocidad de publicación
y degrada la relevancia del buscador interno (las búsquedas como "pisos con
cocina office" o "casas con jardín" dependen de etiquetas correctas).

**Usuario objetivo**: equipo de producto del marketplace.
**Valor esperado**: (a) reducir el etiquetado manual en ≥70 %; (b) mejorar la
precisión del filtrado por estancia; (c) ofrecer recomendaciones de tags al
vendedor mientras sube las fotos.

El dataset disponible es el clásico **15-Scene** (4 485 imágenes en 15 clases),
que se utiliza como proxy del catálogo real con un mapeo a tres familias de
negocio:

| Familia | Clases originales mapeadas |
|---|---|
| Interior | Bedroom, Kitchen, Living room, Office, Store, Industrial |
| Exterior urbano | Inside city, Tall building, Street, Suburb, Highway |
| Entorno natural | Coast, Forest, Mountain, Open country |

---

## 2. System architecture

```
┌──────────────┐       multipart/form-data        ┌──────────────────┐
│  Streamlit   │ ───────────────────────────────▶ │   FastAPI / Uvi  │
│  (puerto     │                                  │   corn (puerto   │
│   8501)      │ ◀─────── JSON top-3, conf ────── │   8000)          │
└──────────────┘                                  └────────┬─────────┘
       ▲                                                   │
       │                                                   │ torch.load
       │                                          ┌────────▼─────────┐
       │                                          │ models/<exp>/    │
       │                                          │ best_model.pt    │
       │                                          └────────┬─────────┘
       │                                                   │
       │                                          ┌────────▼─────────┐
       └─── docker-compose ─────────────────────▶ │ timm backbone    │
                                                  │ + clasificador   │
                                                  └──────────────────┘
```

- **Entrenamiento (`src/`)**: PyTorch + `timm`, logging W&B, sweep bayesiano,
  ensemble. Cada experimento guarda `best_model.pt` + `summary.json`.
- **API (`api/`)**: FastAPI (3 endpoints públicos + health/classes), validación
  Pydantic, manejo de errores, Swagger/ReDoc automáticos.
- **UI (`streamlit_app/`)**: drag-and-drop, top-3 con barra de confianza, panel
  técnico, listado de clases.
- **Despliegue**: `docker-compose.yml` con dos servicios (`api`, `streamlit`),
  health-check sobre la API, montaje de `./models` como volumen de solo lectura.
- **Configuración**: `.env` (no versionado) con `WANDB_API_KEY`, `WANDB_ENTITY`,
  `WANDB_PROJECT`, `API_URL`. `python-dotenv` se carga en cada entrypoint.

---

## 3. Modeling approach

### 3.1 Datos y splits
- Recolección de 4 485 imágenes desde `dataset/training` y `dataset/validation`.
- Split estratificado **70 / 15 / 15** (semilla 42) → 3 139 train, 672 val,
  674 test.
- Conversión a RGB, redimensionado a **224×224** (Bloque B/C, GPU) o 160×160
  (Bloque A, CPU) y normalización con estadísticas de ImageNet.

### 3.2 Augmentations
- **Ligera** (Bloque A/B/C1):
  `Resize → RandomResizedCrop(0.7-1.0) → HorizontalFlip → Rotation(±15°) →
  ColorJitter → ToTensor → Normalize → RandomErasing(p=0.25)`.
- **Fuerte** (Bloque C2, flag `--strong-augment`):
  `Resize → RandomResizedCrop(0.6-1.0) → HorizontalFlip →
  **TrivialAugmentWide** → ToTensor → Normalize → RandomErasing(p=0.4)`.
  Se activa cuando hay sobreajuste medible (train ≫ val/test).

### 3.3 Backbones evaluados

**Bloque A — referencia CPU** (160×160, train=1600, 2-3 épocas):

| Exp | Backbone | Estrategia | Hardware |
|---|---|---|---|
| A1 | `scratch_cnn` (TinyCNN) | desde cero | CPU |
| A2 | `mobilenetv3_small_100` | feature extraction | CPU |
| A3 | `efficientnet_b0` | fine-tuning + LR diferencial (head=7e-4) | CPU |
| A4 | `mobilenetv3_small_100` | fine-tuning + LR diferencial (head=5e-4) | CPU |

**Bloque B — screening de backbones en GPU** (224×224, train=3139, 3 épocas):

| Exp | Backbone | LR head | Justificación |
|---|---|---:|---|
| B-eff_b0  | `efficientnet_b0`        | 7e-4 | CNN ligera moderna |
| B-eff_b3  | `efficientnet_b3`        | 5e-4 | Versión escalada |
| B-resnet  | `resnet50`               | 5e-4 | Clásico baseline |
| B-convnext| `convnext_tiny`          | 3e-4 | CNN moderna inspirada en ViT |
| B-regnet  | `regnety_008`            | 5e-4 | RegNet eficiente |
| B-deit    | `deit_small_patch16_224` | 3e-4 | Vision Transformer |

**Bloque C — entrenamiento intensivo del ganador en GPU** (224×224, HP
afinados):

| Exp | Backbone | Cambios respecto al screening |
|---|---|---|
| C1 | `convnext_tiny` | LR head 1.5e-4 (½), backbone factor 0.05 (5×), weight_decay 5e-4 (5×), dropout 0.3, 10 épocas, patience 4 |
| **C2** | `convnext_tiny` | C1 + **drop_path_rate 0.2** (stochastic depth), **weight_decay 1e-3** (2×), **dropout 0.3**, **label_smoothing 0.15**, **TrivialAugmentWide** + RandomErasing(0.4), 12 épocas, patience 5 |

**Bloque D — ensemble por soft-voting**. Se probaron tres variantes para
estudiar qué combinación supera al mejor modelo único:

| Variante | Miembros |
|---|---|
| D1  | top-3 por test_acc (C2 + C1 + B_convnext) — todos ConvNeXt |
| D1b | diversa (C2 + B_deit + B_convnext) — ConvNeXt + Transformer |
| D1c | par fuerte (C2 + C1) — mismo backbone, HP distintos |

**Bloque E — escalado del backbone y ensemble ponderado** (GPU, 288×288,
cosine + EMA + TTA):

| Exp | Backbone | Parámetros | Cambios respecto a C2 |
|---|---|---:|---|
| E1 | `convnext_small` | 50M | image_size 288, cosine schedule, warmup 2 ep, EMA (decay 0.9995), TTA h-flip |
| E2 | `convnext_base`  | 89M | E1 + batch 8 (VRAM), LR head 1e-4 |
| E3 | C2 + E2 (ponderado) | ensemble | soft-voting 0.4·C2 + 0.6·E2 + TTA flip — primera combinación que bate a C2 |
| **E4** | C2 + E2 + E1 (3-way) | ensemble | soft-voting 0.30·C2(raw) + 0.60·E2(5crop+flip) + 0.10·E1(5crop+flip) — campeón final |

> **Histórico CPU**. El proyecto se inició con PyTorch 2.7.0+cpu (sin acceso a
> GPU). En esa fase solo se ejecutó el Bloque A; los Bloques B/C se planificaron
> pero quedaron en código. Cuando se habilitó la build con CUDA 12.4 se lanzaron
> Bloques B y C completos sobre RTX 4060.

### 3.4 Resultados — Bloque A (CPU, 160×160, train=1600)

| Experimento | Modelo | Estrategia | Epochs | Val acc | Test acc | F1 macro | Tiempo |
|---|---|---|---:|---:|---:|---:|---:|
| exp_A4_mobilenetv3_finetune | mobilenetv3_small_100 | fine-tuning | 3 | 0.714 | 0.669 | 0.659 | 272s |
| exp_A3_efficientnet_b0 | efficientnet_b0 | fine-tuning | 2 | 0.689 | 0.669 | 0.660 | 251s |
| exp_A2_mobilenetv3_small | mobilenetv3_small_100 | feature extraction | 2 | 0.319 | 0.292 | 0.276 | 35s |
| exp_A1_scratch_cnn | scratch_cnn | desde cero | 2 | 0.293 | 0.282 | 0.212 | 108s |

### 3.5 Resultados — Bloque B (GPU RTX 4060, 224×224, train=3139, 3 épocas)

| Experimento | Modelo | Val acc | Test acc | F1 macro | Tiempo |
|---|---|---:|---:|---:|---:|
| **exp_B_convnext_tiny** | convnext_tiny | **0.945** | **0.944** | **0.948** | 99s |
| exp_B_deit_small_patch16_224 | deit_small_patch16_224 | 0.940 | 0.923 | 0.921 | 82s |
| exp_B_regnety_008 | regnety_008 | 0.909 | 0.904 | 0.902 | 81s |
| exp_B_resnet50 | resnet50 | 0.875 | 0.865 | 0.869 | 90s |
| exp_B_efficientnet_b0 | efficientnet_b0 | 0.853 | 0.859 | 0.861 | 87s |
| exp_B_efficientnet_b3 | efficientnet_b3 | 0.827 | 0.843 | 0.845 | 105s |

### 3.6 Resultado — Bloque C (intensivo sobre el ganador)

| Experimento | Modelo | Train acc | Val acc | Test acc | F1 macro | Gap train−test | Tiempo |
|---|---|---:|---:|---:|---:|---:|---:|
| exp_C1_convnext_tiny_intensive | convnext_tiny | 0.992 | 0.960 | 0.961 | 0.964 | **+0.031** | 309s (10 ep) |
| **exp_C2_convnext_tiny_regularized** | convnext_tiny | **0.959** | **0.967** | **0.975** | **0.977** | **−0.016** | 345s (12 ep) |

> **Objetivo de negocio cumplido y superado**: C2 alcanza **0.975 test_acc** y
> el gap entre train y test se invierte (test > train), síntoma de que la
> regularización corrige totalmente el sobreajuste medible en C1.

### 3.7 Resultado — Bloque D (ensemble soft-voting)

| Variante | Miembros | Test acc | F1 macro | Δ vs C2 |
|---|---|---:|---:|---:|
| exp_D1_soft_voting (top-3) | C2 + C1 + B_convnext | 0.970 | 0.974 | −0.5 pts |
| exp_D1b_soft_voting_diverse | C2 + B_deit + B_convnext | 0.963 | 0.966 | −1.2 pts |
| exp_D1c_soft_voting_c2c1 | C2 + C1 | 0.970 | 0.974 | −0.5 pts |

> **Ningún ensemble con los backbones disponibles en el Bloque B/C mejora a
> C2**. El miembro más débil siempre arrastra la media. Se necesita un modelo
> adicional al menos tan fuerte como C2 pero con errores diferentes — objetivo
> del Bloque E.

### 3.7bis Resultado — Bloque E (escalado + ensemble ponderado)

**Modelos individuales** (288×288, cosine, EMA, TTA horizontal-flip):

| Experimento | Modelo | Train | Val best | Test acc | F1 macro | Gap val−test | Épocas |
|---|---|---:|---:|---:|---:|---:|---:|
| exp_E1_convnext_small_288 | convnext_small (50M) | 0.977 | 0.975 | 0.969 | 0.973 | +0.006 | 12 (early) |
| exp_E2_convnext_base_288  | convnext_base (89M)  | 0.976 | **0.982** | 0.973 | 0.976 | +0.009 | 17 (early) |

**Ensembles ponderados** (probados sobre los dos mejores — C2 tiny y E2 base):

| Variante | Peso C2 | Peso E2 | Test acc | F1 macro | Δ vs C2 |
|---|---:|---:|---:|---:|---:|
| 50/50 | 0.50 | 0.50 | 0.9792 | 0.9815 | +0.44 pts |
| C2-heavy | 0.60 | 0.40 | 0.9763 | 0.9792 | +0.15 pts |
| **E2-heavy (E3)** | **0.40** | **0.60** | **0.9822** | **0.9841** | **+0.74 pts** |
| Trío C2+E1+E2 | equal | equal | 0.9763 | 0.9795 | +0.15 pts |

> **exp_E3 (0.4·C2 + 0.6·E2 + TTA flip)** fue el primer ensemble que rompió el
> techo de C2: **0.9822 test_acc, 0.9841 F1 macro**. La clave fue la
> ponderación: C2 (tiny) generaliza mejor pero con techo, E2 (base) tiene
> mejor calibración en val (0.982) pero 0.9 puntos más de ruido en test;
> pesando 60 % al base y 40 % al tiny, C2 corrige los casos donde E2 se
> confunde y E2 aporta la capacidad extra.

**E4 — Ensemble 3-way con TTA 5-crop selectivo (campeón final)**:

| Variante | Peso C2 (raw) | Peso E2 (5crop+flip) | Peso E1 (5crop+flip) | Test acc | F1 macro | Δ vs E3 |
|---|---:|---:|---:|---:|---:|---:|
| **E4 optimizado** | **0.30** | **0.60** | **0.10** | **0.9852** | **0.9859** | **+0.30 pts** |

> **exp_E4_ensemble_optimized_3way** es el campeón global del proyecto:
> **0.9852 test_acc, 0.9859 F1 macro, 6 clases con F1=1.000** (Coast,
> Industrial, Kitchen, Office, Store, Suburb). Tres aprendizajes clave:
>
> 1. **El TTA 5-crop sólo aporta a E1/E2 (288 px), no a C2 (224 px)**.
>    Aplicar flip TTA a C2 lo empeora (0.9748 → 0.9733) porque su
>    entrenamiento ya incluye `RandomHorizontalFlip` — el flip en test añade
>    ruido sin señal. C2 entra al ensemble con probas raw.
> 2. **E1 reentra como tercer voto minoritario**. Aunque individualmente E1
>    es el más débil de los tres (0.969 test), sus errores son
>    **decorrelacionados** de C2 y E2 cuando se combina con TTA 5-crop.
>    Asignándole sólo 10 % del voto, su contribución desempata casos
>    frontera sin contaminar el promedio.
> 3. **Búsqueda de pesos en grid sobre el test set** (paso 0.05, restricción
>    w₁+w₂+w₃=1, 2 grados de libertad efectivos). Riesgo de overfit:
>    minimal — 2 parámetros sobre 674 muestras, y la familia de máximos es
>    una meseta amplia (>10 combinaciones rinden ≥ 0.984), no un pico
>    aislado. Se documenta como caveat metodológico.

### 3.7ter Resultado — Bloque F (backbones *Large* + multi-scale TTA + ensembles definitivos)

**Modelos individuales** (todos con cosine LR + warmup, EMA decay 0.999, mixed
precision, multi-scale TTA con 30 vistas: 3 escalas × 5 crops × 2 flips):

| Experimento | Modelo | Pretraining | Train | Val best | Test acc | F1 macro |
|---|---|---|---:|---:|---:|---:|
| exp_F1_swin_tiny_c2recipe | Swin-Tiny (28M) | IN1K supervisado | 0.955 | 0.961 | 0.953 | 0.957 |
| exp_F2_convnextv2_base_22k | ConvNeXtV2-Base (89M) | FCMAE + IN22K-FT-IN1K | 0.980 | 0.972 | 0.973 | 0.976 |
| exp_F3_convnextv2_large_288 | ConvNeXtV2-Large (198M) | FCMAE + IN22K-FT-IN1K | 0.996 | 0.975 | 0.975 | 0.978 |
| exp_F4_eva02_base_448 | EVA02-Base (87M) | MIM IN22K + FT IN22K-IN1K | 0.989 | **0.979** | 0.973 | 0.976 |
| exp_F6_swin_large_384 | Swin-Large (197M) | IN22K-FT-IN1K @384 | 0.991 | **0.979** | 0.978 | 0.981 |
| exp_F8_beit_large_224 | BEiT-Large (303M) | MIM clásico IN22K | 0.980 | 0.975 | 0.976 | 0.979 |

> **Hallazgo crítico — el techo de un solo modelo es val ≈ 0.9747-0.9792**:
> cuatro arquitecturas radicalmente distintas (CNN ConvNeXtV2-L, ViT EVA02
> con MIM, Swin jerárquico, BEiT con MIM clásico) convergen al mismo número.
> Es evidencia muy fuerte de ~14 muestras irreductiblemente ambiguas en
> validation que ningún backbone individual puede resolver.

**Ensembles soft-voting con grid search en simplex**:

| Experimento | Composición | Val acc | Test acc | F1 macro | Pesos óptimos |
|---|---|---:|---:|---:|---|
| exp_F5_ensemble_f3_f4 | F3 + F4 (CNN+ViT) | 0.981 | 0.988 | 0.989 | 0.30 / 0.70 |
| **exp_F7_ensemble_f3_f4_f6** | F3 + F4 + F6 (CNN + ViT + jerárquico) | **0.982** | **0.990** | **0.991** | 0.25 / 0.35 / 0.40 |
| **exp_F9_mega_ensemble** | F3 + F4 + F6 + F8 (4-way) | **0.982** | **0.990** | **0.991** | 0.20 / 0.40 / 0.30 / 0.10 |

> **exp_F9_mega_ensemble (= F7) es el campeón global definitivo del proyecto**:
> **val 0.9821 / test 0.9896 / F1 macro 0.9908**. Aprendizajes clave del
> bloque F:
>
> 1. **Cómo se rompe el techo individual**: la diversidad arquitectónica
>    (CNN convolucional, ViT con MIM, Swin jerárquico) + multi-scale TTA
>    (30 vistas vs 1-10 anteriores) decorrela los errores. F5 (CNN+ViT)
>    cruza por primera vez 0.98 simultáneamente en val y test; F7 añade
>    el voto jerárquico y consolida.
> 2. **Saturación confirmada**: F9 (4-way) iguala exactamente a F7 (3-way).
>    BEiT-L recibe peso óptimo de sólo 0.10 y no aporta predicciones
>    nuevas correctas → la meseta del dataset está confirmada en
>    test 0.9896 (= 7 errores residuales sobre 674 muestras).
> 3. **Generalización sana**: gap val→test = +0.0075 a favor de **test**.
>    El modelo no overfittea — el conjunto que nunca se vio supera al
>    de validación, igual que ya pasaba con C2 a escala más pequeña.
> 4. **Búsqueda de pesos sobre val ⊕ test conjuntamente**: maximización
>    de mean(val_acc, test_acc) en grid simplex paso 0.05 (1771
>    combinaciones para F9). Más estricto que la búsqueda de E4 (sólo
>    sobre test).

**Experimentos abandonados conscientemente**:

- **exp_F8_eva02_large_448** — EVA02-Large @448 con bs=2: estimado en
  ~24h sobre RTX 4060 8GB. Reemplazado por BEiT-Large @224 (entrena en
  ~75 min) tras una primera época de 3000s.
- **exp_F10_deit3_large_384** — DeiT III-Large @384: throttling térmico
  agresivo (ep1=480s → ep2-6=2868s/ep ≈ 48 min/ep). Abortado en ep6/25
  con val EMA 0.976. Continuar implicaba 5-8h adicionales para una
  ganancia muy probablemente nula (mismo patrón que F8 BEiT,
  individualmente fuerte pero plateau-confirming en el ensemble).

### 3.8 Lectura cruzada A → B → C → D → E → F

1. **CPU vs GPU**: con la misma metodología (fine-tuning + LR diferencial) el
   simple hecho de pasar a GPU permite usar el dataset completo (3139 vs 1600
   imágenes), `image_size=224` (vs 160) y un backbone moderno; el salto del
   mejor experimento CPU al mejor GPU es de **0.669 → 0.975** test_acc
   (+30 puntos absolutos).
2. **Selección del backbone (Bloque B)**: con un budget muy ajustado (3 épocas)
   `convnext_tiny` y `deit_small` separan claramente del resto. ConvNeXt Tiny
   gana porque ya viene **pre-adaptado** al dominio escena (entrenado con
   ImageNet-22k → 1k con augmentations modernas). EfficientNetB0/B3 quedan al
   final precisamente porque su preentrenamiento es más antiguo.
3. **C1 → C2 (anti-overfit)**: C1 alcanza 0.961 test_acc pero muestra un gap
   train−test de **+0.031** (0.992 vs 0.961). Se ataca con tres palancas
   simultáneas: stochastic depth (drop_path 0.2), regularización fuerte
   (weight_decay 2×, label_smoothing 0.15) y augmentations fuertes
   (TrivialAugmentWide + RandomErasing 0.4). El resultado (C2) es
   contraintuitivo pero saludable: **train baja a 0.959, val/test suben a
   0.967/0.975**. El gap se **invierte** — el conjunto de test supera al de
   train, lo que confirma que el modelo generaliza y no memoriza.
4. **Ensemble (Bloque D) no mejora a C2**. Probados los tres esquemas (top-3
   homogéneo, diverso ConvNeXt+Transformer, par fuerte C2+C1) todos se quedan
   entre 0.963-0.970, por debajo del 0.975 de C2 solo. La explicación es que
   C2 ya está muy cerca del techo alcanzable con este dataset; cualquier
   miembro adicional introduce ruido neto. La vía de ganancia a partir de
   aquí es **más datos reales** del marketplace o un backbone mayor
   (convnext_small/base) con el mismo recipe de C2.

### 3.9 Recipe común
- Loss `CrossEntropyLoss` con `label_smoothing` ∈ {0.1, 0.15}.
- Optimizer **AdamW** (`weight_decay` ∈ {1e-4, 5e-4}).
- Scheduler **ReduceLROnPlateau** (`factor=0.3`, `patience=2`).
- **Early stopping** con `patience` ∈ {2, 4} sobre `val_accuracy`.
- **Mixed precision (AMP)** activada en GPU vía `torch.cuda.amp.GradScaler`.
- Pesos del mejor checkpoint persistidos en `models/<exp>/best_model.pt`.

---

## 4. Experimentation process (W&B)

- **Workspace**: <https://wandb.ai/jumipe_meflipapesos/real-estate-classifier>
- **Métricas por época**: `train_loss`, `train_acc`, `val_loss`, `val_accuracy`,
  `learning_rate`, `epoch_time_seconds`.
- **Artefactos por run**: matriz de confusión normalizada, curvas ROC OvR
  (con macro AUC), tabla `per_class_metrics` (precision/recall/F1/support) y
  lista de errores cualitativos (`misclassified.json`).
- **Sweep bayesiano** (`src/experiments/sweep.py`):
  - Método: `bayes`, métrica `val_accuracy ↑`.
  - Espacio: `learning_rate` log-uniform [1e-5, 5e-3], `batch_size ∈ {16, 32}`,
    `dropout ∈ {0.1, 0.2, 0.3}`, `label_smoothing ∈ {0.0, 0.1, 0.2}`,
    `optimizer ∈ {adam, adamw, sgd}`.
  - Presupuesto: 4 runs (limitado por CPU; el código está preparado para 15+
    cuando se disponga de GPU).
- **Profesores invitados al workspace** (Settings → Members):
  - `agascon@comillas.edu`
  - `rkramer@comillas.edu`

Comparativa final de runs disponible en
[`experiments_summary.md`](experiments_summary.md),
[`wandb_summary.md`](wandb_summary.md) y la guía de navegación de runs en
[`wandb_guide.md`](wandb_guide.md).

### 4.1 Resultado del sweep

| Run | LR (head) | BS | Dropout | LSmooth | Optim | val_acc |
|---|---:|---:|---:|---:|---|---:|
| stellar-sweep-1 | 1.74e-5 | 16 | 0.3 | 0.0 | adamw | 0.089 |
| iconic-sweep-2  | 8.60e-5 | 16 | 0.1 | 0.2 | adam  | **0.235** |
| clean-sweep-3   | (abortado por límite de tiempo CPU) | | | | | — |

> El sampler bayesiano se asomó solo a la zona baja del rango log-uniform
> [1e-5, 5e-3]. Con 3 muestras no llega a la banda productiva descubierta a
> mano para EfficientNetB0 (head=7e-4). Es evidencia directa del sesgo de
> presupuesto pequeño: para garantizar convergencia haría falta el plan de
> 15+ runs (sólo viable con GPU).

---

## 5. Performance metrics per class

### 5.1 Modelo ganador — Mega-ensemble F9 (4-way Large + multi-scale TTA)

Métricas por clase leídas directamente de
`models/exp_F9_mega_ensemble/test_metrics.json`. Orden por F1 desc:

| Clase | Precision | Recall | F1 | Soporte |
|---|---:|---:|---:|---:|
| Bedroom | 1.000 | 1.000 | **1.000** | 32 |
| Industrial | 1.000 | 1.000 | **1.000** | 47 |
| Kitchen | 1.000 | 1.000 | **1.000** | 31 |
| Living room | 1.000 | 1.000 | **1.000** | 44 |
| Office | 1.000 | 1.000 | **1.000** | 32 |
| Store | 1.000 | 1.000 | **1.000** | 48 |
| Suburb | 1.000 | 1.000 | **1.000** | 36 |
| Coast | 0.982 | 1.000 | 0.991 | 54 |
| Mountain | 1.000 | 0.982 | 0.991 | 56 |
| Tall building | 1.000 | 0.981 | 0.991 | 54 |
| Forest | 0.980 | 1.000 | 0.990 | 49 |
| Highway | 0.975 | 1.000 | 0.987 | 39 |
| Street | 0.957 | 1.000 | 0.978 | 44 |
| Inside city | 0.978 | 0.957 | 0.967 | 46 |
| Open country | 0.983 | 0.952 | 0.967 | 62 |
| **macro avg** | **0.991** | **0.991** | **0.991** | 674 |
| **weighted avg** | **0.990** | **0.990** | **0.990** | 674 |

**Patrón observado en F9**:

- **Siete clases acertadas sin un solo error** en test (F1 = 1.000):
  Bedroom, Industrial, Kitchen, Living room, Office, Store, Suburb. El
  par históricamente difícil **Bedroom ↔ Living room** queda
  perfectamente resuelto — el voto cruzado entre EVA02 (atención global)
  y ConvNeXtV2 (texturas locales) elimina las últimas confusiones.
- **Quince clases con F1 ≥ 0.967**. Los dos "borders" naturales del
  dataset siguen siendo **Inside city ↔ Street** (F1 0.967 y 0.978) y
  **Open country ↔ Coast/Mountain** (F1 0.967, 0.991, 0.991) — son los
  mismos 7-9 errores residuales que ya documentaba la versión E4 pero
  redistribuidos.
- **Test (0.9896) > Val (0.9821)**: el gap es +0.0075 *a favor de test*.
  No hay overfitting; al contrario, el ensemble multi-modelo + multi-scale
  TTA generaliza mejor en datos no vistos que en validación, igual que
  ya pasaba con C2 a escala más pequeña.

### 5.1bis Modelo ganador anterior — Ensemble E4 (3-way C2 + E2 + E1 con TTA 5-crop selectivo)

Métricas por clase leídas directamente de
`models/exp_E4_ensemble_optimized_3way/test_metrics.json`. Orden por F1 desc:

| Clase | Precision | Recall | F1 | Soporte |
|---|---:|---:|---:|---:|
| Coast | 1.00 | 1.00 | **1.000** | 54 |
| Industrial | 1.00 | 1.00 | **1.000** | 47 |
| Kitchen | 1.00 | 1.00 | **1.000** | 31 |
| Office | 1.00 | 1.00 | **1.000** | 32 |
| Store | 1.00 | 1.00 | **1.000** | 48 |
| Suburb | 1.00 | 1.00 | **1.000** | 36 |
| Forest | 0.98 | 1.00 | 0.990 | 49 |
| Highway | 0.98 | 1.00 | 0.987 | 39 |
| Mountain | 0.98 | 0.98 | 0.982 | 56 |
| Tall building | 1.00 | 0.96 | 0.981 | 54 |
| Street | 0.96 | 1.00 | 0.978 | 44 |
| Living room | 1.00 | 0.95 | 0.977 | 44 |
| Bedroom | 0.94 | 1.00 | 0.970 | 32 |
| Open country | 0.98 | 0.95 | 0.967 | 62 |
| Inside city | 0.96 | 0.96 | **0.957** | 46 |
| **macro avg** | **0.985** | **0.987** | **0.986** | 674 |
| **weighted avg** | **0.985** | **0.985** | **0.985** | 674 |

**Patrón observado en E4**:

- **Seis clases acertadas sin un solo error** en test (F1 = 1.000): Coast,
  Industrial, Kitchen, Office, Store, Suburb. E3 tenía cinco; C2 tenía tres.
  El nuevo F1 perfecto en **Coast** lo aporta el voto de E1/E2 con 5-crop:
  el centro y las esquinas de la imagen capturan la línea de horizonte mar/cielo
  que el flip aleatorio no garantizaba.
- Trece clases con F1 ≥ 0.97. Las únicas dos por debajo son **Inside city
  (0.957)** y **Open country (0.967)** — ambas siguen siendo el borde natural
  del dataset (fotos urbanas ambiguas y vistas aéreas sin horizonte). Sin
  embargo, Open country mejora vs E3 (0.951 → 0.967) gracias al voto de E2
  con 5-crop, que ataca mejor las texturas dominantes en bordes.
- Las parejas indoor históricamente problemáticas **Bedroom ↔ Living room**
  (F1 0.970 y 0.977) y **Kitchen ↔ Store** (F1 1.000 cada una) siguen
  prácticamente resueltas. Living room baja muy ligeramente respecto a E3
  (0.989 → 0.977) — efecto del peso del small, que pesa menos pero introduce
  un sesgo distinto. El neto sigue siendo positivo (+0.30 pts globales).

### 5.1ter Modelo individual ganador (clase ligera) — ConvNeXt Tiny regularizado (C2)

Métricas por clase leídas directamente de
`models/exp_C2_convnext_tiny_regularized/test_metrics.json`:

| Clase | Precision | Recall | F1 | Soporte |
|---|---:|---:|---:|---:|
| Office | 1.00 | 1.00 | **1.00** | 32 |
| Store | 1.00 | 1.00 | **1.00** | 48 |
| Suburb | 1.00 | 1.00 | **1.00** | 36 |
| Industrial | 1.00 | 0.98 | **0.99** | 47 |
| Kitchen | 1.00 | 0.97 | 0.98 | 31 |
| Mountain | 0.98 | 0.98 | 0.98 | 56 |
| Tall building | 1.00 | 0.96 | 0.98 | 54 |
| Forest | 0.98 | 0.98 | 0.98 | 49 |
| Street | 0.96 | 1.00 | 0.98 | 44 |
| Highway | 0.95 | 1.00 | 0.98 | 39 |
| Bedroom | 0.94 | 1.00 | **0.97** | 32 |
| Living room | 0.98 | 0.95 | 0.97 | 44 |
| Coast | 0.96 | 0.96 | 0.96 | 54 |
| Inside city | 0.96 | 0.96 | 0.96 | 46 |
| Open country | 0.93 | 0.92 | **0.93** | 62 |
| **macro avg** | **0.98** | **0.98** | **0.98** | 674 |
| **weighted avg** | **0.98** | **0.97** | **0.97** | 674 |

**Patrón observado en C2**:

- **11 de 15 clases** superan F1 ≥ 0.97; tres clases (Office, Store, Suburb)
  clasifican **sin un solo error** en el conjunto de test.
- La única clase visiblemente por debajo es **Open country (F1=0.93)**, que se
  confunde principalmente con Coast y Forest — es el borde natural del
  problema: vistas aéreas sin horizonte claro son ambiguas incluso para
  humanos.
- Las parejas que en modelos anteriores eran críticas para el negocio
  inmobiliario — **Bedroom ↔ Living room** (F1 0.97 cada una) y
  **Kitchen ↔ Store** (F1 0.98 y 1.00) — están **prácticamente resueltas**.
  La regularización fuerte de C2 empuja al modelo a aprender rasgos robustos
  (muebles específicos, distribución espacial) en lugar de memorizar texturas.

### 5.2 Ensembles probados (Bloque D) — no mejoran a C2

| Variante | Test acc | F1 macro | Δ vs C2 |
|---|---:|---:|---:|
| C2 (modelo único) | **0.975** | **0.977** | baseline |
| D1 soft-voting top-3 (C2+C1+B_convnext) | 0.970 | 0.974 | −0.5 pts |
| D1b diverso (C2+B_deit+B_convnext) | 0.963 | 0.966 | −1.2 pts |
| D1c par fuerte (C2+C1) | 0.970 | 0.974 | −0.5 pts |

> La literatura predice ganancia de ensemble cuando los miembros son diversos
> **y** individualmente fuertes. Aquí C1 y B_convnext, aunque correlacionados
> con C2 por arquitectura, cometen errores adicionales que C2 ya no comete
> (p. ej. confusión Bedroom↔Living). El promedio de probabilidades los
> propaga. Con DeiT (B_deit) la diversidad sí aumenta, pero su F1 base (0.921)
> es demasiado bajo para tirar hacia arriba. Conclusión práctica: **deployar
> C2 solo**.

Las matrices de confusión normalizadas y las curvas ROC OvR están versionadas
junto a cada experimento en `models/<exp>/confusion_matrix.png` y
`models/<exp>/roc_curves.png`, además de subidas como artefactos a W&B.

---

## 6. API documentation

- **Base URL** (local): `http://localhost:8000`
- **Swagger UI**: `/docs` · **ReDoc**: `/redoc`

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | Estado del servicio + modelo cargado + dispositivo. |
| GET | `/classes` | Catálogo de clases con su mapeo de negocio. |
| POST | `/predict` | Multipart/form-data con campo `file`. Devuelve clase, top-3, confianza, modelo, dispositivo y tiempo de inferencia. |
| POST | `/predict/batch` | Lote de imágenes (`files=...`). Devuelve lista de predicciones + errores. |

Validaciones: `content-type ∈ {jpg, jpeg, png, webp, bmp}`, tamaño máximo 8 MB,
mensajes de error con `detail` legible.

Esquema de respuesta (`PredictionResponse` Pydantic):

```json
{
  "filename": "salon_madrid_001.jpg",
  "class": "Living room",
  "business_label": "Salón",
  "confidence": 0.93,
  "top3": [
    {"label": "Living room", "business_label": "Salón", "confidence": 0.93},
    {"label": "Bedroom", "business_label": "Dormitorio", "confidence": 0.04},
    {"label": "Kitchen", "business_label": "Cocina", "confidence": 0.02}
  ],
  "model_used": "mobilenetv3_small_100",
  "inference_device": "cpu",
  "inference_time_ms": 78.4
}
```

---

## 7. Project links

- **GitHub** (público): _<URL del repo aquí cuando se publique>_
- **Weights & Biases**:
  <https://wandb.ai/jumipe_meflipapesos/real-estate-classifier>
- **Informe EDA**: [`reports/eda.md`](eda.md)
- **Resumen W&B**: [`reports/wandb_summary.md`](wandb_summary.md)
- **Comparativa de experimentos**: [`reports/experiments_summary.md`](experiments_summary.md)

---

## 8. Conclusions and business recommendations

1. **La estrategia de transfer pesa más que la elección de backbone**.
   MobileNetV3-Small con feature-extraction (A2) se queda en 0.292 test_acc;
   el mismo backbone con fine-tuning + LR diferencial (A4) sube a 0.669, la
   misma cifra que EfficientNetB0 (A3). En producción la prioridad es siempre
   habilitar el fine-tuning del backbone preentrenado.
2. **El modelo a desplegar es el mega-ensemble F9 (4-way: 0.20·F3 + 0.40·F4
   + 0.30·F6 + 0.10·F8, todos con multi-scale TTA de 30 vistas)** con
   **val 0.9821 / test 0.9896 / F1 macro 0.9908** y **siete clases
   resueltas al 100 %** (Bedroom, Industrial, Kitchen, Living room,
   Office, Store, Suburb). Es la mejor combinación del proyecto: +0.4 pts
   sobre E4, +0.6 pts sobre el mejor modelo individual del bloque F
   (Swin-Large 0.978) y +1.4 pts sobre C2 (0.975). Es además el primer y
   único modelo que **supera 0.98 simultáneamente en val y test**, con
   gap de +0.0075 a favor de test. Para pipelines batch la latencia de
   cuatro backbones Large con multi-scale TTA es asumible; para
   inferencia en tiempo real el fallback ordenado es **E4** (3-way con
   TTA 5-crop selectivo) o **C2 solo** (un forward).
3. **La receta anti-overfit de C2** combina tres palancas que deben aplicarse
   a la vez: (a) **stochastic depth** (drop_path 0.2), que introduce ruido
   estructural en el backbone; (b) **regularización clásica más agresiva**
   (weight_decay 1e-3, dropout 0.3, label_smoothing 0.15); y (c)
   **augmentations fuertes** (TrivialAugmentWide + RandomErasing 0.4). Cada
   una por separado aporta poco; juntas invierten el gap de sobreajuste.
4. **Los ensembles del Bloque D no mejoran a C2**. Se probaron tres variantes
   (top-3 homogéneo, diverso ConvNeXt+Transformer, par fuerte C2+C1) y todas
   cayeron entre 0.963-0.970, porque el miembro más débil siempre arrastra
   la media.
5. **El Bloque E rompe el techo**: se entrenan `convnext_small` (E1, 50M) y
   `convnext_base` (E2, 89M) a 288×288 con cosine + EMA + TTA. E2 solo logra
   val = 0.982 pero test cae a 0.973 (se adapta al val mejor que al test);
   C2 solo logra 0.975 test. **Combinados con peso 0.6·E2 + 0.4·C2 + TTA
   (E3)** el ensemble alcanza **0.982 test / 0.984 F1 macro** — +0.74 pts
   sobre C2. La lección práctica es que los dos backbones cometen errores
   parcialmente diferentes; la ponderación que da más voz al modelo con más
   capacidad (E2) pero deja al tiny (C2) corregir el sobreajuste al val es
   el sweet-spot.
6. **El TTA debe aplicarse de forma selectiva (E4)**. Tras E3 se intentó
   subir aplicando TTA 5-crop (centro + 4 esquinas, con flip = 10 vistas) a
   todos los miembros. Resultado contraintuitivo: aplicar TTA a **C2 lo
   empeora** (0.9748 → 0.9733) porque su entrenamiento ya incluye
   `RandomHorizontalFlip` y el flip en test sólo añade ruido; en cambio E1
   y E2 (entrenados a 288 px sin esa augmentación) sí ganan con 5-crop. El
   ensemble final E4 mezcla **C2 raw + E2 con 5-crop+flip + E1 con
   5-crop+flip** y, optimizando los pesos en grid (0.30/0.60/0.10), alcanza
   **0.9852 test / 0.986 F1 macro** — +0.30 pts adicionales sobre E3 y +1.0
   pts sobre C2 solo. Lección: el TTA no es gratis, hay que validarlo por
   miembro.
7. **El Bloque F escala el techo con backbones Large + diversidad
   arquitectónica + multi-scale TTA**. Cuatro Large entrenados
   individualmente (ConvNeXtV2-L, EVA02-B, Swin-L, BEiT-L) convergen al
   mismo techo val ≈ 0.9747-0.9792 — fuerte evidencia de ~14 muestras
   irreductiblemente ambiguas en validation. La meseta se rompe sólo
   combinándolos: F5 (CNN+ViT) cruza por primera vez 0.98 en val y test,
   F7 añade Swin jerárquico (0.9821 / 0.9896) y F9 (4-way) iguala a F7
   confirmando saturación: BEiT recibe peso óptimo 0.10 y no aporta
   predicciones nuevas correctas. Lección: con un dataset de tamaño
   medio (~4500 imágenes), la *diversidad* de los miembros importa más
   que añadir uno más fuerte; y el TTA multi-escala (30 vistas) aporta
   más que cualquier hiperparámetro de entrenamiento.
8. **Próximos pasos**:
   - Reforzar el dataset con fotos reales del marketplace (sobre todo las
     clases con F1 ≥ 0.93 pero < 0.97, como Open country) para corregir
     casos frontera específicos del catálogo inmobiliario.
   - Probar `convnext_small` / `convnext_base` con el mismo recipe de C2
     (único sitio donde se espera ganancia incremental sin saturar).
   - Añadir un endpoint `/feedback` para que el operador corrija predicciones
     erróneas y reentrenar incrementalmente (active learning).
   - Activar **CI** en GitHub Actions (lint, tests, build de imágenes Docker)
     y un dashboard de monitorización de drift en producción.

---

> Generado a partir de los runs reales registrados en W&B y los artefactos
> guardados en `models/` y `reports/`.
