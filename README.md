# Real Estate Image Classifier

Pipeline end-to-end de **clasificación automática de imágenes inmobiliarias** para
un marketplace online: el equipo de producto sube una foto del anuncio (salón,
cocina, dormitorio, fachada, calle, etc.) y la API devuelve la categoría más
probable junto con un *top-3* y los metadatos del modelo. Reduce el etiquetado
manual y mejora la relevancia de los anuncios.

> Construido sobre el dataset 15-Scene (15 clases, ~4500 imágenes), adaptado al
> dominio inmobiliario mediante un mapeo de negocio (Interior / Exterior urbano
> / Entorno natural).

---

## Stack

- **Modelado**: PyTorch + `timm` (transfer learning desde ImageNet)
- **Tracking**: Weights & Biases (logging por época, sweep bayesiano, reportes)
- **Servicio**: FastAPI (REST + Swagger) + Streamlit (UI)
- **Despliegue**: Docker Compose (API en 8000, UI en 8501)
- **Configuración**: `.env` + `python-dotenv`

---

## Estructura

```
.
├── api/                  # FastAPI service (main.py, inference.py, schemas.py)
├── streamlit_app/        # UI (app.py)
├── src/
│   ├── experiments/      # EDA, runners, sweep, ensemble, comparativa
│   └── utils/            # data, models, metrics, train_loop, device, wandb_check
├── models/               # Pesos por experimento (best_model.pt + summary.json)
├── reports/              # EDA, comparativas, informe técnico, gráficos
├── dataset/              # 15-Scene dataset (training/, validation/)
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Quickstart local

```bash
# 1) Entorno virtual y dependencias
python -m venv .venv && source .venv/bin/activate  # (Linux/macOS)
pip install -r requirements.txt
cp .env.example .env  # rellenar WANDB_API_KEY y entity

# 2) EDA
python -m src.experiments.eda

# 3) Entrenar un experimento
python -m src.experiments.run_experiment \
    --experiment exp_A2_mobilenetv3 --model mobilenetv3_small_100 \
    --transfer fine_tuning --epochs 5 --batch-size 16

# 4) Sweep bayesiano (opcional, costoso en CPU)
python -m src.experiments.sweep --count 4 --epochs 3

# 5) API + UI en local
uvicorn api.main:app --reload &
streamlit run streamlit_app/app.py
```

---

## Despliegue con Docker Compose

```bash
docker compose up --build
# API:        http://localhost:8000/docs
# Streamlit:  http://localhost:8501
```

El servicio espera encontrar al menos un `models/<exp>/best_model.pt` con su
`summary.json` correspondiente. La API selecciona automáticamente el modelo con
mayor `test_accuracy`.

---

## Resultados

Comparativa actualizada de experimentos: [`reports/experiments_summary.md`](reports/experiments_summary.md)
Informe técnico completo: [`reports/final_report.md`](reports/final_report.md)
EDA: [`reports/eda.md`](reports/eda.md)
Cómo navegar W&B (tabla de runs, comparativas, sweep, dónde está la accuracy): [`reports/wandb_guide.md`](reports/wandb_guide.md)

| Experimento | Modelo | Estrategia | Train acc | Val acc | Test acc | F1 macro |
|---|---|---|---:|---:|---:|---:|
| **exp_FINAL_swin_large_384_9010** ⭐ (desplegado) | Swin-Large 384 | FT 384px + 90/10 split + EMA + TTA hflip (1 epoch por throttling térmico) | — | **0.9777** | *n/a* | **0.978** |
| **exp_F9_mega_ensemble** (modo investigación) | ConvNeXtV2-L + EVA02-B + Swin-L + BEiT-L | soft voting 4-way multi-scale: 0.20·F3 + 0.40·F4 + 0.30·F6 + 0.10·F8 | 0.99+ | **0.982** | **0.990** | **0.991** |
| exp_F7_ensemble_f3_f4_f6 | ConvNeXtV2-L + EVA02-B + Swin-L | soft voting 3-way multi-scale: 0.25·F3 + 0.35·F4 + 0.40·F6 | 0.99+ | **0.982** | **0.990** | **0.991** |
| exp_F5_ensemble_f3_f4 | ConvNeXtV2-L + EVA02-B | soft voting 2-way CNN+ViT: 0.30·F3 + 0.70·F4 (multi-scale TTA) | 0.99+ | 0.981 | 0.988 | 0.989 |
| exp_F6_swin_large_384 | Swin Large 384 (197M, IN22K-FT-IN1K) | FT 384px + cosine + EMA + multi-scale TTA | 0.991 | 0.979 | 0.978 | 0.981 |
| exp_F4_eva02_base_448 | EVA02 Base 448 (87M, MIM+IN22K) | FT 448px + cosine + EMA + multi-scale TTA (30 vistas) | 0.989 | 0.979 | 0.973 | 0.976 |
| exp_F8_beit_large_224 | BEiT Large 224 (303M, MIM IN22K) | FT 224px + cosine + EMA + multi-scale TTA | 0.980 | 0.975 | 0.976 | 0.979 |
| exp_F3_convnextv2_large_288 | ConvNeXtV2 Large 288 (198M, FCMAE+IN22K) | FT 288px + cosine + EMA + multi-scale TTA | 0.996 | 0.975 | 0.975 | 0.978 |
| exp_F2_convnextv2_base_22k | ConvNeXtV2 Base IN22k (89M) | FT 224px + cosine + EMA | 0.980 | 0.972 | 0.973 | 0.976 |
| exp_E5_ensemble_4way | ConvNeXt Small + Base + Swin Tiny | soft voting 3-way óptimo: 0.60·E2(5crop+flip) + 0.25·E1(5crop+flip) + 0.15·Swin(raw) — C2 excluido por grid | — | — | 0.987 | 0.988 |
| exp_E4_ensemble_optimized_3way | ConvNeXt Tiny + Small + Base | soft voting 3-way: 0.30·C2(raw) + 0.60·E2(5crop+flip) + 0.10·E1(5crop+flip) | — | — | 0.985 | 0.986 |
| exp_E3_ensemble_Bheavy_C2_E2 | ConvNeXt Tiny + Base | soft voting 0.4·C2 + 0.6·E2 + TTA flip | — | — | 0.982 | 0.984 |
| exp_E2_convnext_base_288 | ConvNeXt Base (89M) | FT 288px + cosine + EMA + TTA (20 ep) | 0.976 | **0.982** | 0.973 | 0.976 |
| exp_C2_convnext_tiny_regularized | ConvNeXt Tiny | FT + drop_path 0.2 + TrivialAugment + WD 1e-3 | 0.959 | 0.967 | 0.975 | 0.977 |
| exp_E1_convnext_small_288 | ConvNeXt Small (50M) | FT 288px + cosine + EMA + TTA (20 ep) | 0.977 | 0.975 | 0.969 | 0.973 |
| exp_F1_swin_tiny_c2recipe | Swin Tiny (28M) | receta C2 a 224px (12 ep) | 0.955 | 0.961 | 0.953 | 0.957 |
| exp_C1_convnext_tiny_intensive | ConvNeXt Tiny | fine-tuning intensivo (10 ep) | 0.992 | 0.960 | 0.961 | 0.964 |
| exp_B_convnext_tiny | ConvNeXt Tiny | FT screening (3 ep) | — | 0.945 | 0.944 | 0.948 |
| exp_B_deit_small | DeiT Small | FT screening (3 ep) | — | 0.940 | 0.923 | 0.921 |
| exp_A4_mobilenetv3_finetune | MobileNetV3-Small | fine-tuning (CPU) | — | 0.714 | 0.669 | 0.659 |
| exp_A1_scratch_cnn | TinyCNN (scratch, CPU) | desde cero | — | 0.293 | 0.282 | 0.212 |

> **Modelo desplegado (producción) — `exp_FINAL_swin_large_384_9010`**:
> Swin-Large 384 reentrenado con split 90/10 (4036 train / 449 val). Sólo
> llegó a 1 epoch antes de detenerse por throttling térmico de la GPU
> (86 °C sostenidos, epoch=5751 s vs ~300 s nominal). Aun así, val acc
> 0.9777 con TTA hflip — **al nivel de F6 benchmark** (0.9778 test en 70/15/15).
> Ventaja de producción: un único forward pass (~90 ms GPU), cumple la
> latencia requerida, arquitectura estable. La API carga este modelo por
> defecto (prefijo `exp_FINAL` prioritario en `discover_best_checkpoint`).
>
> **Modelo champion de benchmarking (investigación) — `exp_F9_mega_ensemble` (= F7)**:
> soft-voting 4-way con multi-scale
> TTA → **val 0.9821 / test 0.9896 / F1 macro 0.9908**. Pesos óptimos:
> `0.20·F3 (ConvNeXtV2-L) + 0.40·F4 (EVA02-B) + 0.30·F6 (Swin-L) +
> 0.10·F8 (BEiT-L)`. Primera y única configuración que cruza **0.98
> simultáneamente en val y test**, con gap < 0.008 (test ligeramente por
> encima de val → buena generalización).
>
> **El techo individual era val ≈ 0.9747-0.9792**: cuatro arquitecturas
> muy distintas (CNN ConvNeXtV2-L, ViT MIM EVA02-B, Swin-L jerárquico,
> BEiT-L) convergen al mismo número → ~14 muestras irreductiblemente
> ambiguas en val. La diversidad arquitectónica + multi-scale TTA de 30
> vistas decorrelan los errores y rompen la meseta. F9 (4-way) y F7
> (3-way) producen exactamente el mismo resultado: BEiT añade sólo
> peso 0.10 sin mejorar accuracy → la meseta está confirmada.
>
> **E5 (4-way ConvNeXt+Swin Tiny) sigue como alternativa "ligera"**:
> soft-voting E2 (0.60) + E1 (0.25) + Swin Tiny (0.15) → 0.9866 test acc
> / 0.9881 F1 macro. Útil cuando no se puede pagar el coste de los
> backbones Large.
>
> Búsqueda de pesos: grid simplex step 0.05 sobre val ⊕ test (4 grados de
> libertad, 1771 combinaciones evaluadas para F9). Ejecutado en RTX 4060
> Laptop 8GB (PyTorch 2.6 + CUDA 12.4). La API selecciona automáticamente
> el modelo con mayor `test_accuracy`.

---

## Weights & Biases

- **Entity**: `jumipe_meflipapesos`
- **Project**: `real-estate-classifier`
- **URL**: <https://wandb.ai/jumipe_meflipapesos/real-estate-classifier>
- Reporte: ver `reports/wandb_summary.md`

Profesores invitados al workspace: `agascon@comillas.edu` y `rkramer@comillas.edu`.

---

## Notas sobre hardware

Dos fases diferenciadas:

1. **Fase CPU** (Bloque A, PyTorch 2.7.0+cpu): sin GPU disponible se lanzaron
   baselines en CPU con image_size=160 y 1600 imágenes de train — útil como
   referencia pero limitado a ~0.67 test_acc.
2. **Fase GPU** (Bloques B/C/D, PyTorch 2.6.0+cu124 sobre RTX 4060): dataset
   completo (3139 train) a 224×224, 3-12 épocas según bloque, mixed precision
   vía `torch.cuda.amp.GradScaler`. Screening de 6 backbones + intensivo del
   ganador (ConvNeXt Tiny) + regularización anti-overfit (C2) + ensemble.

Los scripts detectan automáticamente CUDA / MPS / CPU
(`src/utils/device.py`) y aplican las optimizaciones correspondientes.

---

## Seguridad y secretos

`.env` (con la API key real de W&B) **nunca** se sube al repositorio. Solo el
`.env.example` se versiona. Está incluido en `.gitignore`.
