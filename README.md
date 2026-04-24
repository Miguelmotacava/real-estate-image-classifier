# Real Estate Image Classifier

Clasificador automático de fotos de anuncios inmobiliarios. Se sube una
imagen por la API y se devuelve la categoría (salón, cocina, dormitorio,
fachada, calle, etc.), un top-3 con sus confianzas y metadatos del modelo
usado. Construido sobre el dataset **15-Scene** (15 clases, 4485 imágenes)
y mapeado a un vocabulario de negocio de tres familias: Interior, Exterior
urbano y Entorno natural.

El proyecto incluye la API, una UI de Streamlit que la consume, todo el
tracking del entrenamiento en W&B y los cinco modelos de producción
versionados como artifacts.

**Autores:** Pedro Calderón · Juan Miguel Correa · Miguel Mota Cava
**Asignatura:** Machine Learning II — Máster en Big Data, curso 2025/26

---

## Resultados principales

| Modelo servido | Tipo | val_acc (90/10) | F1 macro | Latencia en GPU |
|---|---|---|---|---|
| `FINAL` (default) | Swin-Large 384 single | 0.9866 | 0.988 | ~90 ms |
| `ensemble` (máxima accuracy) | 4 backbones + multi-scale TTA | **0.9933** | **0.994** | ~4 s |

El campeón es un ensemble soft-voting de ConvNeXtV2-L, EVA02-B 448, Swin-L
384 y BEiT-L 224 con 30 vistas de TTA por miembro. El README técnico con
todos los detalles está en [`reports/final_report.md`](reports/final_report.md).

W&B: https://wandb.ai/jumipe_meflipapesos/real-estate-classifier

---

## Stack

- **Modelado**: PyTorch 2.6 + `timm` (transfer learning desde ImageNet)
- **Tracking**: Weights & Biases (métricas por epoch, sweep bayesiano, artifacts)
- **Servicio**: FastAPI + Swagger + Streamlit
- **Despliegue**: Docker Compose (API en 8000, UI en 8501)

---

## Estructura del repo

```
.
├── api/                  FastAPI (main.py, inference.py, schemas.py, Dockerfile)
├── streamlit_app/        UI (app.py, Dockerfile)
├── src/
│   ├── experiments/      trainers, sweep, ensembles, upload de artifacts
│   └── utils/            data, models, metrics, train_loop, device
├── models/               pesos por experimento (.pt fuera de git, en W&B)
├── reports/              EDA, comparativas, informe técnico, figuras
├── dataset/              15-Scene (4485 imágenes, 15 clases)
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

Los `.pt` pesan ~3 GB y viven en W&B como artifacts versionados. El repo
sólo tiene los `summary.json`, las matrices de confusión, las curvas ROC y
el resto de documentación.

---

## Arrancar el servicio

### Requisitos previos

- Docker Desktop (para la vía recomendada)
- Python 3.11+ (sólo si quieres ejecutar en local sin Docker)
- Git
- 4 GB libres en disco

No hace falta GPU para inferencia: la API detecta CPU automáticamente. En
CPU, los modelos single responden en 1-2 s por imagen y el ensemble puede
tardar varios minutos (la UI tiene un timeout largo para ese caso).

### Paso 1 — clonar

```bash
git clone https://github.com/Miguelmotacava/real-estate-image-classifier.git
cd real-estate-image-classifier
```

### Paso 2 — crear el `.env`

Copia la plantilla y rellena con tu propia API key de W&B (no subas la
tuya al repo; el fichero está en `.gitignore`):

```bash
cp .env.example .env
# edita .env y pon:
#   WANDB_API_KEY=<tu_key>
#   WANDB_ENTITY=jumipe_meflipapesos
#   WANDB_PROJECT=real-estate-classifier
```

### Paso 3 — descargar los modelos desde W&B

Los checkpoints `.pt` no están en el repo por tamaño. Con las credenciales
del paso anterior se bajan como artifacts (~3 GB, 10-30 min según conexión):

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate          # Windows
pip install wandb python-dotenv
wandb login                      # pega la key cuando la pida

python -c "
import wandb
api = wandb.Api()
base = 'jumipe_meflipapesos/real-estate-classifier'
for exp in ['exp_FINAL_swin_large_384_9010','exp_FINAL_F3_9010','exp_FINAL_F4_9010','exp_FINAL_F8_9010','exp_FINAL_ensemble_9010']:
    api.artifact(f'{base}/model-{exp}:latest').download(root=f'models/{exp}')
    print(f'OK: {exp}')
"
```

### Paso 4 — levantar API + UI

**Con Docker** (recomendado):

```bash
docker compose up --build
```

La primera vez tarda 5-10 min construyendo las imágenes. Cuando veas:

```
api-1        | [registry] loaded ensemble with 4 members, val=0.9933
api-1        | INFO:     Uvicorn running on http://0.0.0.0:8000
streamlit-1  |   URL: http://0.0.0.0:8501
```

ya está todo arriba. Para parar: `docker compose down`.

**En local sin Docker** (útil para desarrollo):

```bash
# terminal 1 — API
uvicorn api.main:app --host 0.0.0.0 --port 8000

# terminal 2 — UI
streamlit run streamlit_app/app.py
```

### Puntos de entrada

| URL | Para qué |
|---|---|
| http://localhost:8501 | Streamlit con dropdown de modelos |
| http://localhost:8000/docs | Swagger UI (API interactiva) |
| http://localhost:8000/redoc | Documentación ReDoc |
| http://localhost:8000/models | Lista JSON de modelos disponibles |
| http://localhost:8000/health | Estado del servicio |

---

## Endpoints de la API

Esquema OpenAPI 3.1 expuesto automáticamente por FastAPI. Los tipos están
descritos en `api/schemas.py`.

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/health` | estado y modelo cargado por defecto |
| `GET` | `/classes` | las 15 clases con etiqueta de negocio |
| `GET` | `/models` | los 5 alias servibles (`FINAL`, `F3`, `F4`, `F8`, `ensemble`) |
| `POST` | `/predict?model=<alias>` | clasifica una imagen |
| `POST` | `/predict/batch?model=<alias>` | clasifica varias en una sola llamada |

Ejemplo vía `curl`:

```bash
curl -X POST "http://localhost:8000/predict?model=ensemble" \
  -F "file=@foto_salon.jpg;type=image/jpeg"
```

Respuesta:

```json
{
  "filename": "foto_salon.jpg",
  "class": "Living room",
  "business_label": "Salón",
  "confidence": 0.987,
  "top3": [...],
  "model_used": "ensemble[F3,F4,FINAL,F8]",
  "model_alias": "ensemble",
  "inference_device": "cpu",
  "inference_time_ms": 4217.6
}
```

Errores: `400` imagen corrupta, `404` modelo no existe, `413` imagen >8 MB,
`415` MIME no soportado, `503` registro no cargado. El endpoint batch
nunca falla globalmente: cada imagen inválida aparece en `errors[]` y las
otras siguen.

---

## Dataset y splits

15-Scene: 4485 imágenes en 15 clases balanceadas. El EDA completo está en
[`reports/eda.md`](reports/eda.md). Se usaron dos particiones distintas:

- **70/15/15 estratificado** (seed=42) durante toda la experimentación
  (comparativas, benchmarks con test no visto).
- **90/10 estratificado** (seed=42) para el modelo final desplegado. Una
  vez elegida la arquitectura, el holdout de test se reincorpora como
  train para dar al modelo todos los datos disponibles.

Mapeo clase → negocio (en `api/inference.py`):

- **Interior**: Bedroom, Kitchen, Living room, Office, Store, Industrial
- **Exterior urbano**: Inside city, Tall building, Street, Suburb, Highway
- **Entorno natural**: Coast, Forest, Mountain, Open country

---

## Reentrenar desde cero

No suele ser necesario (los pesos están en W&B), pero se puede. Estimación
en RTX 4060 Laptop 8 GB VRAM: ~6 h de entrenamiento seguidas.

```bash
pip install -r requirements.txt

# Modelo individual Swin-L 384 (90/10)
python -m src.experiments.final_9010

# Miembros del ensemble
python -m src.experiments.final_member_9010 --member F3
python -m src.experiments.final_member_9010 --member F4
python -m src.experiments.final_member_9010 --member F8

# Grid search de pesos del ensemble
python -m src.experiments.final_ensemble_9010
```

Todos los scripts loguean a W&B automáticamente y guardan el checkpoint
como artifact al final.

---

## Experimentación

Se organizó en 6 bloques (A → F + FINAL), cada uno cerrando una pregunta
antes de pasar al siguiente. Resumen en
[`reports/experiments_summary.md`](reports/experiments_summary.md), con
20+ experimentos completos trazados en W&B.

Los 3 drivers que más accuracy aportaron, en orden de impacto:

1. **Diversidad arquitectónica en el ensemble** (+~1.5 pts sobre el mejor
   individual). Cuatro backbones con paradigmas distintos (CNN, ViT-MIM,
   Swin jerárquico, BEiT-MIM) cometen errores decorrelacionados.
2. **Multi-scale TTA (30 vistas por imagen)** (+~0.5 pts). 3 escalas ×
   5 crops × 2 flips.
3. **Receta anti-overfit** (label_smoothing, drop_path, EMA, weight decay
   moderado). Aportó los primeros ~2 pts cuando pasamos de ConvNeXt-Tiny
   bruto a ConvNeXt-Tiny regularizado.

MixUp y CutMix se probaron y se descartaron: con esta receta ya controlan
el overfit los otros mecanismos, y la mezcla de imágenes degradaba las
fronteras Kitchen/Bedroom.

---

## Troubleshooting habitual

| Síntoma | Causa probable | Solución |
|---|---|---|
| API arranca pero `/health` devuelve `degraded` | falta descargar los modelos de W&B | paso 3 de este README |
| `docker compose up` falla con "Docker Desktop is unable to start" | Docker Desktop no está arrancado | ábrelo y espera al icono verde |
| Timeout en Streamlit al usar `ensemble` | estás en CPU y tarda varios minutos | usa un modelo single (FINAL, F3, F4, F8); el timeout del ensemble ya está en 10 min |
| `CUDA out of memory` | GPU con <8 GB pillando bs demasiado grande | edita `batch_size` en el script del miembro correspondiente |
| Falla al descargar un artifact | W&B login no completado | `wandb login` y reintenta |
| La GPU de portátil se calienta y entrena lentísimo | thermal throttling | no hay mucho que hacer más allá de bajar la temperatura ambiente; nos pasó durante el desarrollo |

---

## Enlaces y acceso

- **Repo (público)**: https://github.com/Miguelmotacava/real-estate-image-classifier
- **W&B**: https://wandb.ai/jumipe_meflipapesos/real-estate-classifier
- **Informe técnico ampliado**: [`reports/final_report.md`](reports/final_report.md)
- **Comparativa de experimentos**: [`reports/experiments_summary.md`](reports/experiments_summary.md)
- **Guía W&B**: [`reports/wandb_guide.md`](reports/wandb_guide.md)

`agascon@comillas.edu` y `rkramer@comillas.edu` invitados al workspace de
W&B con permisos de viewer.
