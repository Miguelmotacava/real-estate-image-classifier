"""Streamlit frontend for the real-estate image classifier.

Adds a model selector (single backbones + 4-way ensemble) so the user can
compare predictions across the FINAL Swin-L, the individual F3/F4/F8 and
the production ensemble.
"""
from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")
TIMEOUT = 60

st.set_page_config(
    page_title="Clasificador de imágenes inmobiliarias",
    page_icon="\U0001F3E0",
    layout="wide",
)


@st.cache_data(ttl=60)
def fetch_health() -> dict[str, Any] | None:
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        return resp.json()
    except Exception:
        return None


@st.cache_data(ttl=300)
def fetch_classes() -> dict[str, Any] | None:
    try:
        resp = requests.get(f"{API_URL}/classes", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        return None
    return None


@st.cache_data(ttl=60)
def fetch_models() -> dict[str, Any] | None:
    try:
        resp = requests.get(f"{API_URL}/models", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        return None
    return None


def call_predict(file_bytes: bytes, filename: str, content_type: str, model: str) -> dict | None:
    files = {"file": (filename, file_bytes, content_type)}
    params = {"model": model} if model else {}
    try:
        resp = requests.post(
            f"{API_URL}/predict", files=files, params=params, timeout=TIMEOUT,
        )
    except requests.RequestException as exc:
        st.error(f"Error contactando con la API: {exc}")
        return None
    if resp.status_code != 200:
        st.error(f"La API respondió {resp.status_code}: {resp.text}")
        return None
    return resp.json()


# --- Sidebar -----------------------------------------------------------------
with st.sidebar:
    st.title("\U0001F3E0 Real Estate Classifier")
    st.caption("Clasificación automática de fotos de anuncios inmobiliarios.")
    st.markdown(f"**API:** `{API_URL}`")

    health = fetch_health()
    if health and health.get("status") == "ok":
        st.success(f"API online · default: `{health['model']}` · `{health['device']}`")
    else:
        st.error("API no disponible o sin modelo cargado.")

    st.divider()
    st.subheader("Selecciona el modelo")
    models_payload = fetch_models()
    selected_model: str | None = None
    if models_payload and models_payload.get("models"):
        options = models_payload["models"]
        names = [m["name"] for m in options]
        labels = {m["name"]: m["label"] for m in options}
        default_name = models_payload.get("default") or names[0]
        default_idx = names.index(default_name) if default_name in names else 0
        selected_model = st.selectbox(
            "Modelo de inferencia",
            options=names,
            index=default_idx,
            format_func=lambda n: labels.get(n, n),
            help=(
                "FINAL = recomendado (rápido, val 0.987).\n"
                "Single F3/F4/F8 = backbones individuales.\n"
                "Ensemble = máxima accuracy, ~4× latencia."
            ),
        )
        meta = next((m for m in options if m["name"] == selected_model), None)
        if meta:
            with st.expander("Detalles del modelo seleccionado", expanded=False):
                st.markdown(f"**Tipo:** `{meta['type']}`")
                st.markdown(f"**Backbone:** `{meta['backbone']}`")
                st.markdown(f"**Imagen esperada:** {meta['image_size']}×{meta['image_size']}")
                if meta.get("val_accuracy") is not None:
                    st.markdown(f"**Val accuracy:** `{meta['val_accuracy']:.4f}`")
                st.caption(meta["description"])
    else:
        st.warning("No se pudo obtener la lista de modelos de la API.")

    st.divider()
    st.markdown(
        "**Cómo usarlo**\n\n"
        "1. Elige un modelo arriba.\n"
        "2. Sube una foto del inmueble.\n"
        "3. La API devuelve la clase principal y un top-3.\n"
        "4. Compara modelos cambiando el desplegable y volviendo a clasificar."
    )


# --- Main --------------------------------------------------------------------
st.title("Clasificador de imágenes inmobiliarias")
st.markdown(
    "Sube una imagen del anuncio y obtendrás la categoría predicha "
    "(salón, cocina, dormitorio, fachada, etc.) junto con un top-3."
)

uploaded = st.file_uploader(
    "Arrastra o selecciona una imagen",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=False,
)

if uploaded is None:
    st.info("Sube una imagen para empezar. Formatos soportados: JPG, PNG, WEBP (max 8 MB).")
else:
    cols = st.columns([1, 1])
    with cols[0]:
        st.image(uploaded, caption=uploaded.name, use_container_width=True)
    with cols[1]:
        with st.spinner(f"Clasificando con `{selected_model or 'default'}`…"):
            result = call_predict(
                uploaded.getvalue(),
                uploaded.name,
                uploaded.type or "image/jpeg",
                selected_model or "",
            )
        if result is not None:
            top1 = result["class"]
            business = result["business_label"]
            confidence = result["confidence"]
            st.subheader(f"\U0001F4C2 Categoría: **{business}**")
            st.caption(f"(etiqueta interna del modelo: `{top1}`)")
            st.progress(min(max(confidence, 0.0), 1.0), text=f"Confianza {confidence*100:.1f}%")

            df = pd.DataFrame(
                [
                    {
                        "Categoría (negocio)": item["business_label"],
                        "Etiqueta interna": item["label"],
                        "Confianza": item["confidence"],
                    }
                    for item in result["top3"]
                ]
            )
            st.markdown("**Top-3 predicciones**")
            st.dataframe(
                df.style.format({"Confianza": "{:.1%}"}).bar(
                    subset=["Confianza"], color="#1f77b4"
                ),
                hide_index=True,
                use_container_width=True,
            )

            with st.expander("Detalles técnicos"):
                st.json(
                    {
                        "modelo_alias": result.get("model_alias", "n/a"),
                        "modelo_backbone": result["model_used"],
                        "dispositivo": result["inference_device"],
                        "tiempo_inferencia_ms": result["inference_time_ms"],
                    }
                )

st.divider()

with st.expander("Catálogo de clases soportadas"):
    classes_payload = fetch_classes()
    if classes_payload:
        df = pd.DataFrame(classes_payload["classes"])
        df.columns = ["Etiqueta interna", "Familia", "Etiqueta de negocio"]
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.warning("No se pudieron obtener las clases de la API.")

st.caption(
    "Construido con FastAPI + PyTorch + Streamlit. "
    "Modelo entrenado sobre el dataset 15-Scene adaptado a un marketplace inmobiliario."
)
