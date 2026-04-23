"""Streamlit frontend for the real-estate image classifier."""
from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")
TIMEOUT = 30

st.set_page_config(
    page_title="Clasificador de im\u00e1genes inmobiliarias",
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


def call_predict(file_bytes: bytes, filename: str, content_type: str) -> dict | None:
    files = {"file": (filename, file_bytes, content_type)}
    try:
        resp = requests.post(f"{API_URL}/predict", files=files, timeout=TIMEOUT)
    except requests.RequestException as exc:
        st.error(f"Error contactando con la API: {exc}")
        return None
    if resp.status_code != 200:
        st.error(f"La API respondi\u00f3 {resp.status_code}: {resp.text}")
        return None
    return resp.json()


# --- Sidebar -----------------------------------------------------------------
with st.sidebar:
    st.title("\U0001F3E0 Real Estate Classifier")
    st.caption("Clasificaci\u00f3n autom\u00e1tica de fotos de anuncios inmobiliarios.")
    st.markdown(f"**API:** `{API_URL}`")

    health = fetch_health()
    if health and health.get("status") == "ok":
        st.success(f"Modelo cargado: `{health['model']}` en `{health['device']}`")
        st.caption(f"Imagen esperada: {health['image_size']}x{health['image_size']} px")
    else:
        st.error("API no disponible o sin modelo cargado.")

    st.divider()
    st.markdown(
        "**C\u00f3mo usarlo**\n\n"
        "1. Sube una foto del inmueble (sal\u00f3n, cocina, fachada, etc.).\n"
        "2. La API devuelve la clase principal y un top-3 de candidatos.\n"
        "3. Usa esos resultados para etiquetar autom\u00e1ticamente el anuncio."
    )


# --- Main --------------------------------------------------------------------
st.title("Clasificador de im\u00e1genes inmobiliarias")
st.markdown(
    "Sube una imagen del anuncio y obtendr\u00e1s la categor\u00eda predicha "
    "(sal\u00f3n, cocina, dormitorio, fachada, etc.) junto con un top-3."
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
        with st.spinner("Clasificando con la API\u2026"):
            result = call_predict(
                uploaded.getvalue(), uploaded.name, uploaded.type or "image/jpeg"
            )
        if result is not None:
            top1 = result["class"]
            business = result["business_label"]
            confidence = result["confidence"]
            st.subheader(f"\U0001F4C2 Categor\u00eda: **{business}**")
            st.caption(f"(etiqueta interna del modelo: `{top1}`)")
            st.progress(min(max(confidence, 0.0), 1.0), text=f"Confianza {confidence*100:.1f}%")

            df = pd.DataFrame(
                [
                    {
                        "Categor\u00eda (negocio)": item["business_label"],
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

            with st.expander("Detalles t\u00e9cnicos"):
                st.json(
                    {
                        "modelo": result["model_used"],
                        "dispositivo": result["inference_device"],
                        "tiempo_inferencia_ms": result["inference_time_ms"],
                    }
                )

st.divider()

with st.expander("Cat\u00e1logo de clases soportadas"):
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
