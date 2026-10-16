"""
MedTranslate — Streamlit frontend.
Run: streamlit run frontend/app.py --server.port 8501
"""

import streamlit as st
import requests
import os
import hashlib

from config import APP_TITLE, APP_SUBTITLE, BACKEND_URL
from components.upload_card import render_upload
from components.result_display import render_explanation
from components.structured_view import render_structured
from components.reference_materials import render_reference_materials
from components.disclaimer import render_disclaimer

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedTranslate",
    page_icon="🩺",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────
css_path = os.path.join(os.path.dirname(__file__), "styles", "custom.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)
st.divider()

# ── Upload section ────────────────────────────────────────────────────
uploaded_file = render_upload()

if uploaded_file is not None:
    upload_identity = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
    if st.session_state.get("analysis_upload_identity") != upload_identity:
        st.session_state["analysis_upload_identity"] = upload_identity
        st.session_state["analysis_result"] = None

    if st.button("🔍 Анализировать", type="primary", use_container_width=True):
        # A failed request must not leave the previous analysis visible.
        st.session_state["analysis_result"] = None
        with st.spinner("Читаем документ... Это может занять 10–20 секунд."):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/api/analyze",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                if (
                    not isinstance(data, dict)
                    or not isinstance(data.get("explanation"), str)
                    or not isinstance(data.get("structured"), dict)
                ):
                    raise ValueError("unexpected response shape")
                st.session_state["analysis_result"] = data
            except requests.exceptions.ConnectionError:
                st.error("❌ Не удалось подключиться к серверу.")
            except requests.exceptions.Timeout:
                st.error("⏱ Сервер не ответил вовремя. Попробуйте ещё раз.")
            except requests.exceptions.HTTPError as e:
                st.error(f"❌ Ошибка: {e.response.status_code} — {e.response.text}")
            except (ValueError, requests.exceptions.RequestException):
                st.error("❌ Сервер вернул некорректный ответ. Попробуйте ещё раз.")
else:
    st.session_state.pop("analysis_upload_identity", None)
    st.session_state.pop("analysis_result", None)

# ── Results ─────────────────────────────────────────────────────────
data = st.session_state.get("analysis_result")
if data is not None:
    st.divider()
    render_explanation(data["explanation"])

    with st.expander("📋 Структурированные данные"):
        render_structured(data["structured"])

    render_reference_materials(data.get("retrieved_materials"))

    render_disclaimer()
