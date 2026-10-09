"""Medical disclaimer — always shown after results."""

import streamlit as st


def render_disclaimer():
    st.divider()
    st.warning(
        "⚕️ **Важно:** Это информационное объяснение, а не диагноз и не медицинская рекомендация. "
        "Для интерпретации результатов обязательно обратитесь к врачу.",
        icon="⚠️",
    )