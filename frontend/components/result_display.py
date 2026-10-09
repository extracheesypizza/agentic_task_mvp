"""Renders the LLM-generated plain-language explanation."""

import streamlit as st


def render_explanation(text: str):
    """Display the explanation in a nice card."""
    st.subheader("📝 Объяснение простыми словами")

    # Render markdown (the LLM output uses ✅ ⚠️ 🔴 markers)
    st.markdown(text)