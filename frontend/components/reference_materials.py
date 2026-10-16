"""Display external reference materials returned by the analysis API."""

from urllib.parse import urlsplit

import streamlit as st


def _safe_http_url(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = urlsplit(value)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            return None
        # Accessing port also validates malformed port numbers.
        parsed.port
    except ValueError:
        return None
    return value


def render_reference_materials(materials):
    """Render a collapsed list of retrieved materials, if any were returned."""
    if not isinstance(materials, list) or not materials:
        return

    with st.expander("📚 Найденные справочные материалы", expanded=False):
        st.caption(
            "Результаты поиска для справки; они не являются цитатами, "
            "подтверждающими каждое утверждение модели."
        )
        for material in materials:
            if not isinstance(material, dict) or not isinstance(material.get("term"), str):
                continue
            st.text(material["term"])
            source = material.get("source")
            st.text(source if isinstance(source, str) and source else "Источник не указан")
            url = _safe_http_url(material.get("source_url"))
            if url:
                st.link_button("Открыть источник", url)
            else:
                st.text("Ссылка не указана")
