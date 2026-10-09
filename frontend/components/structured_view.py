# frontend/components/structured_view.py
"""
Render the structured extraction result (findings, medications, metadata).

Nesting-safe by design: uses st.toggle instead of st.expander for the raw
JSON panel, so this component can be called from inside any container
(tabs, expanders, columns) without hitting Streamlit's nested-expander
restriction.
"""

import json

import streamlit as st

STATUS_ICONS = {
    "normal": "✅",
    "below_normal": "⚠️",
    "above_normal": "🔴",
    "critical": "🔴",
    "unreadable": "❓",
}

STATUS_LABELS = {
    "normal": "В норме",
    "below_normal": "Ниже нормы",
    "above_normal": "Выше нормы",
    "critical": "Критично",
    "unreadable": "Не распознано",
}

DOC_TYPE_LABELS = {
    "lab_result": "Результаты анализов",
    "prescription": "Рецепт",
    "discharge_summary": "Выписной эпикриз",
    "other": "Другой документ",
    "unreadable": "Документ не распознан",
}


def render_structured(data: dict, show_raw_toggle: bool = True) -> None:
    """Render the extracted structured data."""
    if not data:
        st.info("Нет структурных данных для отображения.")
        return

    _render_header(data)
    _render_findings(data.get("findings") or [])
    _render_medications(data.get("medications") or [])
    _render_notes(data.get("raw_notes"))

    if show_raw_toggle:
        _render_raw_json(data)


def _render_header(data: dict) -> None:
    doc_type = data.get("document_type") or "other"
    label = DOC_TYPE_LABELS.get(doc_type, doc_type)
    st.caption(f"📄 Тип документа: **{label}**")

    patient = data.get("patient_info") or {}
    if patient.get("name") or patient.get("date"):
        parts = [p for p in (patient.get("name"), patient.get("date")) if p]
        st.caption(" · ".join(str(p) for p in parts))


def _render_findings(findings: list[dict]) -> None:
    if not findings:
        return

    st.subheader("Показатели")

    for f in findings:
        status = (f.get("status") or "").strip()
        icon = STATUS_ICONS.get(status, "❓")
        name = f.get("parameter") or "—"
        value = f.get("value")
        unit = f.get("unit") or ""
        ref = f.get("reference_range") or ""

        value_str = f"{value} {unit}".strip() if value not in (None, "") else "—"

        st.markdown(f"{icon} **{name}** — `{value_str}`")
        if ref:
            st.caption(f"Референсные значения: {ref}")


def _render_medications(medications: list[dict]) -> None:
    if not medications:
        return

    st.subheader("Назначения")
    for m in medications:
        name = m.get("name") or "—"
        dosage = m.get("dosage") or ""
        frequency = m.get("frequency") or ""
        detail = " · ".join(x for x in (dosage, frequency) if x)
        st.markdown(f"💊 **{name}**" + (f" — {detail}" if detail else ""))


def _render_notes(raw_notes: str | None) -> None:
    if raw_notes and raw_notes.strip():
        st.subheader("Заметки врача")
        st.write(raw_notes.strip())


def _render_raw_json(data: dict) -> None:
    """
    Debug view for the raw extraction JSON.

    Uses st.toggle rather than st.expander so this component stays
    nestable. The key includes id(data) to avoid state collisions if the
    component is rendered more than once on a page.
    """
    toggle_key = f"show_raw_json_{id(data)}"

    if st.toggle("🔧 Показать исходный JSON", value=False, key=toggle_key):
        st.json(data, expanded=False)
        st.download_button(
            "⬇️ Скачать JSON",
            data=json.dumps(data, ensure_ascii=False, indent=2),
            file_name="extraction.json",
            mime="application/json",
            key=f"dl_raw_json_{id(data)}",
        )