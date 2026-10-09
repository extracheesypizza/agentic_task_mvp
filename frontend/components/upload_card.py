"""File upload widget with preview and validation."""

import streamlit as st
from PIL import Image
import io

from config import ALLOWED_EXTENSIONS, MAX_UPLOAD_SIZE_MB


def render_upload():
    """Renders the upload area. Returns the uploaded file or None."""

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded = st.file_uploader(
            label=f"Загрузите фото анализа (PNG, JPG, до {MAX_UPLOAD_SIZE_MB} МБ)",
            type=ALLOWED_EXTENSIONS,
            help="Сфотографируйте результат анализа так, чтобы текст был читаемым.",
        )

    with col2:
        if uploaded is not None:
            # Validate size
            size_mb = uploaded.size / (1024 * 1024)
            if size_mb > MAX_UPLOAD_SIZE_MB:
                st.warning(f"⚠️ Файл слишком большой ({size_mb:.1f} МБ). Максимум: {MAX_UPLOAD_SIZE_MB} МБ.")
                return None

            # Show preview
            try:
                image = Image.open(io.BytesIO(uploaded.getvalue()))
                st.image(image, caption="Предпросмотр", use_container_width=True)
            except Exception:
                st.warning("Не удалось отобразить предпросмотр.")

            st.success(f"📄 {uploaded.name} ({size_mb:.1f} МБ)")

    return uploaded