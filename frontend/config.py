"""Frontend configuration — where the backend lives, limits, etc."""

import os

# Backend URL: inside Docker it's the service name, locally it's localhost
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Upload constraints (must match backend)
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
ALLOWED_EXTENSIONS = ["png", "jpg", "jpeg"]
ALLOWED_MIME_TYPES = ["image/png", "image/jpeg"]

# App metadata
APP_TITLE = "🩺 MedTranslate"
APP_SUBTITLE = "Загрузите фото анализа — получите понятное объяснение"
APP_VERSION = "0.1.0"