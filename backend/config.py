from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Ollama (local LLM) ────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_api_key: str = "ollama"  # not validated by Ollama, SDK requires a value

    model_extraction: str = "medgemma:4b"
    model_explanation: str = "medgemma:4b"

    # ── Local embeddings (sentence-transformers) ──────────────────
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── ChromaDB ──────────────────────────────────────────────────
    chroma_persist_dir: str = "./data/chroma"

    # ── OCR ───────────────────────────────────────────────────────
    tesseract_lang: str = "eng+rus"

    # ── Upload limits ─────────────────────────────────────────────
    max_upload_size_mb: int = 10

    # ── App ───────────────────────────────────────────────────────
    backend_url: str = "http://localhost:8000"
    cors_origins: list[str] = ["http://localhost:8501", "http://frontend:8501"]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()