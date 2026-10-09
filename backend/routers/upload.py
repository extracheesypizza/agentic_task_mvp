"""
POST /api/analyze — main endpoint.
Upload an image → get a plain-language medical explanation.
"""

import logging

from fastapi import APIRouter, UploadFile, File, HTTPException

from backend.config import settings
from backend.services.ocr import extract_text_from_image
from backend.services.extractor import extract_structured
from backend.rag.retriever import get_context
from backend.services.explainer import explain
from backend.services.safety import sanitize

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    # ── Validate ──────────────────────────────────────────────────
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Upload an image (PNG/JPG).")

    image_bytes = await file.read()
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > settings.max_upload_size_mb:
        raise HTTPException(413, f"File too large (max {settings.max_upload_size_mb} MB).")

    # ── 1. OCR (local, free) ─────────────────────────────────────
    try:
        raw_text = extract_text_from_image(image_bytes, lang=settings.tesseract_lang)
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        raise HTTPException(500, "OCR processing failed.")

    if len(raw_text.strip()) < 20:
        raise HTTPException(
            422,
            "Could not read text from this image. Try a clearer, well-lit photo.",
        )

    # ── 2. Structured extraction (LLM with fallback) ─────────────
    try:
        structured = extract_structured(raw_text)
    except RuntimeError as e:
        logger.error(f"Extraction failed: {e}")
        raise HTTPException(503, "LLM service unavailable. Try again later.")

    # ── 3. RAG context (local embeddings) ────────────────────────
    terms = [f["parameter"] for f in structured.get("findings", []) if f.get("parameter")]
    rag_context = get_context(terms) if terms else ""

    # ── 4. Explanation (LLM with fallback) ───────────────────────
    try:
        explanation = explain(structured, rag_context)
    except RuntimeError as e:
        logger.error(f"Explanation failed: {e}")
        raise HTTPException(503, "LLM service unavailable. Try again later.")

    # ── 5. Safety filter ─────────────────────────────────────────
    explanation = sanitize(explanation)

    return {
        "raw_text": raw_text,
        "structured": structured,
        "explanation": explanation,
    }