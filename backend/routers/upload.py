"""
POST /api/analyze — main endpoint.
Upload an image → get a plain-language medical explanation.
"""

import logging

from fastapi import APIRouter, UploadFile, File, HTTPException
from openai import OpenAIError
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

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

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    image_bytes = await file.read(max_bytes + 1)
    if len(image_bytes) > max_bytes:
        raise HTTPException(413, f"File too large (max {settings.max_upload_size_mb} MB).")

    # ── 1. OCR (local, free) ─────────────────────────────────────
    try:
        raw_text = await run_in_threadpool(extract_text_from_image, image_bytes, lang=settings.tesseract_lang)
    except (UnidentifiedImageError, Image.DecompressionBombError):
        raise HTTPException(422, "Could not decode this image. Upload a valid PNG/JPG.")
    except Exception as e:
        logger.error("OCR failed: %s", type(e).__name__)
        raise HTTPException(500, "OCR processing failed.")

    if len(raw_text.strip()) < 20:
        raise HTTPException(
            422,
            "Could not read text from this image. Try a clearer, well-lit photo.",
        )

    # ── 2. Structured extraction (LLM with fallback) ─────────────
    try:
        structured = await run_in_threadpool(extract_structured, raw_text)
    except ValueError:
        raise HTTPException(502, "The model returned invalid document data. Please try again.")
    except (RuntimeError, OpenAIError):
        logger.warning("Extraction model service unavailable")
        raise HTTPException(503, "LLM service unavailable. Try again later.")

    if structured.get("document_type") == "unreadable":
        raise HTTPException(422, "Could not understand this document. Try a clearer photo.")

    # ── 3. RAG context (local embeddings) ────────────────────────
    terms = [f["parameter"] for f in structured.get("findings", []) if f.get("parameter")]
    try:
        rag_context = await run_in_threadpool(get_context, terms) if terms else ""
    except Exception as e:
        logger.error("Reference retrieval failed: %s", type(e).__name__)
        raise HTTPException(503, "Reference service unavailable. Try again later.")

    # ── 4. Explanation (LLM with fallback) ───────────────────────
    try:
        explanation = await run_in_threadpool(explain, structured, rag_context)
    except ValueError:
        raise HTTPException(502, "The model returned an invalid explanation. Please try again.")
    except (RuntimeError, OpenAIError):
        logger.warning("Explanation model service unavailable")
        raise HTTPException(503, "LLM service unavailable. Try again later.")

    if not isinstance(explanation, str) or not explanation.strip():
        raise HTTPException(502, "The model returned an empty explanation. Please try again.")

    # ── 5. Safety filter ─────────────────────────────────────────
    explanation = sanitize(explanation)

    return {
        "raw_text": raw_text,
        "structured": structured,
        "explanation": explanation,
    }
