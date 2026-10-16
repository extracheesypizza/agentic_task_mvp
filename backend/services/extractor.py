"""
Structured extraction: raw OCR text → typed JSON.

Uses the local Ollama model configured as MODEL_EXTRACTION. Retries on
malformed JSON, because a 7B model occasionally emits prose around the
object or truncates a field.
"""

import json
import logging

from openai import OpenAIError

from backend.config import settings
from backend.schemas import Extraction
from backend.services.llm_client import chat

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """You are a DATA EXTRACTION TOOL for an educational medical
literacy application. Your task is purely structural: parse raw text and output
JSON. You are NOT providing medical advice, diagnosis, or treatment recommendations.

Given raw OCR text from a laboratory document, extract into JSON:
{
  "document_type": "lab_result | prescription | discharge_summary | unreadable",
  "findings": [
    {
      "parameter": "name of the measured substance",
      "value": "numeric value exactly as written",
      "unit": "unit of measurement",
      "reference_range": "normal range as printed on the document",
      "status": "below_normal | normal | above_normal | unknown"
    }
  ],
  "medications": [
    { "name": "...", "dosage": "...", "frequency": "..." }
  ]
}

Rules:
- Extract ONLY what is literally written. Never infer or invent values.
- If a field is absent, use null.
- If the text is garbled or unreadable, set "document_type" to "unreadable"
  and return empty lists.
- Output ONLY valid JSON. No markdown fences, no commentary."""


def _coerce(raw: str) -> dict:
    """Parse model output, tolerating stray markdown fences."""
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("Model returned no text")
    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(
            line for line in text.split("\n") if not line.strip().startswith("```")
        )
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(text[start:end])
        raise


def extract_structured(raw_text: str, max_retries: int = 3) -> dict:
    """
    Convert raw OCR text into a structured dict.

    Raises ValueError for invalid output (HTTP 502), RuntimeError for an
    unavailable model service (HTTP 503). No clinical values are inferred.
    """
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            raw = chat(
                system_prompt=EXTRACTION_PROMPT,
                user_prompt=raw_text,
                model=settings.model_extraction,
                temperature=0,
                max_tokens=1500,
                json_mode=True,
            )
        except OpenAIError as e:
            last_error = e
            logger.warning("Extraction attempt %d failed: %s", attempt + 1, type(e).__name__)
            continue

        try:
            data = Extraction.model_validate(_coerce(raw))
        except ValueError as e:
            last_error = e
            logger.warning("Extraction attempt %d returned invalid data", attempt + 1)
            # Nudge harder on the next attempt
            raw_text = (
                "IMPORTANT: Respond with ONLY a valid JSON object. "
                "No markdown, no explanation.\n\n" + raw_text
            )
            continue

        return data.model_dump()

    if isinstance(last_error, OpenAIError):
        raise RuntimeError("LLM service unavailable after extraction retries") from last_error
    raise ValueError(f"Failed to extract valid structured data after {max_retries} attempts") from last_error
