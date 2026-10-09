"""
Structured extraction: raw OCR text → typed JSON.

Uses the local Ollama model configured as MODEL_EXTRACTION. Retries on
malformed JSON, because a 7B model occasionally emits prose around the
object or truncates a field.
"""

import json
import logging

from backend.config import settings
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


def _is_valid(data: dict) -> bool:
    """A usable extraction has a document_type and at least one finding or medication."""
    if not isinstance(data, dict) or not data.get("document_type"):
        return False
    return bool(data.get("findings") or data.get("medications"))


def extract_structured(raw_text: str, max_retries: int = 3) -> dict:
    """
    Convert raw OCR text into a structured dict.

    Raises ValueError if the model cannot produce valid JSON after retries,
    so the caller returns a clean 502 instead of a 500 traceback.
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
        except Exception as e:
            last_error = e
            logger.warning("Extraction attempt %d failed: %s", attempt + 1, e)
            continue

        try:
            data = _coerce(raw)
        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            logger.warning("Extraction attempt %d returned invalid JSON: %s", attempt + 1, e)
            # Nudge harder on the next attempt
            raw_text = (
                "IMPORTANT: Respond with ONLY a valid JSON object. "
                "No markdown, no explanation.\n\n" + raw_text
            )
            continue

        if _is_valid(data):
            return data

        last_error = ValueError(f"Extraction missing required fields: {list(data.keys())}")
        logger.warning("Extraction attempt %d produced an empty result", attempt + 1)

    raise ValueError(f"Failed to extract structured data after {max_retries} attempts: {last_error}")