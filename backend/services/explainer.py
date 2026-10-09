"""
Explanation engine: structured findings + RAG context → plain-language Russian
explanation for a patient. Grounded in the retrieved glossary to limit
hallucination, and constrained to never diagnose.
"""

import json
import logging

from backend.config import settings
from backend.services.llm_client import chat

logger = logging.getLogger(__name__)

EXPLANATION_PROMPT = """Ты — образовательный медицинский ассистент, помогающий пациенту
понять результаты его анализов. Ты НЕ врач.

ПРАВИЛА (обязательные):
1. НИКОГДА не ставь диагноз. Не пиши «у вас X» или «это означает, что вы больны».
2. Не назначай лечение и не рекомендова препараты.
3. Всегда добавляй: «Обсудите эти результаты с вашим лечащим врачом».
4. Пиши простым русским языком, понятным неспециалисту.
5. Объясняй, что измеряет каждый показатель и зачем он нужен.
6. Отмечай значения: ✅ В норме, ⚠️ Незначительное отклонение, 🔴 Вне нормы.
7. В конце задай 3–5 вопросов, которые пациент может задать врачу.

СПРАВОЧНЫЕ ДАННЫЕ ИЗ НАШЕЙ МЕДИЦИНСКОЙ БАЗЫ:
{rag_context}

ДАННЫЕ ПАЦИЕНТА (извлечены из его документа):
{structured_json}

Напиши тёплое, понятное, структурированное объяснение на русском языке."""

NO_CONTEXT_FALLBACK = (
    "⚠️ Справочные данные по некоторым показателям не найдены в нашей базе. "
    "Обязательно уточните их значение у врача.\n\n"
)


def explain(structured: dict, rag_context: str) -> str:
    """Generate the patient-facing explanation."""
    prompt = EXPLANATION_PROMPT.format(
        rag_context=rag_context or "(справочные данные отсутствуют)",
        structured_json=json.dumps(structured, indent=2, ensure_ascii=False),
    )

    text = chat(
        system_prompt="Ты — образовательный медицинский ассистент. Ты не ставишь диагнозы.",
        user_prompt=prompt,
        model=settings.model_explanation,
        temperature=0.3,
        max_tokens=2000,
    )

    if not rag_context:
        text = NO_CONTEXT_FALLBACK + text

    logger.info(
        "Generated explanation for %d findings (model=%s)",
        len(structured.get("findings", [])),
        settings.model_explanation,
    )
    return text