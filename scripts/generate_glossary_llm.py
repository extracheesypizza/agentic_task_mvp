"""
Use a free LLM via OpenRouter to generate plain-language Russian
explanations for medical terms. Handles None responses and rate limits.
"""

import json
import time
import os
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

MODEL = os.getenv("MODEL_EXPLANATION", "mistralai/mistral-7b-instruct:free")

# Fallback models to try if the primary returns None
FALLBACK_MODELS = [
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "poolside/laguna-s-2.1:free",
    "apodex/apodex-1.1-mini:free",
    "cohere/north-mini-code:free",
    "liquid/lfm-2.5-2.6b:free",
]

OUTPUT = Path("data/raw/wikipedia")
OUTPUT.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = """Ты — медицинский просветитель. Напиши простое объяснение медицинского термина для пациента без медицинского образования.

Ответь СТРОГО в формате JSON без markdown-обёрток:
{
  "term": "название",
  "plain": "простое объяснение, 2-4 предложения",
  "category": "общий анализ крови | биохимия крови | гормоны | коагулограмма | анализ мочи | инструментальное исследование | общий термин",
  "reference_range": "норма если знаешь, иначе пустая строка",
  "unit": "единица измерения если знаешь, иначе пустая строка",
  "what_it_means": "что означает отклонение, 1-2 предложения, БЕЗ диагноза"
}

ПРАВИЛА:
- Только русский язык.
- НЕ ставь диагнозы.
- Если не знаешь норму — оставь пустую строку."""


def call_llm(term: str, model: str) -> str | None:
    """Single LLM call. Returns raw text or None."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Напиши объяснение для термина: {term}"},
            ],
            temperature=0.3,
            max_tokens=600,
        )

        content = response.choices[0].message.content

        # ── THE FIX: check for None before .strip() ───────────────
        if content is None:
            return None

        return content.strip()

    except Exception as e:
        print(f"    API error: {e}")
        return None


def parse_json_response(text: str) -> dict | None:
    """Try to extract JSON from LLM output."""
    if not text:
        return None

    # Remove markdown code fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first line (```json) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                pass
    return None


def generate_explanation(term: str, max_retries: int = 3) -> dict | None:
    """Try primary model, then fallbacks. Retry on None."""

    models_to_try = [MODEL] + FALLBACK_MODELS

    for attempt in range(max_retries):
        for model in models_to_try:
            raw = call_llm(term, model)

            if raw is None:
                continue  # try next model

            print(f"  ✅ {term} → {model}")
            parsed = parse_json_response(raw)
            if parsed and parsed.get("plain"):
                return parsed

        # Wait before retrying (rate limit recovery)
        if attempt < max_retries - 1:
            wait = (attempt + 1) * 2
            time.sleep(wait)

    return None


def load_terms() -> list[str]:
    """Load terms from Wikipedia scrape + extras."""
    wiki_file = Path("data/raw/wikipedia/wiki_medical_terms.json")
    terms = []

    if wiki_file.exists():
        with open(wiki_file, encoding="utf-8") as f:
            data = json.load(f)
            terms = [item["term"] for item in data]
    else:
        print("⚠ No Wikipedia data found. Using built-in list.")

    # Fallback: built-in list if Wikipedia scrape produced nothing
    if not terms:
        terms = [
            "Гемоглобин", "Эритроциты", "Лейкоциты", "Тромбоциты",
            "Креатинин", "Мочевина", "Билирубин", "Глюкоза",
            "Холестерин", "Аланинаминотрансфераза", "Аспартатаминотрансфераза",
            "Тиреотропный гормон", "Инсулин", "Кортизол",
            "С-реактивный белок", "СОЭ", "Гематокрит",
            "Общий анализ крови", "Биохимический анализ крови",
            "Коагулограмма", "Общий анализ мочи",
            "Протромбиновое время", "Фибриноген",
            "Витамин D", "Витамин B12", "Фолиевая кислота",
            "Альбумин", "Общий белок", "Железо", "Ферритин",
            "Электрокардиография", "Ультразвуковое исследование",
            "Биопсия", "Анемия", "Тромбоз", "Ишемия",
            "Гипертония", "Сахарный диабет", "Гастрит", "Пневмония",
        ]

    return terms


if __name__ == "__main__":
    terms = load_terms()
    print(f"📋 Terms to process: {len(terms)}")
    print(f"🤖 Primary model: {MODEL}")
    print(f"🔄 Fallbacks: {FALLBACK_MODELS}")
    print()

    results = []
    failed = []

    for term in tqdm(terms, desc="Generating"):
        entry = generate_explanation(term)

        if entry and entry.get("plain"):
            entry["source"] = "llm_generated"
            results.append(entry)
        else:
            failed.append(term)

        # Polite delay between requests (free tier rate limit)
        time.sleep(1.5)

    # ── Save results ──────────────────────────────────────────────
    out = OUTPUT / "llm_glossary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Generated: {len(results)} entries → {out}")

    if failed:
        print(f"⚠ Failed ({len(failed)}):")
        for t in failed:
            print(f"   - {t}")

        # Save failed terms for a second run
        fail_file = OUTPUT / "failed_terms.json"
        with open(fail_file, "w", encoding="utf-8") as f:
            json.dump(failed, f, ensure_ascii=False, indent=2)
        print(f"   Saved to {fail_file} — rerun later to retry.")