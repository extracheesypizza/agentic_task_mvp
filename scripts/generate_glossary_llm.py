"""
Use a free LLM via OpenRouter to generate draft plain-language Russian
explanations for medical terms. Handles invalid responses and bounded retries.
"""

import json
import time
import os
import sys
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/raw/llm_generated"

# Preserve the original free-model defaults; availability is provider-dependent.
DEFAULT_MODEL = "mistralai/mistral-7b-instruct:free"
FALLBACK_MODELS = [
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "poolside/laguna-s-2.1:free",
    "apodex/apodex-1.1-mini:free",
    "cohere/north-mini-code:free",
    "liquid/lfm-2.5-2.6b:free",
]


def load_config() -> dict[str, str]:
    """Read generator-only settings without changing the app environment."""
    config_file = ROOT / ".env.glossary.local"
    local = dotenv_values(config_file) if config_file.exists() else {}

    def value(name: str, default: str = "") -> str:
        return (os.getenv(name, local.get(name) or default) or "").strip()

    config = {
        "api_key": value("OPENROUTER_API_KEY"),
        "base_url": value("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        "model": value("GLOSSARY_MODEL", DEFAULT_MODEL),
    }
    missing = [name for name, key in [("OPENROUTER_API_KEY", "api_key"), ("GLOSSARY_MODEL", "model")] if not config[key]]
    if missing:
        raise ValueError("Set " + ", ".join(missing) + " in the environment or .env.glossary.local before generation.")
    return config


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


def call_llm(term: str, model: str, client: OpenAI) -> str | None:
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

        if not isinstance(content, str):
            return None

        return content.strip()

    except Exception as e:
        print(f"    API error: {type(e).__name__}")
        return None


def parse_json_response(text: str) -> dict | None:
    """Try to extract JSON from LLM output."""
    if not isinstance(text, str) or not text.strip():
        return None

    # Remove markdown code fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first line (```json) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            try:
                data = json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                return None
        else:
            return None
    if not isinstance(data, dict):
        return None
    if not all(isinstance(data.get(key), str) and data[key].strip() for key in ("term", "plain")):
        return None
    if any(data.get(key) is not None and not isinstance(data[key], str)
           for key in ("category", "reference_range", "unit", "what_it_means")):
        return None
    return data


def generate_explanation(term: str, model: str, client: OpenAI, max_retries: int = 3) -> dict | None:
    """Try the primary model, then the existing free fallbacks, as before."""

    for attempt in range(max_retries):
        for candidate in [model] + FALLBACK_MODELS:
            raw = call_llm(term, candidate, client)
            parsed = parse_json_response(raw)
            if parsed and parsed["term"].strip().casefold() == term.strip().casefold():
                print(f"  ✅ {term} → {candidate}")
                return parsed

        # Wait before retrying (rate limit recovery)
        if attempt < max_retries - 1:
            wait = (attempt + 1) * 2
            time.sleep(wait)

    return None


def load_terms() -> list[str]:
    """Load terms from Wikipedia scrape + extras."""
    wiki_file = ROOT / "data/raw/wikipedia/wiki_medical_terms.json"
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


def main() -> int:
    try:
        config = load_config()
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1

    client = OpenAI(base_url=config["base_url"], api_key=config["api_key"])
    try:
        return run_generation(client, config["model"])
    finally:
        client.close()


def run_generation(client: OpenAI, model: str) -> int:
    terms = load_terms()
    print(f"📋 Terms to process: {len(terms)}")
    print(f"🤖 Primary model: {model}")
    print(f"🔄 Fallbacks: {FALLBACK_MODELS}")
    print()

    results = []
    failed = []

    for term in tqdm(terms, desc="Generating"):
        entry = generate_explanation(term, model, client)

        if entry and entry.get("plain"):
            entry["source"] = "llm_generated"
            results.append(entry)
        else:
            failed.append(term)

        # Polite delay between requests (free tier rate limit)
        time.sleep(1.5)

    if not results:
        print("No entries generated; existing output files were not changed.", file=sys.stderr)
        return 1

    # ── Save results ──────────────────────────────────────────────
    OUTPUT.mkdir(parents=True, exist_ok=True)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
