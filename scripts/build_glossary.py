# scripts/build_glossary.py
"""
Merge all sources into the final glossary.json for ChromaDB.
Priority: LLM-generated > Wikipedia > Wikidata (dedup by term name).
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw"
OUTPUT = ROOT / "backend/rag/glossary.json"


def load(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of records in {path}")
    return data


def valid_entry(item, text_key: str) -> bool:
    """Reject broken records before they can shadow a usable source."""
    return (
        isinstance(item, dict)
        and isinstance(item.get("term"), str)
        and bool(item["term"].strip())
        and isinstance(item.get(text_key), str)
        and len(item[text_key].strip()) > 30
        and all(item.get(key) is None or isinstance(item[key], str)
                for key in ("category", "reference_range", "unit", "what_it_means"))
    )


def build():
    entries = []
    seen = set()

    # Existing source priority is preserved; it is not a quality guarantee.
    # ── Priority 1: LLM-generated ───────────────────────────────
    for item in load(DATA / "llm_generated/llm_glossary.json"):
        if not valid_entry(item, "plain"):
            continue
        term = item.get("term", "").strip().lower()
        if not term or term in seen:
            continue
        seen.add(term)
        entries.append({
            "term": item["term"],
            "plain": item.get("plain", ""),
            "category": item.get("category") or "",
            "reference_range": item.get("reference_range") or "",
            "unit": item.get("unit") or "",
            "what_it_means": item.get("what_it_means") or "",
            "source": "llm_generated",
        })

    # ── Priority 2: Wikipedia RU ─────────────────────────────────
    for item in load(DATA / "wikipedia/wiki_medical_terms.json"):
        if not valid_entry(item, "plain"):
            continue
        term = item.get("term", "").strip().lower()
        if not term or term in seen:
            continue
        seen.add(term)
        entries.append({
            "term": item["term"],
            "plain": item.get("plain", "")[:600],
            "category": "medical_term",
            "reference_range": "",
            "unit": "",
            "what_it_means": "",
            "source": "wikipedia_ru",
        })

    # ── Priority 3: Wikidata ─────────────────────────────────────
    for item in load(DATA / "wikidata/wikidata_medical_terms.json"):
        if not valid_entry(item, "description"):
            continue
        term = item.get("term", "").strip().lower()
        if not term or term in seen:
            continue
        seen.add(term)
        desc = item.get("description", "")
        if len(desc) > 20:
            entries.append({
                "term": item["term"],
                "plain": desc,
                "category": "medical_term",
                "reference_range": "",
                "unit": "",
                "what_it_means": "",
                "source": "wikidata",
            })

    # ── Filter: keep only entries with meaningful content ────────
    entries = [e for e in entries if len(e["plain"]) > 30]
    if not entries:
        raise ValueError("No usable glossary records; existing output was not changed")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    # Stats
    from collections import Counter
    sources = Counter(e["source"] for e in entries)
    print(f"✅ Final glossary: {len(entries)} terms → {OUTPUT}")
    print(f"   By source: {dict(sources)}")


if __name__ == "__main__":
    build()
