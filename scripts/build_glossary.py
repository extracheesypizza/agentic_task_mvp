# scripts/build_glossary.py
"""
Merge all sources into the final glossary.json for ChromaDB.
Priority: LLM-generated > Wikipedia > Wikidata (dedup by term name).
"""

import json
from pathlib import Path

DATA = Path("data/raw")
OUTPUT = Path("backend/rag/glossary.json")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)


def load(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build():
    entries = []
    seen = set()

    # ── Priority 1: LLM-generated (best quality) ─────────────────
    for item in load(DATA / "llm_generated/llm_glossary.json"):
        term = item.get("term", "").strip().lower()
        if not term or term in seen:
            continue
        seen.add(term)
        entries.append({
            "term": item["term"],
            "plain": item.get("plain", ""),
            "category": item.get("category", ""),
            "reference_range": item.get("reference_range", ""),
            "unit": item.get("unit", ""),
            "what_it_means": item.get("what_it_means", ""),
            "source": "llm_generated",
        })

    # ── Priority 2: Wikipedia RU ─────────────────────────────────
    for item in load(DATA / "wikipedia/wiki_medical_terms.json"):
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

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    # Stats
    from collections import Counter
    sources = Counter(e["source"] for e in entries)
    print(f"✅ Final glossary: {len(entries)} terms → {OUTPUT}")
    print(f"   By source: {dict(sources)}")


if __name__ == "__main__":
    build()