# scripts/scrape_wikidata.py
"""
Query Wikidata for medical lab tests with Russian labels and descriptions.
SPARQL endpoint: https://query.wikidata.org/sparql
"""

import requests
import json
import time
from pathlib import Path
from tqdm import tqdm

OUTPUT = Path("data/raw/wikidata")
OUTPUT.mkdir(parents=True, exist_ok=True)

SPARQL_URL = "https://query.wikidata.org/sparql"
HEADERS = {
    "User-Agent": "MedTranslateBot/1.0 (student project)",
    "Accept": "application/sparql-results+json",
}

# Query: medical tests and procedures with Russian labels + descriptions
QUERIES = [
    # Lab tests
    """
    SELECT ?item ?itemLabel ?itemDescription WHERE {
      ?item wdt:P31 wd:Q210980.
      ?item rdfs:label ?itemLabel.
      FILTER(LANG(?itemLabel) = "ru").
      OPTIONAL { ?item schema:description ?itemDescription. FILTER(LANG(?itemDescription) = "ru"). }
    }
    LIMIT 200
    """,
    # Blood tests specifically
    """
    SELECT ?item ?itemLabel ?itemDescription WHERE {
      ?item wdt:P279 wd:Q189594.
      ?item rdfs:label ?itemLabel.
      FILTER(LANG(?itemLabel) = "ru").
      OPTIONAL { ?item schema:description ?itemDescription. FILTER(LANG(?itemDescription) = "ru"). }
    }
    LIMIT 100
    """,
    # Medical procedures
    """
    SELECT ?item ?itemLabel ?itemDescription WHERE {
      ?item wdt:P31 wd:Q192162.
      ?item rdfs:label ?itemLabel.
      FILTER(LANG(?itemLabel) = "ru").
      OPTIONAL { ?item schema:description ?itemDescription. FILTER(LANG(?itemDescription) = "ru"). }
    }
    LIMIT 100
    """,
]


def run_query(sparql: str) -> list[dict]:
    try:
        resp = requests.get(
            SPARQL_URL,
            params={"query": sparql},
            headers=HEADERS,
            timeout=30,
        )
        if resp.status_code == 429:
            print("  ⚠ Rate limited, waiting 10s...")
            time.sleep(10)
            resp = requests.get(
                SPARQL_URL, params={"query": sparql}, headers=HEADERS, timeout=30
            )
        resp.raise_for_status()
        data = resp.json()
        bindings = data.get("results", {}).get("bindings", [])

        results = []
        for b in bindings:
            term = b.get("itemLabel", {}).get("value", "")
            desc = b.get("itemDescription", {}).get("value", "")
            if term:
                results.append({"term": term, "description": desc})
        return results

    except Exception as e:
        print(f"  ✗ SPARQL error: {e}")
        return []


if __name__ == "__main__":
    all_results = []
    seen = set()

    for i, query in enumerate(QUERIES):
        print(f"\n🔍 Running query {i+1}/{len(QUERIES)}...")
        rows = run_query(query)
        for row in rows:
            t = row["term"].lower()
            if t not in seen:
                seen.add(t)
                all_results.append(row)
        time.sleep(2)  # Wikidata rate limit

    out = OUTPUT / "wikidata_medical_terms.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved {len(all_results)} terms → {out}")