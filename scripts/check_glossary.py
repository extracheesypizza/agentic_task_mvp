"""Read-only structural/provenance summary: python -m scripts.check_glossary."""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from scripts.build_glossary import OUTPUT as DEFAULT_PATH
from scripts.build_glossary import _source_metadata, valid_entry


def inspect_records(records: list) -> dict:
    if not isinstance(records, list):
        raise ValueError("Expected a JSON list of glossary records")

    sources = Counter()
    term_rows = defaultdict(list)
    invalid_rows = []
    empty_definitions = source_urls = missing_sources = 0

    for row, item in enumerate(records, start=1):
        if not valid_entry(item, "plain"):
            invalid_rows.append(row)
        record = item if isinstance(item, dict) else {}
        text = record.get("plain")
        if not isinstance(text, str) or not text.strip():
            empty_definitions += 1
        term = record.get("term")
        if isinstance(term, str) and term.strip():
            term_rows[term.strip().lower()].append(row)
        source = record.get("source")
        if not isinstance(source, str) or not source.strip():
            missing_sources += 1
            source = "unknown"
        sources[source.strip()] += 1
        if _source_metadata(record):
            source_urls += 1

    duplicates = {term: rows for term, rows in term_rows.items() if len(rows) > 1}
    return {
        "total": len(records),
        "sources": dict(sources),
        "invalid_rows": invalid_rows,
        "duplicates": duplicates,
        "empty_definitions": empty_definitions,
        "source_urls": source_urls,
        "missing_sources": missing_sources,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check glossary structure and source coverage without changing files.")
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_PATH)
    path = parser.parse_args(argv).path
    try:
        report = inspect_records(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        print(f"Could not check corpus: {exc}", file=sys.stderr)
        return 1

    print(f"Corpus: {path}")
    print(f"Records: {report['total']}")
    print("Sources:")
    for source, count in sorted(report["sources"].items()):
        print(f"  {source}: {count}")
    print(f"Invalid records: {len(report['invalid_rows'])}")
    if report["invalid_rows"]:
        print("  Rows: " + ", ".join(map(str, report["invalid_rows"])))
    print(f"Empty/unusable definitions: {report['empty_definitions']}")
    print(f"Duplicate term groups: {len(report['duplicates'])}")
    for term, rows in sorted(report["duplicates"].items()):
        print(f"  {term}: rows " + ", ".join(map(str, rows)))
    print(f"Source URLs: {report['source_urls']}/{report['total']}")
    if report["source_urls"] < report["total"]:
        print("WARNING: some records have no HTTP(S) source URL; legacy records remain supported.")
    if report["missing_sources"]:
        print(f"WARNING: source label missing or invalid in {report['missing_sources']} records.")

    has_errors = not report["total"] or bool(report["invalid_rows"] or report["duplicates"])
    print("Result: " + ("structural errors found" if has_errors else "structure OK (medical accuracy not assessed)"))
    return 1 if has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
