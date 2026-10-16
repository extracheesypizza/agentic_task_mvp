"""Read-only corpus checks on small synthetic fixtures."""

import importlib
import json
from pathlib import Path

import pytest

from scripts.build_glossary import valid_entry


TEXT = "A sufficiently long synthetic glossary definition for testing."


def checker():
    try:
        return importlib.import_module("scripts.check_glossary")
    except ModuleNotFoundError:
        pytest.fail("The read-only corpus check command is missing")


def write_corpus(tmp_path, data):
    path = tmp_path / "glossary.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_legacy_records_without_urls_remain_acceptable(tmp_path, capsys):
    path = write_corpus(tmp_path, [{"term": "Hb", "plain": TEXT, "source": "wikipedia_ru"}])
    assert checker().main([str(path)]) == 0
    output = capsys.readouterr().out
    assert "Records: 1" in output
    assert "wikipedia_ru: 1" in output
    assert "Source URLs: 0/1" in output
    assert "WARNING" in output


def test_summary_counts_sources_duplicates_and_unusable_records():
    records = [
        {"term": "Hb", "plain": TEXT, "source": "wikipedia_ru", "source_url": "https://example.org/Hb"},
        {"term": " hb ", "plain": TEXT, "source": "wikidata"},
        {"term": "Empty", "plain": "   "},
        None,
    ]
    report = checker().inspect_records(records)
    assert report["total"] == 4
    assert report["sources"] == {"wikipedia_ru": 1, "wikidata": 1, "unknown": 2}
    assert report["invalid_rows"] == [3, 4]
    assert report["duplicates"] == {"hb": [1, 2]}
    assert report["empty_definitions"] == 2
    assert report["source_urls"] == 1
    assert report["missing_sources"] == 2


@pytest.mark.parametrize("record", [
    {"term": "Hb", "plain": TEXT},
    {"term": "Hb", "plain": "short"},
    {"term": "Hb", "plain": TEXT, "unit": []},
    {"term": 123, "plain": TEXT},
    {"term": "Hb", "plain": TEXT, "unit": None},
    "not a record",
])
def test_record_validity_matches_existing_builder(record):
    report = checker().inspect_records([record])
    assert (not report["invalid_rows"]) == valid_entry(record, "plain")


@pytest.mark.parametrize("url", [None, "", [], "javascript:alert(1)"])
def test_bad_optional_urls_are_warnings_not_structural_errors(tmp_path, capsys, url):
    path = write_corpus(tmp_path, [{"term": "Hb", "plain": TEXT, "source_url": url}])
    assert checker().main([str(path)]) == 0
    assert "Source URLs: 0/1" in capsys.readouterr().out


def test_duplicates_are_reported_with_row_numbers_and_nonzero_exit(tmp_path, capsys):
    path = write_corpus(tmp_path, [{"term": "Hb", "plain": TEXT}, {"term": " HB ", "plain": TEXT}])
    assert checker().main([str(path)]) == 1
    output = capsys.readouterr().out
    assert "Duplicate term groups: 1" in output
    assert "hb: rows 1, 2" in output


def test_invalid_rows_and_empty_definitions_are_reported(tmp_path, capsys):
    path = write_corpus(tmp_path, [{"term": "Hb", "plain": ""}, None])
    assert checker().main([str(path)]) == 1
    output = capsys.readouterr().out
    assert "Invalid records: 2" in output
    assert "Empty/unusable definitions: 2" in output


@pytest.mark.parametrize("data", [[], {}, None, "wrong shape"])
def test_empty_or_non_list_corpus_returns_nonzero(tmp_path, data):
    path = write_corpus(tmp_path, data)
    assert checker().main([str(path)]) == 1


@pytest.mark.parametrize("content", [b"not JSON", b"\xff"])
def test_unreadable_content_has_clear_error_without_traceback(tmp_path, capsys, content):
    path = tmp_path / "bad.json"
    path.write_bytes(content)
    assert checker().main([str(path)]) == 1
    captured = capsys.readouterr()
    assert "Could not check corpus" in captured.err
    assert "Traceback" not in captured.err


def test_missing_file_does_not_create_directories(tmp_path, capsys):
    assert checker().main([str(tmp_path / "absent/glossary.json")]) == 1
    assert list(tmp_path.iterdir()) == []
    assert "Could not check corpus" in capsys.readouterr().err


def test_check_does_not_change_content_mtime_or_create_files(tmp_path):
    path = write_corpus(tmp_path, [{"term": "Hb", "plain": TEXT}])
    before = path.read_bytes(), path.stat().st_mtime_ns, sorted(tmp_path.iterdir())
    assert checker().main([str(path)]) == 0
    assert (path.read_bytes(), path.stat().st_mtime_ns, sorted(tmp_path.iterdir())) == before


def test_default_input_is_the_existing_builder_output(monkeypatch, tmp_path):
    module = checker()
    path = write_corpus(tmp_path, [{"term": "Hb", "plain": TEXT}])
    monkeypatch.setattr(module, "DEFAULT_PATH", path)
    monkeypatch.chdir(tmp_path)
    assert module.main([]) == 0
