"""Collector output safety using mocked HTTP and temporary files only."""

import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[1]
TEXT = "A sufficiently long synthetic description for the test fixture."


@pytest.fixture(params=["wiki_ru", "wikidata"])
def collector(request, monkeypatch, tmp_path):
    kind = request.param
    spec = importlib.util.spec_from_file_location(f"collector_{kind}_under_test", ROOT / f"scripts/scrape_{kind}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "OUTPUT", tmp_path / "output")
    monkeypatch.setattr(module.time, "sleep", Mock())
    if kind == "wiki_ru":
        monkeypatch.setattr(module, "tqdm", lambda iterable, **kwargs: iterable)
        monkeypatch.setattr(module, "SEED_TERMS", ["Hb"])
        filename = "wiki_medical_terms.json"
    else:
        monkeypatch.setattr(module, "QUERIES", ["synthetic query"])
        filename = "wikidata_medical_terms.json"
    return module, kind, module.OUTPUT / filename


def response(kind, populated=False):
    result = Mock(status_code=200, text="synthetic response")
    if kind == "wiki_ru":
        pages = [{"title": "Hb", "extract": TEXT, "fullurl": "https://ru.wikipedia.org/wiki/Hb"}] if populated else []
        result.json.return_value = {"query": {"pages": pages}}
    else:
        rows = [{"itemLabel": {"value": "Hb"}, "itemDescription": {"value": TEXT}, "item": {"value": "http://www.wikidata.org/entity/Q123"}}] if populated else []
        result.json.return_value = {"results": {"bindings": rows}}
    return result


def existing_file(path):
    path.parent.mkdir(parents=True)
    original = b'[\n  {"term": "previous", "plain": "previous data"}\n]\n'
    path.write_bytes(original)
    return original


def test_network_failure_preserves_previous_file(collector, monkeypatch, capsys):
    module, _, path = collector
    original = existing_file(path)
    monkeypatch.setattr(module.requests, "get", Mock(side_effect=module.requests.exceptions.ConnectionError("synthetic offline failure")))
    assert module.main() == 1
    assert path.read_bytes() == original
    assert "unchanged" in capsys.readouterr().err


def test_empty_api_response_preserves_previous_file(collector, monkeypatch, capsys):
    module, kind, path = collector
    original = existing_file(path)
    monkeypatch.setattr(module.requests, "get", Mock(return_value=response(kind)))
    assert module.main() == 1
    assert path.read_bytes() == original
    assert "unchanged" in capsys.readouterr().err


def test_empty_result_does_not_create_output_directory(collector, monkeypatch):
    module, kind, _ = collector
    monkeypatch.setattr(module.requests, "get", Mock(return_value=response(kind)))
    assert module.main() == 1
    assert not module.OUTPUT.exists()


def test_success_still_saves_collected_content_and_source_url(collector, monkeypatch, capsys):
    module, kind, path = collector
    existing_file(path)
    monkeypatch.setattr(module.requests, "get", Mock(return_value=response(kind, populated=True)))
    assert module.main() == 0
    records = json.loads(path.read_text())
    assert len(records) == 1
    assert records[0]["term"] == "Hb"
    assert records[0]["plain" if kind == "wiki_ru" else "description"] == TEXT
    assert records[0]["source_url"] == ("https://ru.wikipedia.org/wiki/Hb" if kind == "wiki_ru" else "http://www.wikidata.org/entity/Q123")
    assert capsys.readouterr().err == ""


def test_partial_success_keeps_previous_save_behavior(collector, monkeypatch):
    module, kind, path = collector
    if kind == "wiki_ru":
        monkeypatch.setattr(module, "SEED_TERMS", ["Hb", "missing"])
    else:
        monkeypatch.setattr(module, "QUERIES", ["one", "two"])
    monkeypatch.setattr(module.requests, "get", Mock(side_effect=[response(kind, populated=True), response(kind)]))
    assert module.main() == 0
    assert [entry["term"] for entry in json.loads(path.read_text())] == ["Hb"]
