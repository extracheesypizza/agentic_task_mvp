"""Optional source provenance, with mocked APIs/index and temporary input files."""

import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[1]
TEXT = "A sufficiently long synthetic description used only for testing."


def import_file(monkeypatch, name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def response(data):
    result = Mock(status_code=200, text="synthetic response")
    result.json.return_value = data
    return result


@pytest.fixture
def wiki(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    return import_file(monkeypatch, "wiki_urls_under_test", "scripts/scrape_wiki_ru.py")


def test_wikipedia_preserves_url_returned_for_resolved_page(wiki, monkeypatch):
    url = "https://ru.wikipedia.org/wiki/Гемоглобин"
    get = Mock(return_value=response({"query": {"pages": [{"title": "Гемоглобин", "extract": TEXT, "fullurl": url}]}}))
    monkeypatch.setattr(wiki.requests, "get", get)
    entry = wiki.fetch_article("Hb")
    assert entry == {"term": "Гемоглобин", "plain": TEXT, "source": "wikipedia_ru", "source_url": url}
    assert "info" in get.call_args.kwargs["params"]["prop"]
    assert get.call_args.kwargs["params"]["inprop"] == "url"


def test_wikipedia_without_url_keeps_legacy_record_shape(wiki, monkeypatch):
    monkeypatch.setattr(wiki.requests, "get", Mock(return_value=response({"query": {"pages": [{"title": "Hb", "extract": TEXT}]}})))
    assert wiki.fetch_article("Hb") == {"term": "Hb", "plain": TEXT, "source": "wikipedia_ru"}


def test_missing_wikipedia_page_still_returns_none(wiki, monkeypatch):
    monkeypatch.setattr(wiki.requests, "get", Mock(return_value=response({"query": {"pages": [{"missing": True}]}})))
    assert wiki.fetch_article("missing") is None


@pytest.mark.parametrize("item_url", ["http://www.wikidata.org/entity/Q123", None])
def test_wikidata_preserves_item_uri_only_when_present(monkeypatch, tmp_path, item_url):
    monkeypatch.chdir(tmp_path)
    module = import_file(monkeypatch, "wikidata_urls_under_test", "scripts/scrape_wikidata.py")
    binding = {"itemLabel": {"value": "Hb"}, "itemDescription": {"value": TEXT}}
    if item_url:
        binding["item"] = {"value": item_url}
    monkeypatch.setattr(module.requests, "get", Mock(return_value=response({"results": {"bindings": [binding]}})))
    expected = {"term": "Hb", "description": TEXT}
    if item_url:
        expected["source_url"] = item_url
    assert module.run_query("synthetic query") == [expected]


def test_scraper_imports_do_not_create_data_directories(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    import_file(monkeypatch, "wiki_import_under_test", "scripts/scrape_wiki_ru.py")
    import_file(monkeypatch, "wikidata_import_under_test", "scripts/scrape_wikidata.py")
    assert list(tmp_path.iterdir()) == []


def merge(monkeypatch, tmp_path, entries_by_path):
    builder = import_file(monkeypatch, "builder_urls_under_test", "scripts/build_glossary.py")
    data = tmp_path / "raw"
    monkeypatch.setattr(builder, "DATA", data)
    monkeypatch.setattr(builder, "OUTPUT", tmp_path / "out/glossary.json")
    for relative, records in entries_by_path.items():
        path = data / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records))
    builder.build()
    return json.loads(builder.OUTPUT.read_text())


def test_merge_keeps_urls_attached_to_the_correct_source(monkeypatch, tmp_path):
    entries = merge(monkeypatch, tmp_path, {
        "wikipedia/wiki_medical_terms.json": [{"term": "Hb", "plain": TEXT, "source_url": "https://ru.wikipedia.org/wiki/Hb"}],
        "wikidata/wikidata_medical_terms.json": [{"term": "Protein", "description": TEXT, "source_url": "http://www.wikidata.org/entity/Q123"}],
    })
    assert [(e["source"], e["source_url"]) for e in entries] == [
        ("wikipedia_ru", "https://ru.wikipedia.org/wiki/Hb"),
        ("wikidata", "http://www.wikidata.org/entity/Q123"),
    ]


@pytest.mark.parametrize("invalid_url", [None, "", 17, [], {"url": "wrong shape"}, "javascript:alert(1)"])
def test_invalid_optional_url_does_not_drop_legacy_content(monkeypatch, tmp_path, invalid_url):
    entries = merge(monkeypatch, tmp_path, {
        "wikipedia/wiki_medical_terms.json": [{"term": "Hb", "plain": TEXT, "source_url": invalid_url}],
    })
    assert len(entries) == 1
    assert entries[0]["plain"] == TEXT
    assert "source_url" not in entries[0]


def test_generated_draft_does_not_gain_wikipedia_url_from_duplicate(monkeypatch, tmp_path):
    entries = merge(monkeypatch, tmp_path, {
        "llm_generated/llm_glossary.json": [{"term": "Hb", "plain": TEXT, "source_url": "https://model-invented.invalid/source"}],
        "wikipedia/wiki_medical_terms.json": [{"term": "Hb", "plain": TEXT, "source_url": "https://ru.wikipedia.org/wiki/Hb"}],
    })
    assert len(entries) == 1
    assert entries[0]["source"] == "llm_generated"
    assert "source_url" not in entries[0]


@pytest.mark.parametrize("source_url", ["https://ru.wikipedia.org/wiki/Hb", None, [], "javascript:alert(1)"])
def test_index_preserves_optional_url_without_changing_documents(monkeypatch, tmp_path, source_url):
    # Only the existing Chroma storage boundary is replaced; the index builder is real.
    chroma = types.ModuleType("chromadb")
    config = types.ModuleType("chromadb.config")
    utils = types.ModuleType("chromadb.utils")
    embeddings = types.ModuleType("chromadb.utils.embedding_functions")
    collection = Mock()
    collection.count.return_value = 1
    client = Mock()
    client.get_or_create_collection.return_value = collection
    chroma.PersistentClient = Mock(return_value=client)
    config.Settings = Mock()
    embeddings.SentenceTransformerEmbeddingFunction = Mock()
    for name, module in {"chromadb": chroma, "chromadb.config": config, "chromadb.utils": utils, "chromadb.utils.embedding_functions": embeddings}.items():
        monkeypatch.setitem(sys.modules, name, module)
    index = import_file(monkeypatch, "index_urls_under_test", "backend/rag/build_index.py")
    path = tmp_path / "glossary.json"
    record = {"term": "Hb", "plain": TEXT}
    if source_url is not None:
        record["source_url"] = source_url
    path.write_text(json.dumps([record]))
    monkeypatch.setattr(index, "GLOSSARY_PATH", path)
    assert index.build_index() == 1
    written = collection.upsert.call_args.kwargs
    assert written["ids"] == ["Hb"]
    assert written["documents"] == [TEXT]
    metadata = {"term": "Hb", "category": "", "unit": "", "reference_range": ""}
    if isinstance(source_url, str) and source_url.startswith("https://"):
        metadata["source_url"] = source_url
    assert written["metadatas"] == [metadata]
