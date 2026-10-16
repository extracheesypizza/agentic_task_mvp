"""Real retrieval formatting with a fake Chroma collection; no model or index writes."""

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture
def retrieval(monkeypatch):
    chroma = types.ModuleType("chromadb")
    chroma.api = types.SimpleNamespace(ClientAPI=object)
    config = types.ModuleType("chromadb.config")
    config.Settings = Mock()
    embeddings = types.ModuleType("chromadb.utils.embedding_functions")
    embeddings.SentenceTransformerEmbeddingFunction = Mock()
    for name, module in {
        "chromadb": chroma,
        "chromadb.config": config,
        "chromadb.utils": types.ModuleType("chromadb.utils"),
        "chromadb.utils.embedding_functions": embeddings,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location(
        "materials_retriever_under_test", Path(__file__).parents[1] / "backend/rag/retriever.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    collection = Mock()
    collection.count.return_value = 2
    monkeypatch.setattr(module, "_get_collection", Mock(return_value=collection))
    return module, collection


def test_same_query_and_order_produce_unchanged_context_and_matching_materials(retrieval):
    module, collection = retrieval
    collection.query.return_value = {
        "documents": [["Hb definition", "Protein definition"], ["duplicate Hb"]],
        "metadatas": [[
            {"term": "Hb", "unit": "g/l", "reference_range": "printed range", "source": "wikipedia_ru", "source_url": "https://example.org/Hb"},
            {"term": "Protein"},
        ], [{"term": "Hb", "source_url": "https://example.org/duplicate"}]],
    }
    context, materials = module.get_context_with_materials(["Hb", "Protein"])
    collection.query.assert_called_once_with(query_texts=["Hb", "Protein"], n_results=2)
    assert context == "**Hb**: Hb definition\n  - Единицы измерения: g/l\n  - Референсные значения: printed range\n\n**Protein**: Protein definition"
    assert materials == [
        {"term": "Hb", "source": "wikipedia_ru", "source_url": "https://example.org/Hb"},
        {"term": "Protein", "source": None, "source_url": None},
    ]


@pytest.mark.parametrize("url", [None, "", 7, "javascript:alert(1)", "https://", "https://[broken", "https://example.org:bad"])
def test_invalid_optional_url_does_not_remove_context(retrieval, url):
    module, collection = retrieval
    collection.query.return_value = {"documents": [["definition"]], "metadatas": [[{"term": "Hb", "source_url": url}]]}
    context, materials = module.get_context_with_materials(["Hb"], top_k=1)
    assert context == "**Hb**: definition"
    assert materials == [{"term": "Hb", "source": None, "source_url": None}]
    collection.query.assert_called_once_with(query_texts=["Hb"], n_results=1)


def test_empty_terms_never_open_collection(retrieval):
    module, collection = retrieval
    assert module.get_context_with_materials([]) == ("", [])
    module._get_collection.assert_not_called()


def test_empty_collection_never_queries(retrieval):
    module, collection = retrieval
    collection.count.return_value = 0
    assert module.get_context_with_materials(["Hb"]) == ("", [])
    collection.query.assert_not_called()


def test_legacy_get_context_still_returns_only_text(retrieval):
    module, collection = retrieval
    collection.query.return_value = {"documents": [["definition"]], "metadatas": [[{"term": "Hb"}]]}
    assert module.get_context(["Hb"]) == "**Hb**: definition"
    collection.query.assert_called_once()


def test_no_matches_return_empty_materials_without_changing_query_limit(retrieval):
    module, collection = retrieval
    collection.count.return_value = 10
    collection.query.return_value = {"documents": [[]], "metadatas": [[]]}
    assert module.get_context_with_materials(["missing"]) == ("", [])
    collection.query.assert_called_once_with(query_texts=["missing"], n_results=5)


@pytest.mark.parametrize("url", ["http://example.org/Hb", "https://example.org/Hb"])
def test_optional_http_links_remain_attached_to_their_terms(retrieval, url):
    module, collection = retrieval
    collection.query.return_value = {
        "documents": [["definition"]],
        "metadatas": [[{"term": "Hb", "source": "", "source_url": url}]],
    }
    _, materials = module.get_context_with_materials(["Hb"])
    assert materials == [{"term": "Hb", "source": None, "source_url": url}]
