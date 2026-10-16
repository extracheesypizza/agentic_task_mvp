"""Offline preparation tests. All writes stay in pytest's temporary directory."""

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[1]


def import_script(monkeypatch, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def scripts(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    # Lets the old eager client import so the tests can expose its behavior.
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic-test-key")
    generator = import_script(monkeypatch, "generate_glossary_llm")
    builder = import_script(monkeypatch, "build_glossary")
    monkeypatch.delenv("OPENROUTER_API_KEY")
    monkeypatch.delenv("GLOSSARY_MODEL", raising=False)
    return generator, builder


def test_import_does_not_create_output_directories(scripts, tmp_path):
    assert list(tmp_path.iterdir()) == []


def test_import_without_credentials_does_not_construct_api_client(monkeypatch, tmp_path):
    import openai
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    factory = Mock(side_effect=AssertionError("API client must not be created during import"))
    monkeypatch.setattr(openai, "OpenAI", factory)
    import_script(monkeypatch, "generate_glossary_llm")
    factory.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def test_generator_and_builder_agree_on_output_from_any_working_directory(scripts):
    generator, builder = scripts
    assert generator.OUTPUT / "llm_glossary.json" == builder.DATA / "llm_generated/llm_glossary.json"
    assert generator.OUTPUT.is_absolute()
    assert builder.OUTPUT.is_absolute()


def test_app_model_is_not_used_as_generator_model(scripts, monkeypatch, tmp_path):
    generator, _ = scripts
    monkeypatch.setattr(generator, "ROOT", tmp_path, raising=False)
    monkeypatch.setenv("MODEL_EXPLANATION", "medgemma:4b")
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic-test-key")
    assert generator.load_config()["model"] == "mistralai/mistral-7b-instruct:free"


def test_config_requires_key_before_client_or_generation(scripts, monkeypatch, tmp_path, capsys):
    generator, _ = scripts
    monkeypatch.setattr(generator, "ROOT", tmp_path, raising=False)
    monkeypatch.setenv("GLOSSARY_MODEL", "provider/synthetic-model")
    client_factory = Mock()
    monkeypatch.setattr(generator, "OpenAI", client_factory)
    assert generator.main() == 1
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err
    client_factory.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def test_generator_local_env_does_not_load_app_env(scripts, monkeypatch, tmp_path):
    generator, _ = scripts
    monkeypatch.setattr(generator, "ROOT", tmp_path, raising=False)
    (tmp_path / ".env").write_text("MODEL_EXPLANATION=local-only\nGLOSSARY_MODEL=wrong-app-config\n")
    (tmp_path / ".env.glossary.local").write_text("GLOSSARY_MODEL=provider/from-file\nOPENROUTER_API_KEY=synthetic-test-key\n")
    config = generator.load_config()
    assert config["model"] == "provider/from-file"
    assert "GLOSSARY_MODEL" not in os.environ
    monkeypatch.setenv("GLOSSARY_MODEL", "provider/from-process")
    assert generator.load_config()["model"] == "provider/from-process"


@pytest.mark.parametrize("payload", [None, "", "[]", "null", "42", '[{"plain":"text"}]', '{"term":123,"plain":"text"}', '{"term":"Hb","plain":[]}', '{"term":"Hb","plain":"   "}'])
def test_parser_rejects_invalid_entries(scripts, payload):
    generator, _ = scripts
    assert generator.parse_json_response(payload) is None


@pytest.mark.parametrize("wrapper", ["{}", "```json\n{}\n```", "Result: {}\nDone."])
def test_parser_keeps_valid_model_content(scripts, wrapper):
    generator, _ = scripts
    entry = {"term": "Гемоглобин", "plain": "Синтетическое объяснение для теста."}
    assert generator.parse_json_response(wrapper.format(json.dumps(entry))) == entry


def test_generation_keeps_primary_then_fallback_order_and_skips_wrong_term(scripts, monkeypatch):
    generator, _ = scripts
    monkeypatch.setattr(generator, "FALLBACK_MODELS", ["provider/fallback:free"])
    call = Mock(side_effect=[json.dumps({"term": "other", "plain": "Long synthetic explanation for another term."}), json.dumps({"term": "Hb", "plain": "Long synthetic explanation for the requested term."})])
    monkeypatch.setattr(generator, "call_llm", call)
    monkeypatch.setattr(generator.time, "sleep", Mock())
    client = Mock()
    result = generator.generate_explanation("Hb", "provider/selected", client, max_retries=2)
    assert result["term"] == "Hb"
    assert call.call_count == 2
    assert [c.args[1] for c in call.call_args_list] == ["provider/selected", "provider/fallback:free"]


def test_generation_retry_budget_is_bounded_when_all_models_fail(scripts, monkeypatch):
    generator, _ = scripts
    monkeypatch.setattr(generator, "FALLBACK_MODELS", ["provider/fallback:free"])
    call = Mock(return_value=None)
    monkeypatch.setattr(generator, "call_llm", call)
    monkeypatch.setattr(generator.time, "sleep", Mock())
    assert generator.generate_explanation("Hb", "provider/primary:free", Mock(), max_retries=2) is None
    assert [c.args[1] for c in call.call_args_list] == ["provider/primary:free", "provider/fallback:free"] * 2


def write_source(data, relative_path, entries):
    path = data / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, ensure_ascii=False))


def test_merge_skips_bad_records_without_shadowing_valid_lower_priority_sources(scripts, monkeypatch, tmp_path):
    _, builder = scripts
    data, output = tmp_path / "raw", tmp_path / "out/glossary.json"
    monkeypatch.setattr(builder, "DATA", data)
    monkeypatch.setattr(builder, "OUTPUT", output)
    write_source(data, "llm_generated/llm_glossary.json", [
        None, "wrong", {"term": "Hb", "plain": "short"},
        {"term": 123, "plain": "A sufficiently long synthetic explanation."},
        {"term": "bad", "plain": ["wrong type"]},
        {"term": "Glucose", "plain": "A sufficiently long synthetic generated explanation."},
    ])
    write_source(data, "wikipedia/wiki_medical_terms.json", [
        {"term": "Hb", "plain": "A sufficiently long synthetic Wikipedia explanation."},
        {"term": "Glucose", "plain": "A sufficiently long lower priority explanation."},
    ])
    write_source(data, "wikidata/wikidata_medical_terms.json", [
        {"term": "Other", "description": "short"},
        {"term": "Protein", "description": "A sufficiently long synthetic Wikidata description."},
    ])
    builder.build()
    entries = json.loads(output.read_text())
    assert [(e["term"], e["source"]) for e in entries] == [
        ("Glucose", "llm_generated"), ("Hb", "wikipedia_ru"), ("Protein", "wikidata"),
    ]


def test_malformed_source_does_not_overwrite_existing_output(scripts, monkeypatch, tmp_path):
    _, builder = scripts
    data, output = tmp_path / "raw", tmp_path / "glossary.json"
    monkeypatch.setattr(builder, "DATA", data)
    monkeypatch.setattr(builder, "OUTPUT", output)
    output.write_text("existing corpus")
    write_source(data, "llm_generated/llm_glossary.json", {"wrong": "container"})
    with pytest.raises(ValueError):
        builder.build()
    assert output.read_text() == "existing corpus"


def test_empty_merge_does_not_erase_existing_corpus(scripts, monkeypatch, tmp_path):
    _, builder = scripts
    output = tmp_path / "glossary.json"
    monkeypatch.setattr(builder, "DATA", tmp_path / "absent-sources")
    monkeypatch.setattr(builder, "OUTPUT", output)
    output.write_text("existing corpus")
    with pytest.raises(ValueError, match="No usable"):
        builder.build()
    assert output.read_text() == "existing corpus"


def test_all_failed_generation_does_not_erase_previous_drafts(scripts, monkeypatch, tmp_path):
    generator, _ = scripts
    output = tmp_path / "generated"
    output.mkdir()
    previous = output / "llm_glossary.json"
    previous.write_text("existing drafts")
    monkeypatch.setattr(generator, "OUTPUT", output)
    monkeypatch.setattr(generator, "load_config", lambda: {"api_key": "synthetic", "model": "provider/synthetic", "base_url": "https://model.invalid/v1"})
    monkeypatch.setattr(generator, "load_terms", lambda: ["Hb"])
    monkeypatch.setattr(generator, "generate_explanation", Mock(return_value=None))
    monkeypatch.setattr(generator.time, "sleep", Mock())
    monkeypatch.setattr(generator, "OpenAI", Mock(return_value=Mock()))
    assert generator.main() == 1
    assert previous.read_text() == "existing drafts"


def test_generated_output_feeds_merge_using_only_temporary_fixtures(scripts, monkeypatch, tmp_path):
    generator, builder = scripts
    data = tmp_path / "raw"
    monkeypatch.setattr(generator, "OUTPUT", data / "llm_generated")
    monkeypatch.setattr(builder, "DATA", data)
    monkeypatch.setattr(builder, "OUTPUT", tmp_path / "app/glossary.json")
    monkeypatch.setattr(generator, "load_config", lambda: {"api_key": "synthetic", "model": "provider/synthetic", "base_url": "https://model.invalid/v1"})
    monkeypatch.setattr(generator, "load_terms", lambda: ["Hb"])
    client = Mock()
    client.chat.completions.create.return_value.choices = [
        SimpleNamespace(message=SimpleNamespace(content=json.dumps({"term": "Hb", "plain": "A synthetic definition long enough for this test."})))
    ]
    monkeypatch.setattr(generator, "OpenAI", Mock(return_value=client))
    monkeypatch.setattr(generator.time, "sleep", Mock())
    assert generator.main() == 0
    builder.build()
    records = json.loads(builder.OUTPUT.read_text())
    assert records[0]["term"] == "Hb"
    assert records[0]["source"] == "llm_generated"
    assert client.chat.completions.create.call_args.kwargs["model"] == "provider/synthetic"
    client.close.assert_called_once()


def test_client_is_closed_when_local_input_loading_fails(scripts, monkeypatch):
    generator, _ = scripts
    monkeypatch.setattr(generator, "load_config", lambda: {"api_key": "synthetic", "model": "provider/synthetic", "base_url": "https://model.invalid/v1"})
    monkeypatch.setattr(generator, "load_terms", Mock(side_effect=ValueError("bad input file")))
    client = Mock()
    monkeypatch.setattr(generator, "OpenAI", Mock(return_value=client))
    with pytest.raises(ValueError, match="bad input file"):
        generator.main()
    client.close.assert_called_once()
