"""Manual LLM command checks, with no real model requests."""

import importlib.util
import runpy
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError

from backend.services import llm_client


SCRIPT = Path(__file__).resolve().parents[1] / "backend/test.py"


def load_smoke(monkeypatch):
    chat = Mock(return_value="Synthetic model response")
    monkeypatch.setattr(llm_client, "chat", chat)
    spec = importlib.util.spec_from_file_location("llm_smoke_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        pytest.fail(f"Manual check must import through the backend package: {exc}")
    return module, chat


@pytest.fixture
def smoke(monkeypatch):
    return load_smoke(monkeypatch)


def test_import_does_not_call_model(monkeypatch):
    _, chat = load_smoke(monkeypatch)
    chat.assert_not_called()


def test_success_uses_configured_explanation_model(smoke, monkeypatch, capsys):
    module, chat = smoke
    monkeypatch.setattr(module.settings, "model_explanation", "synthetic-smoke-model")
    assert module.main() == 0
    chat.assert_called_once()
    assert chat.call_args.kwargs["model"] == "synthetic-smoke-model"
    captured = capsys.readouterr()
    assert captured.out.strip() == "Synthetic model response"
    assert captured.err == ""


@pytest.mark.parametrize("failure", [
    APIConnectionError(request=httpx.Request("POST", "http://model.invalid")),
    APITimeoutError(request=httpx.Request("POST", "http://model.invalid")),
    APIStatusError("model unavailable", response=httpx.Response(404, request=httpx.Request("POST", "http://model.invalid")), body=None),
])
def test_service_failure_is_clear_and_returns_nonzero(smoke, capsys, failure):
    module, chat = smoke
    chat.side_effect = failure
    assert module.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Ollama" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("content", [None, "", "   "])
def test_empty_response_is_not_reported_as_success(smoke, capsys, content):
    module, chat = smoke
    chat.return_value = content
    assert module.main() == 1
    assert "empty response" in capsys.readouterr().err


@pytest.mark.parametrize("success", [True, False])
def test_module_entrypoint_sets_process_exit_code(monkeypatch, success):
    chat = Mock(return_value="Synthetic response")
    if not success:
        chat.side_effect = APIConnectionError(request=httpx.Request("POST", "http://model.invalid"))
    monkeypatch.setattr(llm_client, "chat", chat)
    with pytest.raises(SystemExit) as exited:
        runpy.run_module("backend.test", run_name="__main__")
    assert exited.value.code == (0 if success else 1)
    chat.assert_called_once()
