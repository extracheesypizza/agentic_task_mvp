"""Real FastAPI request/response tests; OCR, retrieval and LLM are test doubles."""

import importlib.util
import io
import sys
import types
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from openai import APIConnectionError
from PIL import Image, UnidentifiedImageError


@pytest.fixture
def api(monkeypatch):
    # Avoid importing Chroma/Torch or running the production model-loading lifespan.
    retriever = types.ModuleType("backend.rag.retriever")
    retriever.get_context = Mock(return_value="Synthetic reference")
    with monkeypatch.context() as local:
        local.setitem(sys.modules, "backend.rag.retriever", retriever)
        spec = importlib.util.spec_from_file_location("analyze_under_test", Path(__file__).parents[1] / "backend/routers/upload.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    for name, value in {
        "extract_text_from_image": "Гемоглобин 123 г/л. Синтетический бланк.",
        "extract_structured": {"document_type": "lab_result", "findings": [{"parameter": "Гемоглобин", "value": "123"}], "medications": []},
        "get_context": "Synthetic reference",
        "explain": "Синтетическое объяснение.",
    }.items():
        monkeypatch.setattr(module, name, Mock(return_value=value))
    app = FastAPI()
    app.include_router(module.router)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, module


def upload(client, content=b"synthetic", mime="image/png"):
    return client.post("/api/analyze", files={"file": ("sample.png", content, mime)})


def test_happy_path_preserves_response_contract(api):
    client, module = api
    response = upload(client)
    assert response.status_code == 200
    assert set(response.json()) == {"raw_text", "structured", "explanation"}
    assert response.json()["structured"]["findings"][0]["value"] == "123"
    module.get_context.assert_called_once_with(["Гемоглобин"])
    assert "Disclaimer" in response.json()["explanation"]


def test_rejects_non_image_before_processing(api):
    client, module = api
    assert upload(client, mime="application/pdf").status_code == 400
    module.extract_text_from_image.assert_not_called()


def test_rejects_large_upload_before_processing(api, monkeypatch):
    client, module = api
    monkeypatch.setattr(module.settings, "max_upload_size_mb", 1)
    assert upload(client, b"x" * (1024 * 1024 + 1)).status_code == 413
    module.extract_text_from_image.assert_not_called()


@pytest.mark.parametrize("failure", [ValueError("invalid shape"), ValueError("invalid JSON")])
def test_bad_extraction_returns_502(api, failure):
    client, module = api
    module.extract_structured.side_effect = failure
    assert upload(client).status_code == 502
    module.explain.assert_not_called()


def test_unreadable_extraction_stops_pipeline(api):
    client, module = api
    module.extract_structured.return_value = {"document_type": "unreadable", "findings": [], "medications": []}
    assert upload(client).status_code == 422
    module.get_context.assert_not_called()
    module.explain.assert_not_called()


def test_short_ocr_stops_before_model(api):
    client, module = api
    module.extract_text_from_image.return_value = ""
    assert upload(client).status_code == 422
    module.extract_structured.assert_not_called()


def test_invalid_image_returns_422(api):
    client, module = api
    module.extract_text_from_image.side_effect = UnidentifiedImageError("invalid image")
    assert upload(client).status_code == 422


@pytest.mark.parametrize("stage", ["extract_structured", "explain"])
def test_provider_outage_returns_503(api, stage):
    client, module = api
    getattr(module, stage).side_effect = APIConnectionError(request=httpx.Request("POST", "http://model.invalid"))
    assert upload(client).status_code == 503


def test_retrieval_outage_returns_503_without_generating(api):
    client, module = api
    module.get_context.side_effect = RuntimeError("index unavailable")
    assert upload(client).status_code == 503
    module.explain.assert_not_called()


def test_prescription_without_findings_skips_retrieval(api):
    client, module = api
    module.extract_structured.return_value = {"document_type": "prescription", "findings": [], "medications": [{"name": "as printed"}]}
    assert upload(client).status_code == 200
    module.get_context.assert_not_called()
    module.explain.assert_called_once_with(module.extract_structured.return_value, "")


def test_empty_explanation_returns_502(api):
    client, module = api
    module.explain.return_value = "   "
    assert upload(client).status_code == 502


@pytest.mark.parametrize("content", [None, "", "   "])
def test_real_explainer_rejects_empty_model_response_without_rag(api, monkeypatch, content):
    from backend.services import explainer
    client, module = api
    module.extract_structured.return_value = {"document_type": "prescription", "findings": [], "medications": [{"name": "as printed"}]}
    monkeypatch.setattr(module, "explain", explainer.explain)
    monkeypatch.setattr(explainer, "chat", Mock(return_value=content))
    assert upload(client).status_code == 502


def test_real_ocr_decode_rejects_corrupt_image(api, monkeypatch):
    from backend.services.ocr import extract_text_from_image
    client, module = api
    monkeypatch.setattr(module, "extract_text_from_image", extract_text_from_image)
    assert upload(client, b"not an image").status_code == 422


def test_pipeline_with_real_image_decode_parser_and_explainer(api, monkeypatch):
    """Exercise actual stage boundaries, substituting only OCR engine, RAG and LLM."""
    import json
    import pytesseract
    from backend.services import extractor, explainer, ocr

    client, module = api
    monkeypatch.setattr(module, "extract_text_from_image", ocr.extract_text_from_image)
    monkeypatch.setattr(module, "extract_structured", extractor.extract_structured)
    monkeypatch.setattr(module, "explain", explainer.explain)
    monkeypatch.setattr(pytesseract, "image_to_string", Mock(return_value="Гемоглобин 123,0 г/л. Синтетический документ."))
    monkeypatch.setattr(extractor, "chat", Mock(return_value=json.dumps({
        "document_type": "lab_result", "findings": [{"parameter": "Гемоглобин", "value": "123,0", "unit": "г/л"}],
    })))
    explanation_chat = Mock(return_value="Гемоглобин: 123,0 г/л. Обсудите результат с врачом.")
    monkeypatch.setattr(explainer, "chat", explanation_chat)
    image = io.BytesIO()
    Image.new("RGB", (10, 10), "white").save(image, format="PNG")
    response = upload(client, image.getvalue())
    assert response.status_code == 200
    assert response.json()["structured"]["findings"][0]["value"] == "123,0"
    prompt = explanation_chat.call_args.kwargs["user_prompt"]
    assert "Synthetic reference" in prompt
    assert "123,0" in prompt
    assert "Disclaimer" in response.json()["explanation"]


def test_slow_analysis_does_not_block_other_api_requests(api):
    client, module = api
    started, release = threading.Event(), threading.Event()

    def slow_ocr(*args, **kwargs):
        started.set()
        assert release.wait(5), "test did not release OCR worker"
        return "Гемоглобин 123 г/л. Синтетический документ."

    module.extract_text_from_image.side_effect = slow_ocr

    @client.app.get("/probe")
    async def probe():
        return {"ok": True}

    with ThreadPoolExecutor(max_workers=2) as pool:
        analysis = pool.submit(upload, client)
        try:
            assert started.wait(3)
            response = pool.submit(client.get, "/probe").result(timeout=2)
            assert response.status_code == 200
            assert not analysis.done()
        finally:
            release.set()
        assert analysis.result(timeout=3).status_code == 200
