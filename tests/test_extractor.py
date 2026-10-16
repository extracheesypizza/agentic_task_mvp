"""Model-output regressions, using the real parser and no model server."""

import json
from unittest.mock import Mock

import httpx
import pytest
from openai import APIConnectionError

from backend.services import extractor


VALID = {
    "document_type": "lab_result",
    "findings": [{"parameter": "Гемоглобин", "value": "123,0", "unit": "г/л"}],
    "medications": [],
}


def reply(monkeypatch, payload):
    chat = Mock(return_value=payload)
    monkeypatch.setattr(extractor, "chat", chat)
    return chat


@pytest.mark.parametrize("wrapper", ["{}", "```json\n{}\n```", "Result: {}\nDone."])
def test_valid_extraction_preserves_document_values(monkeypatch, wrapper):
    reply(monkeypatch, wrapper.format(json.dumps(VALID)))
    result = extractor.extract_structured("synthetic OCR")
    assert result["findings"][0]["value"] == "123,0"
    assert result["findings"][0]["unit"] == "г/л"


@pytest.mark.parametrize("payload", [
    [], None, 42,
    {"document_type": "invented", "findings": [{"parameter": "Hb"}]},
    {"document_type": "lab_result", "findings": "wrong"},
    {"document_type": "lab_result", "findings": ["wrong"]},
    {"document_type": "lab_result", "findings": [{"parameter": ["Hb"]}]},
    {"document_type": "lab_result", "findings": [{"parameter": "Hb", "status": "diagnosed"}]},
    {"document_type": "prescription", "medications": "wrong"},
    {"document_type": "lab_result", "findings": [], "medications": []},
    {**VALID, "patient_info": "Alice"},
    {**VALID, "raw_notes": ["note"]},
])
def test_invalid_shapes_retry_then_raise_value_error(monkeypatch, payload):
    chat = reply(monkeypatch, json.dumps(payload))
    with pytest.raises(ValueError):
        extractor.extract_structured("synthetic OCR", max_retries=2)
    assert chat.call_count == 2


def test_unreadable_is_a_valid_terminal_result(monkeypatch):
    payload = {"document_type": "unreadable", "findings": [], "medications": []}
    chat = reply(monkeypatch, json.dumps(payload))
    assert extractor.extract_structured("garbled OCR")["document_type"] == "unreadable"
    chat.assert_called_once()


def test_prescription_normalizes_null_findings(monkeypatch):
    reply(monkeypatch, json.dumps({"document_type": "prescription", "findings": None,
                                  "medications": [{"name": "as printed", "dosage": None}]}))
    result = extractor.extract_structured("synthetic OCR")
    assert result["findings"] == []
    assert result["medications"][0]["dosage"] is None


def test_bad_response_can_recover_on_next_attempt(monkeypatch):
    chat = Mock(side_effect=["not json", json.dumps(VALID)])
    monkeypatch.setattr(extractor, "chat", chat)
    assert extractor.extract_structured("synthetic OCR")["document_type"] == "lab_result"
    assert chat.call_count == 2


def test_provider_failure_is_distinct_from_bad_data(monkeypatch):
    chat = Mock(side_effect=APIConnectionError(request=httpx.Request("POST", "http://model.invalid")))
    monkeypatch.setattr(extractor, "chat", chat)
    with pytest.raises(RuntimeError):
        extractor.extract_structured("synthetic OCR", max_retries=2)
    assert chat.call_count == 2


@pytest.mark.parametrize("payload", [None, "", "   ", "[]", "null"])
def test_empty_or_non_object_model_content_is_validation_error(monkeypatch, payload):
    reply(monkeypatch, payload)
    with pytest.raises(ValueError):
        extractor.extract_structured("synthetic OCR", max_retries=1)


@pytest.mark.parametrize("value", [123, 123.5, "123,0", None])
def test_numeric_and_missing_values_are_not_rewritten(monkeypatch, value):
    payload = {**VALID, "findings": [{"parameter": "Hb", "value": value}]}
    reply(monkeypatch, json.dumps(payload))
    result = extractor.extract_structured("synthetic OCR")
    assert result["findings"][0]["value"] == value
    assert type(result["findings"][0]["value"]) is type(value)


def test_additional_document_metadata_is_preserved(monkeypatch):
    payload = {**VALID, "patient_info": {"date": "2026-10-16"}, "raw_notes": "as printed"}
    reply(monkeypatch, json.dumps(payload))
    result = extractor.extract_structured("synthetic OCR")
    assert result["patient_info"] == payload["patient_info"]
    assert result["raw_notes"] == payload["raw_notes"]


@pytest.mark.parametrize("status", ["critical", "unreadable"])
def test_statuses_already_supported_by_original_frontend_are_preserved(monkeypatch, status):
    payload = {**VALID, "findings": [{"parameter": "Hb", "value": "123", "status": status}]}
    chat = reply(monkeypatch, json.dumps(payload))
    assert extractor.extract_structured("synthetic OCR")["findings"][0]["status"] == status
    chat.assert_called_once()
