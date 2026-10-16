from unittest.mock import Mock

import pytest

from backend.services import explainer


@pytest.mark.parametrize("rag_context", ["", "Synthetic reference"])
@pytest.mark.parametrize("content", [None, "", "   ", 123])
def test_empty_or_wrong_type_model_response_is_rejected_before_fallback(monkeypatch, rag_context, content):
    monkeypatch.setattr(explainer, "chat", Mock(return_value=content))
    with pytest.raises(ValueError):
        explainer.explain({"findings": []}, rag_context)


def test_valid_explanation_without_context_keeps_warning(monkeypatch):
    monkeypatch.setattr(explainer, "chat", Mock(return_value="Синтетическое объяснение."))
    result = explainer.explain({"findings": []}, "")
    assert result.startswith(explainer.NO_CONTEXT_FALLBACK)
    assert result.endswith("Синтетическое объяснение.")
