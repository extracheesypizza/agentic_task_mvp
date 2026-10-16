import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
sys.path.insert(0, str(FRONTEND))


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "explanation": "Анализ готов",
            "structured": {"document_type": "lab_result", "findings": []},
        }


class MaterialsResponse(FakeResponse):
    def json(self):
        return {
            "explanation": "Анализ готов",
            "structured": {"document_type": "lab_result", "findings": []},
            "retrieved_materials": [
                {
                    "term": "<script>alert(1)</script>",
                    "source": "Минздрав",
                    "source_url": "https://example.org/material",
                }
            ],
        }


def test_analysis_result_survives_raw_json_toggle(monkeypatch):
    import requests
    import components.upload_card as upload_card

    uploaded = SimpleNamespace(
        name="result.png",
        type="image/png",
        size=4,
        getvalue=lambda: b"image",
    )
    monkeypatch.setattr(upload_card, "render_upload", lambda: uploaded)
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: FakeResponse())

    app = AppTest.from_file(str(FRONTEND / "app.py"), default_timeout=10).run()
    assert not app.exception
    app.button[0].click().run()

    assert "Анализ готов" in [element.value for element in app.markdown]
    app.expander[0].toggle[0].set_value(True).run()

    assert "Анализ готов" in [element.value for element in app.markdown]


def test_new_upload_clears_previous_analysis(monkeypatch):
    import requests
    import components.upload_card as upload_card

    def make_upload(content):
        return SimpleNamespace(
            name="result.png",
            type="image/png",
            size=len(content),
            getvalue=lambda: content,
        )

    current_upload = [make_upload(b"first image")]
    monkeypatch.setattr(upload_card, "render_upload", lambda: current_upload[0])
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: FakeResponse())

    app = AppTest.from_file(str(FRONTEND / "app.py"), default_timeout=10).run()
    app.button[0].click().run()
    assert "Анализ готов" in [element.value for element in app.markdown]

    current_upload[0] = make_upload(b"different image")
    app.run()

    assert "Анализ готов" not in [element.value for element in app.markdown]


def test_invalid_analysis_response_shows_friendly_error(monkeypatch):
    import requests
    import components.upload_card as upload_card

    uploaded = SimpleNamespace(
        name="result.png",
        type="image/png",
        size=4,
        getvalue=lambda: b"image",
    )

    class InvalidResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return ["unexpected", "shape"]

    current_response = [FakeResponse()]
    monkeypatch.setattr(upload_card, "render_upload", lambda: uploaded)
    monkeypatch.setattr(
        requests, "post", lambda *args, **kwargs: current_response[0]
    )

    app = AppTest.from_file(str(FRONTEND / "app.py"), default_timeout=10).run()
    app.button[0].click().run()
    assert "Анализ готов" in [element.value for element in app.markdown]

    current_response[0] = InvalidResponse()
    app.button[0].click().run()

    assert not app.exception
    assert "Анализ готов" not in [element.value for element in app.markdown]
    assert any(
        "Сервер вернул некорректный ответ" in element.value for element in app.error
    )


def test_retrieved_materials_are_shown_as_literal_text_with_safe_link(monkeypatch):
    import requests
    import components.upload_card as upload_card

    uploaded = SimpleNamespace(
        name="result.png", type="image/png", size=4, getvalue=lambda: b"image"
    )
    monkeypatch.setattr(upload_card, "render_upload", lambda: uploaded)
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: MaterialsResponse())

    app = AppTest.from_file(str(FRONTEND / "app.py"), default_timeout=10).run()
    app.button[0].click().run()
    assert not app.exception

    assert [item.label for item in app.expander] == [
        "📋 Структурированные данные",
        "📚 Найденные справочные материалы",
    ]
    assert "<script>alert(1)</script>" in [element.value for element in app.text]
    assert "Минздрав" in [element.value for element in app.text]
    assert any("не являются цитатами" in item.value for item in app.caption)
    links = app.get("link_button")
    assert len(links) == 1
    assert links[0].proto.url == "https://example.org/material"


@pytest.mark.parametrize(
    "source_url, include_url",
    [
        (None, True),
        (None, False),
        ("javascript:alert(1)", True),
        ("https://[broken", True),
        ("https://example.org:bad", True),
    ],
)
def test_material_without_safe_url_has_source_and_no_link(monkeypatch, source_url, include_url):
    import requests
    import components.upload_card as upload_card

    class NoUrlResponse(FakeResponse):
        def json(self):
            material = {"term": "Гемоглобин", "source": None}
            if include_url:
                material["source_url"] = source_url
            return {
                "explanation": "Готово",
                "structured": {},
                "retrieved_materials": [material],
            }

    uploaded = SimpleNamespace(
        name="result.png", type="image/png", size=4, getvalue=lambda: b"image"
    )
    monkeypatch.setattr(upload_card, "render_upload", lambda: uploaded)
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: NoUrlResponse())
    app = AppTest.from_file(str(FRONTEND / "app.py"), default_timeout=10).run()
    app.button[0].click().run()
    assert not app.exception

    assert "Источник не указан" in [element.value for element in app.text]
    assert "Ссылка не указана" in [element.value for element in app.text]
    assert not app.get("link_button")


def test_material_expander_hidden_when_missing_or_empty_and_survives_rerun(monkeypatch):
    import requests
    import components.upload_card as upload_card

    uploaded = SimpleNamespace(
        name="result.png", type="image/png", size=4, getvalue=lambda: b"image"
    )
    responses = [MaterialsResponse()]
    monkeypatch.setattr(upload_card, "render_upload", lambda: uploaded)
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: responses[0])

    app = AppTest.from_file(str(FRONTEND / "app.py"), default_timeout=10).run()
    app.button[0].click().run()
    assert not app.exception
    assert len(app.expander) == 2
    assert "<script>alert(1)</script>" in [element.value for element in app.text]
    app.run()
    assert not app.exception
    assert "<script>alert(1)</script>" in [element.value for element in app.text]

    class EmptyResponse(FakeResponse):
        def json(self):
            return {"explanation": "Готово", "structured": {}, "retrieved_materials": []}

    responses[0] = EmptyResponse()
    app.button[0].click().run()
    assert not app.exception
    assert len(app.expander) == 1

    class MissingResponse(FakeResponse):
        def json(self):
            return {"explanation": "Готово", "structured": {}}

    responses[0] = MissingResponse()
    app.button[0].click().run()
    assert not app.exception
    assert len(app.expander) == 1
