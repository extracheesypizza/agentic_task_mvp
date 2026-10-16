import sys
from pathlib import Path
from types import SimpleNamespace

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
