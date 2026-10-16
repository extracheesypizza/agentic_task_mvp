from pathlib import Path

from dotenv import dotenv_values

from backend.config import Settings


def test_env_example_covers_all_settings_and_parses():
    path = Path(__file__).parents[1] / ".env.example"
    assert {key.lower() for key in dotenv_values(path)} == set(Settings.model_fields)
    settings = Settings(_env_file=path)
    assert settings.max_upload_size_mb > 0
    assert isinstance(settings.cors_origins, list)


def test_example_selects_the_model_compose_downloads():
    path = Path(__file__).parents[1] / ".env.example"
    values = dotenv_values(path)
    assert values["MODEL_EXTRACTION"] == "medgemma:4b"
    assert values["MODEL_EXPLANATION"] == "medgemma:4b"
