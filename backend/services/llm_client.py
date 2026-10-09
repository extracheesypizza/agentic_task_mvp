# """
# LLM client — talks to local Ollama via its OpenAI-compatible API.
# No external providers, no rate limits, no content filters.
# """

# from openai import OpenAI

# from backend.config import settings


# def get_client() -> OpenAI:
#     return OpenAI(
#         base_url=settings.ollama_base_url,
#         api_key=settings.ollama_api_key,
#     )


# def chat(
#     system_prompt: str,
#     user_prompt: str,
#     model: str,
#     temperature: float = 0,
#     max_tokens: int = 2048,
#     json_mode: bool = False,
# ) -> str:
#     client = get_client()

#     kwargs = {}
#     if json_mode:
#         kwargs["response_format"] = {"type": "json_object"}

#     response = client.chat.completions.create(
#         model=model,
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt},
#         ],
#         temperature=temperature,
#         max_tokens=max_tokens,
#         **kwargs,
#     )
#     return response.choices[0].message.content

"""
LLM client — talks to local Ollama via its OpenAI-compatible API.

The model is preloaded at application startup (see warmup()) and pinned in
memory with keep_alive=-1, so no user request pays the cold-start cost.
"""

import logging
import threading
import time

from openai import OpenAI

from backend.config import settings

logger = logging.getLogger(__name__)

# keep_alive=-1 tells Ollama not to unload the model after idle timeout.
# Without this the default 5-minute expiry unloads it mid-demo.
EXTRA_BODY = {"keep_alive": -1}

_warm_state: dict[str, bool] = {}
_lock = threading.Lock()


def get_client() -> OpenAI:
    return OpenAI(
        base_url=settings.ollama_base_url,
        api_key=settings.ollama_api_key,
    )


def chat(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float = 0,
    max_tokens: int = 2048,
    json_mode: bool = False,
) -> str:
    client = get_client()

    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=EXTRA_BODY,
        **kwargs,
    )
    return response.choices[0].message.content


# ── Warmup ────────────────────────────────────────────────────────────
def _warm_one(model: str, attempts: int = 12, delay: float = 5.0) -> bool:
    """
    Load one model into memory with a trivial request.

    Retries because warmup runs during startup, when the Ollama container
    may still be initialising.
    """
    client = get_client()

    for attempt in range(attempts):
        try:
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                temperature=0,
                extra_body=EXTRA_BODY,
            )
            logger.info("✓ Model '%s' loaded and pinned in memory.", model)
            return True
        except Exception as e:
            if attempt == attempts - 1:
                logger.warning("✗ Warmup failed for '%s': %s", model, e)
                return False
            logger.info("Ollama not ready for '%s', retrying in %.0fs...", model, delay)
            time.sleep(delay)

    return False


def warmup_all() -> None:
    """Warm every model the app uses. Runs in a background thread."""
    models = list(dict.fromkeys([
        settings.model_extraction,
        settings.model_explanation,
    ]))

    for model in models:
        ok = _warm_one(model)
        with _lock:
            _warm_state[model] = ok


def warmup_status() -> dict:
    """Reported by /health/ready."""
    with _lock:
        snapshot = dict(_warm_state)

    expected = {settings.model_extraction, settings.model_explanation}
    return {
        "models": snapshot,
        "all_warm": bool(snapshot) and all(snapshot.get(m) for m in expected),
    }