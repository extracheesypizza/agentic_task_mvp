"""Explicit manual model check: python -m backend.test (makes one chat call)."""

import logging
import sys

from openai import OpenAIError

from backend.config import settings
from backend.services.llm_client import chat


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    try:
        result = chat(
            system_prompt="You are a helpful assistant.",
            user_prompt="Что такое гемоглобин?",
            model=settings.model_explanation,
        )
    except (OpenAIError, RuntimeError) as exc:
        print(
            f"LLM check failed ({type(exc).__name__}). "
            "Check that Ollama is running and the configured MODEL_EXPLANATION is available.",
            file=sys.stderr,
        )
        return 1

    if not isinstance(result, str) or not result.strip():
        print("LLM returned an empty response. Check the configured model.", file=sys.stderr)
        return 1

    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
