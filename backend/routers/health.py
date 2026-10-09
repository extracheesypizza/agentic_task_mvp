# # backend/routers/health.py
# from fastapi import APIRouter, Request

# from backend.config import settings

# router = APIRouter(tags=["health"])


# @router.get("/health")
# async def health():
#     """Liveness probe — the process is up."""
#     return {"status": "ok"}


# @router.get("/health/ready")
# async def readiness(request: Request):
#     """Readiness probe — process up AND dependencies usable."""
#     from backend.rag.retriever import index_status

#     rag = index_status()
#     return {
#         "status": "ready" if rag["ready"] else "degraded",
#         "rag": rag,
#         "llm_provider": "ollama (local)",
#         "extraction_model": settings.model_extraction,
#         "explanation_model": settings.model_explanation,
#         "embedding_model": settings.embedding_model,
#         "ocr_lang": settings.tesseract_lang,
#     }

from fastapi import APIRouter

from backend.config import settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    """Liveness probe — the process is up."""
    return {"status": "ok"}


@router.get("/health/ready")
async def readiness():
    """Readiness probe — process up AND dependencies usable."""
    from backend.rag.retriever import index_status
    from backend.services.llm_client import warmup_status

    rag = index_status()
    llm = warmup_status()

    return {
        "status": "ready" if (rag["ready"] and llm["all_warm"]) else "degraded",
        "rag": rag,
        "llm": llm,
        "llm_provider": "ollama (local)",
        "extraction_model": settings.model_extraction,
        "explanation_model": settings.model_explanation,
    }