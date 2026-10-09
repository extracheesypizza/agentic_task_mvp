# # backend/main.py
# from contextlib import asynccontextmanager

# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware

# from backend.config import settings
# from backend.routers import health, upload


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """Pre-load the embedding model and RAG index before serving requests."""
#     import logging
#     logging.basicConfig(level=logging.INFO)

#     from backend.rag.retriever import index_status

#     # Triggers lazy load + auto-build if the collection is missing
#     app.state.rag = index_status()
#     yield


# app = FastAPI(
#     title="MedTranslate",
#     description="Upload a photo of a lab result and get a plain-language explanation.",
#     version="0.1.0",
#     lifespan=lifespan,
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=settings.cors_origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# app.include_router(health.router)
# app.include_router(upload.router)

from contextlib import asynccontextmanager
import logging
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.routers import health, upload

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load the RAG index and pin the LLM in memory before serving."""
    logging.basicConfig(level=logging.INFO)

    # RAG index (builds itself if the collection is missing)
    from backend.rag.retriever import index_status
    app.state.rag = index_status()

    # LLM warmup — off the event loop, so startup isn't blocked
    from backend.services.llm_client import warmup_all
    thread = threading.Thread(target=warmup_all, name="llm-warmup", daemon=True)
    thread.start()
    app.state.warmup_thread = thread

    logger.info("Startup complete; LLM warmup running in background.")
    yield


app = FastAPI(
    title="MedTranslate",
    description="Upload a photo of a lab result and get a plain-language explanation.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(upload.router)