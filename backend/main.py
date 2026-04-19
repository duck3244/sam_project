"""FastAPI entrypoint.

Run:
    uvicorn backend.main:app --reload --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend import config
from backend.routers import export as export_router
from backend.routers import pipeline as pipeline_router
from backend.routers import session as session_router


config.ensure_dirs()

app = FastAPI(title="SAM Auto-Label API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(session_router.router)
app.include_router(pipeline_router.router)
app.include_router(export_router.router)


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "domain": config.DOMAIN_NAME,
        "classes": config.DOMAIN_CLASSES,
        "sam_models": list(config.SAM_CHECKPOINTS.keys()),
    }
