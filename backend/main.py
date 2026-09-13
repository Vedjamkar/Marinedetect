"""
FastAPI app factory / entrypoint.

Boots clean with zero model weights on disk — /health and
/api/v1/model-status must always work; /api/v1/inference reports 503 with a
precise reason instead of crashing when no YOLO weights are present.

Run with:
    python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
(run from the repository root so `backend` resolves as a package)
"""
import logging
from pathlib import Path

import contextlib

import numpy as np
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.api import health, inference, status
from backend.config import settings

logging.basicConfig(level=settings.log_level)

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the detector at startup.

    Model load is lazy-once, so without this the FIRST request pays ~2.7s of
    load time. That is a bad surprise in a live demo, so we absorb it here.
    Failure is non-fatal: the service is designed to run without weights and
    model-status reports the reason.
    """
    try:
        from backend.services.yolo_service import yolo_service

        if yolo_service.status()["loaded"]:
            yolo_service.predict(np.zeros((64, 64, 3), dtype=np.uint8))
            logging.getLogger(__name__).info("detector warmed at startup")
    except Exception:
        logging.getLogger(__name__).warning("detector warm-up skipped", exc_info=True)
    yield


app = FastAPI(title="Marinedetect Inference API", version="0.1.0", lifespan=lifespan)

app.include_router(health.router)
app.include_router(status.router)
app.include_router(inference.router)

# Server-generated UUID filenames only (never user-derived) — see
# api/inference.py. Mounted at /outputs, served as static files.
app.mount("/outputs", StaticFiles(directory=str(settings.outputs_dir)), name="outputs")

# Demo UI. Served at / so the whole thing is one URL to open on a projector.
_static_dir = Path(__file__).resolve().parent / "static"
if _static_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="demo")
