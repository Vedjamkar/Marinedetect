"""POST /api/v1/inference must 503 with a precise reason (not crash) when no
YOLO weights are present — this is the hermetic path a demo hits before any
training has happened."""
import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app

client = TestClient(app)


def _fake_png_bytes() -> bytes:
    arr = (np.random.default_rng(0).random((64, 64)) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def test_inference_503_when_yolo_unavailable():
    files = {"image": ("test.png", _fake_png_bytes(), "image/png")}
    resp = client.post("/api/v1/inference", files=files)
    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert detail["mode"] == "unavailable"
    assert detail["trained_model"] is False
    assert "weights not found" in detail["reason"]
