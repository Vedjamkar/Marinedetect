"""
The documented promise: the shadow measurement and height calculation work with
no model weights at all. This pins it. If someone reintroduces an early 503 the
"runs without weights" claim in every doc becomes false without anyone noticing.

Hermetic - conftest already points the weight paths at a nonexistent directory.
"""
import json

import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)

GEOM = {"towfish_altitude_m": 12.0, "slant_range_m": 40.0,
        "across_track_resolution_m_per_px": 0.05}


def _synthetic_target_png(true_height_m: float) -> bytes:
    """Flat seabed with speckle, one bright return, one shadow whose length is
    derived from the chosen height by inverting h = H*Ls/(G+Ls)."""
    H, R, res = GEOM["towfish_altitude_m"], GEOM["slant_range_m"], GEOM["across_track_resolution_m_per_px"]
    G = (R**2 - H**2) ** 0.5
    ls_px = int(round(true_height_m * G / (H - true_height_m) / res))
    rng = np.random.default_rng(3)
    img = (120.0 * rng.gamma(25.0, 1 / 25.0, size=(460, 520))).astype(np.float32)
    top, hl = 150, 16
    img[top:top + hl, 200:330] = 238
    img[top + hl:top + hl + ls_px, 205:325] = 24
    img = cv2.GaussianBlur(np.clip(img, 0, 255).astype(np.uint8), (3, 3), 0)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_full_frame_height_works_with_no_detector_loaded():
    resp = client.post(
        "/api/v1/inference",
        files={"image": ("t.png", _synthetic_target_png(1.60), "image/png")},
        data={"geometry": json.dumps(GEOM), "analyze_full_frame": "true"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # Honest about the missing detector...
    assert body["detections"] == []
    assert body["mode"] == "unavailable"
    assert "No detector loaded" in body["full_frame"]["note"]

    # ...and the physics still delivers.
    h = body["full_frame"]["height"]
    assert h["available"], h
    assert abs(h["height_m"] - 1.60) / 1.60 < 0.05, h["height_m"]
    assert h["method"] == "classical_profile"


def test_plain_inference_without_full_frame_still_503s_with_no_detector():
    """Without the model-free path requested, no detector is a real failure
    and must say so rather than returning an empty success."""
    resp = client.post(
        "/api/v1/inference",
        files={"image": ("t.png", _synthetic_target_png(1.60), "image/png")},
        data={"geometry": json.dumps(GEOM)},
    )
    assert resp.status_code == 503
    assert "weights not found" in resp.json()["detail"]["reason"]
