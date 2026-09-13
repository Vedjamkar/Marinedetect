"""GET /api/v1/model-status must not crash when weights are absent, and must
name the exact reason for each unavailable model."""
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_model_status_with_zero_weights():
    resp = client.get("/api/v1/model-status")
    assert resp.status_code == 200
    body = resp.json()

    assert body["mode"] == "unavailable"
    assert body["trained_model"] is False

    assert body["yolo"]["loaded"] is False
    assert "weights not found" in body["yolo"]["reason"]

    assert body["unet"]["loaded"] is False
    assert "weights not found" in body["unet"]["reason"]

    assert body["device"] in ("cuda", "cpu")
    assert "yolo_confidence" in body["thresholds"]
