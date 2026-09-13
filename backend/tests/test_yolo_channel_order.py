"""
Guard against the RGB/BGR channel-order regression.

Ultralytics follows the OpenCV convention: an ndarray `source` is interpreted as
BGR. Our pipeline carries RGB. Passing RGB straight through silently swaps red
and blue, and the detector misses targets it would otherwise find at high
confidence — the same frame scored 0 detections as RGB and 1 at 0.93 as BGR.

Nothing raises, nothing logs, and the API keeps returning 200 with an empty
detections list, so it reads as "the model is just not very good". This test
exists so that failure mode cannot come back unnoticed.

Hermetic: uses a fake model that records what it was handed. No weights needed.
"""
import numpy as np

from backend.services import yolo_service as ys


class _RecordingModel:
    """Stands in for an ultralytics YOLO model and captures the array it gets."""

    def __init__(self):
        self.received = None
        self.names = {0: "ship"}

    def predict(self, source, **kwargs):
        self.received = np.array(source, copy=True)
        return []


def test_detector_is_handed_bgr_not_rgb(monkeypatch):
    svc = ys.YoloService() if hasattr(ys, "YoloService") else ys.yolo_service
    fake = _RecordingModel()

    monkeypatch.setattr(svc, "_model", fake, raising=False)
    monkeypatch.setattr(svc, "_ensure_loaded", lambda: None, raising=False)
    monkeypatch.setattr(svc, "_load_error", None, raising=False)

    # An image where the three channels are unmistakably different, so a swap
    # cannot hide behind a greyscale frame with R == G == B.
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[..., 0] = 200   # R
    rgb[..., 1] = 100   # G
    rgb[..., 2] = 10    # B

    svc.predict(rgb)

    assert fake.received is not None, "the model was never called"
    got = fake.received

    # What ultralytics receives must be BGR: blue first, red last.
    assert got[0, 0, 0] == 10, (
        f"first channel should be BLUE (10) but was {got[0, 0, 0]} — "
        "RGB is being passed to ultralytics, which reads ndarrays as BGR"
    )
    assert got[0, 0, 1] == 100, "green channel should be unchanged"
    assert got[0, 0, 2] == 200, (
        f"third channel should be RED (200) but was {got[0, 0, 2]} — channel order is wrong"
    )


def test_conversion_is_a_pure_swap_not_a_copy_of_one_channel():
    """A greyscale-looking result would also pass a naive check, so assert the
    round trip really is a reversal."""
    rgb = np.random.default_rng(0).integers(0, 255, (6, 6, 3), dtype=np.uint8)
    import cv2

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    assert np.array_equal(bgr, rgb[:, :, ::-1])
    assert np.array_equal(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), rgb)
