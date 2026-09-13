"""
YOLO detection service.

Lazy load-once from settings.yolo_model_path. Reports unavailable with a
specific reason rather than crashing when weights are missing, unreadable,
or wrong.

COCO guard: a checkpoint whose class names are (mostly) the 80 COCO classes
is refused in trained mode — it is not a sonar-trained model and relabeling
"car"/"person" detections as sonar targets would be dishonest.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from backend.config import settings
from backend.services.lock import INFERENCE_LOCK

logger = logging.getLogger(__name__)

COCO_CLASS_NAMES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
}
# If at least this fraction of the checkpoint's class names are COCO names,
# treat it as a COCO (or COCO-derived) checkpoint, not a sonar-trained one.
COCO_OVERLAP_THRESHOLD = 0.5


def coco_overlap(name_values: set) -> set:
    """Pure helper (independently testable, no model loading required):
    the subset of `name_values` that are COCO class names."""
    return {str(v).lower().strip() for v in name_values} & COCO_CLASS_NAMES


def is_coco_like(name_values: set) -> bool:
    """True if `name_values` looks like a COCO (or COCO-derived) checkpoint's
    class list, per COCO_OVERLAP_THRESHOLD."""
    normalized = {str(v).lower().strip() for v in name_values}
    if not normalized:
        return False
    return len(coco_overlap(normalized)) / len(normalized) >= COCO_OVERLAP_THRESHOLD


@dataclass
class RawDetection:
    class_name: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2) in pixel coords
    detector: str = "unknown"   # which model produced this, by file stem


class YoloService:
    """One detector. Several of these run side by side - see DetectorPool below."""

    def __init__(self, path=None, name: str = None) -> None:
        self._path = path
        self._name = name
        self._model = None
        self._attempted = False
        self._load_error: Optional[str] = None
        self._device: Optional[str] = None

    # -- status / loading -------------------------------------------------

    def _resolve_device(self) -> str:
        if settings.device != "auto":
            return settings.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _ensure_loaded(self) -> None:
        if self._attempted:
            return
        self._attempted = True
        self._device = self._resolve_device()

        path = self.model_path
        if not path.exists():
            self._load_error = f"weights not found at {path}"
            return

        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - ultralytics is a hard dependency
            self._load_error = f"ultralytics import failed: {exc}"
            return

        try:
            model = YOLO(str(path))
        except Exception as exc:
            self._load_error = f"failed to load checkpoint at {path}: {exc}"
            return

        names = getattr(model, "names", None) or {}
        name_values = set(names.values())
        if is_coco_like(name_values):
            sample = ", ".join(sorted(coco_overlap(name_values))[:6])
            self._load_error = (
                f"checkpoint at {path} has class names matching COCO ({sample}, ...); "
                "this looks like a generic COCO-pretrained checkpoint, not a sonar-trained "
                "model, so it is refused in trained mode. Set a sonar-trained YOLO_MODEL_PATH."
            )
            return

        self._model = model
        logger.info("YOLO model loaded from %s on device=%s", path, self._device)

    def status(self) -> dict:
        self._ensure_loaded()
        return {
            "loaded": self._model is not None,
            "reason": self._load_error,
            "path": str(self.model_path),
            "name": self.name,
        }

    @property
    def model_path(self):
        from pathlib import Path as _P
        if self._path is not None:
            return _P(self._path)
        return settings.yolo_model_abs_path

    @property
    def name(self) -> str:
        return self._name or self.model_path.stem

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = self._resolve_device()
        return self._device

    # -- inference ----------------------------------------------------------

    def predict(self, image_rgb: np.ndarray) -> list[RawDetection]:
        """Run detection on an RGB uint8 image array. Raises RuntimeError if unavailable."""
        self._ensure_loaded()
        if self._model is None:
            raise RuntimeError(self._load_error or "YOLO model unavailable")

        # Ultralytics follows the OpenCV convention: an ndarray source is read as
        # BGR. Our pipeline works in RGB, so it must be converted here. Passing RGB
        # straight through silently swaps the red and blue channels and the model
        # misses detections it would otherwise make at high confidence - verified:
        # the same frame scored 0 detections as RGB and 1 at 0.93 confidence as BGR.
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        with INFERENCE_LOCK:
            results = self._model.predict(
                source=image_bgr,
                conf=settings.yolo_confidence,
                iou=settings.yolo_iou,
                device=self.device,
                verbose=False,
            )

        detections: list[RawDetection] = []
        if not results:
            return detections
        result = results[0]
        names = result.names or {}
        boxes = result.boxes
        if boxes is None:
            return detections
        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0].item())
            cls_idx = int(box.cls[0].item())
            class_name = str(names.get(cls_idx, f"class_{cls_idx}"))
            detections.append(RawDetection(class_name=class_name, confidence=conf,
                                           bbox=tuple(xyxy), detector=self.name))
        return detections


class DetectorPool:
    """Runs every configured detector over the same frame and merges the results.

    Two detectors ship with this project — a seabed-object model and a marine-debris
    model — and they recognise disjoint classes. Making the operator edit .env and
    restart to switch between them means, in practice, only ever seeing one of them.
    Running both costs a few milliseconds and every detection records which model
    produced it, so nothing is attributed to the wrong one.
    """

    def __init__(self) -> None:
        self._services = None

    @property
    def services(self) -> list:
        if self._services is None:
            paths = settings.yolo_model_abs_paths
            self._services = [YoloService(path=p) for p in paths]
        return self._services

    def status(self) -> dict:
        entries = [s.status() for s in self.services]
        loaded = [e for e in entries if e["loaded"]]
        return {
            "loaded": bool(loaded),
            "reason": None if loaded else "; ".join(
                f"{e['name']}: {e['reason']}" for e in entries) or "no detectors configured",
            "path": ", ".join(e["path"] for e in entries),
            "detectors": entries,
        }

    @property
    def device(self) -> str:
        return self.services[0].device if self.services else "cpu"

    def predict(self, image_rgb: np.ndarray) -> list[RawDetection]:
        out: list[RawDetection] = []
        errors = []
        any_loaded = False
        for svc in self.services:
            if not svc.status()["loaded"]:
                errors.append(f"{svc.name}: {svc.status()['reason']}")
                continue
            any_loaded = True
            out.extend(svc.predict(image_rgb))
        if not any_loaded:
            raise RuntimeError("; ".join(errors) or "no detector available")
        # Strongest first, so the headline detection is the most confident one
        # regardless of which model found it.
        out.sort(key=lambda d: d.confidence, reverse=True)
        return out


yolo_service = DetectorPool()
