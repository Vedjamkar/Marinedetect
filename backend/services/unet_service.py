"""
U-Net segmentation service (3-class: background / highlight / shadow).

Shape-validating loader: before attempting to load a checkpoint's state
dict, the final conv layer's shape is checked against the configured
UNET_NUM_CLASSES and, on mismatch, the exact tensor name and both shapes are
reported rather than letting torch raise an opaque RuntimeError.

With no checkpoint present this reports unavailable — expected, not faked.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from backend.config import settings
from backend.ml.unet_arch import UNet
from backend.services.lock import INFERENCE_LOCK

logger = logging.getLogger(__name__)

FINAL_LAYER_WEIGHT_KEY = "final_conv.weight"
FINAL_LAYER_BIAS_KEY = "final_conv.bias"


def _extract_state_dict(raw):
    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        return raw["state_dict"]
    return raw


def check_final_layer_shape(state_dict: dict, model: UNet) -> Optional[str]:
    """Returns a precise mismatch message, or None if compatible."""
    model_sd = model.state_dict()
    for key in (FINAL_LAYER_WEIGHT_KEY, FINAL_LAYER_BIAS_KEY):
        if key not in state_dict:
            return f"checkpoint is missing expected tensor '{key}'"
        ckpt_shape = tuple(state_dict[key].shape)
        model_shape = tuple(model_sd[key].shape)
        if ckpt_shape != model_shape:
            return (
                f"shape mismatch on '{key}': checkpoint has {ckpt_shape}, "
                f"configured model expects {model_shape} "
                f"(UNET_NUM_CLASSES={settings.unet_num_classes})"
            )
    return None


class UnetService:
    def __init__(self) -> None:
        self._model: Optional[UNet] = None
        self._attempted = False
        self._load_error: Optional[str] = None
        self._device: Optional[str] = None

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

        path = settings.unet_model_abs_path
        if not path.exists():
            self._load_error = f"weights not found at {path}"
            return

        try:
            import torch
        except Exception as exc:  # pragma: no cover
            self._load_error = f"torch import failed: {exc}"
            return

        try:
            raw = torch.load(str(path), map_location="cpu")
        except Exception as exc:
            self._load_error = f"failed to read checkpoint at {path}: {exc}"
            return

        state_dict = _extract_state_dict(raw)
        if not isinstance(state_dict, dict):
            self._load_error = f"checkpoint at {path} is not a state dict (got {type(state_dict).__name__})"
            return

        model = UNet(in_channels=1, num_classes=settings.unet_num_classes)
        mismatch = check_final_layer_shape(state_dict, model)
        if mismatch:
            self._load_error = f"U-Net checkpoint shape mismatch: {mismatch}"
            return

        try:
            model.load_state_dict(state_dict)
        except Exception as exc:
            self._load_error = f"failed to load state dict from {path}: {exc}"
            return

        model.eval()
        model.to(self._device)
        self._model = model
        logger.info("U-Net model loaded from %s on device=%s", path, self._device)

    def status(self) -> dict:
        self._ensure_loaded()
        return {
            "loaded": self._model is not None,
            "reason": self._load_error,
            "path": str(settings.unet_model_abs_path),
        }

    def segment(self, roi_gray: np.ndarray) -> np.ndarray:
        """Returns an HxW int class-map (values 0..num_classes-1) at the
        original ROI resolution. Raises RuntimeError if unavailable."""
        self._ensure_loaded()
        if self._model is None:
            raise RuntimeError(self._load_error or "U-Net model unavailable")

        import cv2
        import torch

        h0, w0 = roi_gray.shape[:2]
        size = settings.unet_input_size
        resized = cv2.resize(roi_gray.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
        normed = resized / 255.0
        tensor = torch.from_numpy(normed).float().unsqueeze(0).unsqueeze(0).to(self._device)

        with INFERENCE_LOCK:
            with torch.no_grad():
                logits = self._model(tensor)
                if settings.unet_num_classes == 1:
                    probs = torch.sigmoid(logits)
                    class_map = (probs[0, 0] >= settings.unet_mask_threshold).long().cpu().numpy()
                else:
                    probs = torch.softmax(logits, dim=1)
                    class_map = torch.argmax(probs, dim=1)[0].cpu().numpy()

        class_map_full = cv2.resize(
            class_map.astype(np.uint8), (w0, h0), interpolation=cv2.INTER_NEAREST
        )
        return class_map_full


unet_service = UnetService()
