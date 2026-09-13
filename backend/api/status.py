"""GET /api/v1/model-status — must not crash when weights are absent."""
from fastapi import APIRouter

from backend.config import settings
from backend.schemas.responses import ModelEntry, ModelStatus
from backend.services.unet_service import unet_service
from backend.services.yolo_service import yolo_service

router = APIRouter()


@router.get("/api/v1/model-status", response_model=ModelStatus)
def model_status() -> ModelStatus:
    yolo_status = yolo_service.status()
    unet_status = unet_service.status()

    if yolo_status["loaded"]:
        mode = "trained"
        trained_model = True
    elif settings.allow_classical_fallback:
        # ALLOW_CLASSICAL_FALLBACK is opt-in per PLAN.md 1.3; the classical
        # detector fallback itself is out of scope for this build (not in
        # BACKEND.md's step list), so this mode is reported but not served.
        mode = "classical_baseline"
        trained_model = False
    else:
        mode = "unavailable"
        trained_model = False

    return ModelStatus(
        mode=mode,
        trained_model=trained_model,
        device=yolo_service.device,
        yolo=ModelEntry(**yolo_status),
        unet=ModelEntry(**unet_status),
        thresholds={
            "yolo_confidence": settings.yolo_confidence,
            "yolo_iou": settings.yolo_iou,
            "roi_pad_ratio": settings.roi_pad_ratio,
            "shadow_direction": settings.shadow_direction,
            "shadow_threshold_k": settings.shadow_threshold_k,
            "unet_num_classes": settings.unet_num_classes,
            "unet_input_size": settings.unet_input_size,
        },
    )
