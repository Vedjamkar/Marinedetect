"""
Binds each YOLO detection to its padded ROI, classical shadow measurement,
height estimate (if computable), and U-Net segmentation (if available).

Per BACKEND.md step 8: detection confidence and segmentation metrics are
never blended into a single "combined confidence" score. They stay separate
fields on the Detection response.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from backend.config import settings
from backend.schemas.geometry import SonarGeometry
from backend.schemas.responses import BBox, Detection, Segmentation
from backend.services.shadow_service import (
    compute_height,
    cross_check_shadows,
    measure_shadow,
    measure_shadow_from_mask,
)
from backend.services.unet_service import unet_service
from backend.services.yolo_service import RawDetection
from backend.utils.image_utils import crop_padded_roi

logger = logging.getLogger(__name__)


@dataclass
class FusedDetection:
    """Detection plus the internal arrays visualization.py needs.

    `detection` is the serializable schema (goes in the API response).
    `roi_gray` / `class_map` are internal-only, used to render the annotated
    image and profile plot for this detection.
    """

    detection: Detection
    roi_gray: np.ndarray
    class_map: Optional[np.ndarray]  # None unless U-Net segmentation is available


def _segment_roi(roi_gray: np.ndarray) -> tuple[Segmentation, Optional[np.ndarray]]:
    status = unet_service.status()
    if not status["loaded"]:
        return Segmentation(available=False, reason=status["reason"]), None

    try:
        class_map = unet_service.segment(roi_gray)
    except Exception as exc:
        logger.exception("U-Net segmentation failed")
        return Segmentation(available=False, reason=f"segmentation failed: {exc}"), None

    if settings.unet_num_classes == 1:
        highlight_area = int(np.sum(class_map == 1))
        shadow_area = None
    else:
        highlight_area = int(np.sum(class_map == 1))
        shadow_area = int(np.sum(class_map == 2))

    segmentation = Segmentation(
        available=True,
        highlight_area_px=highlight_area,
        shadow_area_px=shadow_area,
    )
    return segmentation, class_map


def fuse_detections(
    gray: np.ndarray,
    raw_detections: list[RawDetection],
    geometry: SonarGeometry,
) -> list[FusedDetection]:
    fused: list[FusedDetection] = []
    for idx, raw in enumerate(raw_detections):
        roi_gray, roi_bbox = crop_padded_roi(
            gray, raw.bbox, settings.roi_pad_ratio, settings.shadow_direction
        )

        # Two independent shadow measurements off the same padded ROI.
        # Classical always runs — it needs no weights. U-Net runs only when a
        # checkpoint is loaded.
        shadow_classical = measure_shadow(
            roi_gray, settings.shadow_direction, settings.shadow_threshold_k
        )
        segmentation, class_map = _segment_roi(roi_gray)

        shadow_unet = None
        if class_map is not None:
            shadow_unet = measure_shadow_from_mask(class_map, settings.shadow_direction)

        # The classical profile drives the height, and U-Net cross-checks it.
        #
        # This ordering is a measurement, not a preference. The classical method
        # recovers a planted synthetic height to within 0.4%, while the current
        # U-Net checkpoint scores shadow IoU 0.196 on weakly-labelled data.
        # Letting the weaker measurement produce the reported number would make
        # heights worse, so it does not. Revisit this once the U-Net is trained
        # on real annotations and its shadow IoU actually beats the profile.
        shadow = shadow_classical
        if not shadow.available and shadow_unet is not None and shadow_unet.available:
            # Classical found nothing but U-Net did: use it rather than
            # reporting no height at all. `method` records which one it was.
            shadow = shadow_unet

        height = compute_height(shadow, geometry)
        cross_check = cross_check_shadows(
            shadow_classical, shadow_unet, settings.shadow_agreement_tolerance
        )

        x1, y1, x2, y2 = raw.bbox
        rx1, ry1, rx2, ry2 = roi_bbox

        detection = Detection(
            id=idx,
            class_name=raw.class_name,
            confidence=raw.confidence,
            detector=getattr(raw, 'detector', 'unknown'),
            bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
            roi_bbox=BBox(x1=rx1, y1=ry1, x2=rx2, y2=ry2),
            shadow=shadow,
            shadow_classical=shadow_classical,
            shadow_unet=shadow_unet,
            cross_check=cross_check,
            height=height,
            segmentation=segmentation,
        )
        fused.append(FusedDetection(detection=detection, roi_gray=roi_gray, class_map=class_map))
    return fused
