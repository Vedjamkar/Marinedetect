"""POST /api/v1/inference — multipart image upload + optional geometry JSON.

Sync `def` handler (FastAPI runs it in a threadpool); GPU/CPU model calls are
additionally serialized via services.lock.INFERENCE_LOCK inside each service.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.config import settings
from backend.schemas.geometry import SonarGeometry
from backend.schemas.responses import FullFrameAnalysis, InferenceResponse, PreprocessingInfo
from backend.services.fusion_service import fuse_detections
from backend.services.shadow_service import compute_height, measure_shadow
from backend.services.yolo_service import yolo_service
from backend.utils.image_utils import ImageLoadError, UploadTooLargeError, load_image_safe, save_upload_streaming
from backend.utils.visualization import draw_annotations, render_segmentation_panel, render_profile_figure

logger = logging.getLogger(__name__)
router = APIRouter()


def _sweep_old_outputs() -> None:
    """Best-effort deletion of files in outputs/ older than OUTPUT_RETENTION_HOURS."""
    cutoff = time.time() - settings.output_retention_hours * 3600
    try:
        for p in settings.outputs_dir.iterdir():
            try:
                if p.is_file() and p.stat().st_mtime < cutoff:
                    p.unlink()
            except OSError:
                continue
    except FileNotFoundError:
        pass


@router.post("/api/v1/inference", response_model=InferenceResponse)
def run_inference(
    image: UploadFile = File(...),
    geometry: Optional[str] = Form(default=None, description="JSON-encoded SonarGeometry fields"),
    analyze_full_frame: bool = Form(
        default=False,
        description="Also measure the shadow over the whole image, independent of any detection.",
    ),
) -> InferenceResponse:
    _sweep_old_outputs()

    # --- parse geometry (never defaulted/guessed; absent fields stay None) ---
    if geometry:
        try:
            geometry_obj = SonarGeometry(**json.loads(geometry))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"invalid geometry JSON: {exc}") from exc
    else:
        geometry_obj = SonarGeometry()

    # --- model availability gate ---
    yolo_status = yolo_service.status()
    if not yolo_status["loaded"]:
        raise HTTPException(
            status_code=503,
            detail={
                "mode": "unavailable",
                "trained_model": False,
                "reason": yolo_status["reason"],
            },
        )

    timings_ms: dict[str, float] = {}
    t_total_start = time.perf_counter()

    # --- stream upload to disk, aborting mid-write if oversized ---
    suffix = Path(image.filename or "").suffix or ".bin"
    upload_path = settings.uploads_dir / f"{uuid.uuid4().hex}{suffix}"
    try:
        save_upload_streaming(image.file, upload_path, settings.max_upload_bytes)
    except UploadTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc

    try:
        # --- load + normalize ---
        t0 = time.perf_counter()
        try:
            gray, rgb = load_image_safe(upload_path, settings.max_image_pixels)
        except ImageLoadError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        timings_ms["preprocess"] = (time.perf_counter() - t0) * 1000

        # No preprocess.json sidecar mechanism is implemented in this build
        # (PLAN.md 1.1's identity default, applied statically).
        preprocessing = PreprocessingInfo(
            applied=False,
            reason="no preprocess.json sidecar mechanism in this build; defaulting to identity to avoid train/test skew",
        )

        # --- detect ---
        t0 = time.perf_counter()
        try:
            raw_detections = yolo_service.predict(rgb)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail={"mode": "unavailable", "reason": str(exc)}) from exc
        timings_ms["detect"] = (time.perf_counter() - t0) * 1000

        # --- fuse: shadow measurement + height + segmentation per detection ---
        t0 = time.perf_counter()
        fused = fuse_detections(gray, raw_detections, geometry_obj)
        timings_ms["fuse"] = (time.perf_counter() - t0) * 1000

        # --- optional full-frame shadow analysis (no detector involved) ---
        full_frame = FullFrameAnalysis(requested=analyze_full_frame)
        if analyze_full_frame:
            ff_shadow = measure_shadow(
                gray, settings.shadow_direction, settings.shadow_threshold_k
            )
            full_frame.shadow = ff_shadow
            full_frame.height = compute_height(ff_shadow, geometry_obj)

        # --- render annotated image + per-detection profile plots ---
        t0 = time.perf_counter()
        annotated = draw_annotations(rgb, fused, settings.shadow_direction)
        annotated_name = f"{uuid.uuid4().hex}.png"
        annotated_path = settings.outputs_dir / annotated_name
        cv2.imwrite(str(annotated_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

        # Per-detection segmentation panel (raw ROI | classified ROI). This is
        # what makes the U-Net output inspectable rather than two pixel counts.
        for item in fused:
            if item.class_map is None or not item.detection.segmentation.available:
                continue
            panel = render_segmentation_panel(item.roi_gray, item.class_map)
            if panel is None:
                continue
            seg_name = f"{uuid.uuid4().hex}.png"
            cv2.imwrite(
                str(settings.outputs_dir / seg_name),
                cv2.cvtColor(panel, cv2.COLOR_RGB2BGR),
            )
            item.detection.segmentation.mask_png_url = f"/outputs/{seg_name}"

        for idx, item in enumerate(fused):
            if item.detection.shadow.profile is None:
                continue
            fig = render_profile_figure(item, idx)
            plot_name = f"{uuid.uuid4().hex}.png"
            plot_path = settings.outputs_dir / plot_name
            fig.savefig(plot_path)
            import matplotlib.pyplot as plt

            plt.close(fig)
            item.detection.shadow.profile_plot_url = f"/outputs/{plot_name}"
        timings_ms["render"] = (time.perf_counter() - t0) * 1000

        timings_ms["total"] = (time.perf_counter() - t_total_start) * 1000

        return InferenceResponse(
            mode="trained",
            trained_model=True,
            detections=[item.detection for item in fused],
            full_frame=full_frame,
            geometry=geometry_obj,
            preprocessing=preprocessing,
            annotated_image_url=f"/outputs/{annotated_name}",
            timings_ms={k: round(v, 2) for k, v in timings_ms.items()},
        )
    finally:
        upload_path.unlink(missing_ok=True)
