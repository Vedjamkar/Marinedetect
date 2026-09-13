"""
Response schemas for the inference API.

Honesty rules encoded here (see BACKEND.md):
  - Fields that require an input the caller didn't supply are `available:
    False` with a `reason` naming exactly what's missing. They are never
    silently defaulted.
  - Every measurement names the method that produced it. A classical
    shadow-geometry measurement is never labelled as if it came from a
    neural network, and vice versa.
  - No blended/combined confidence score across detection + segmentation.
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field

from backend.schemas.geometry import SonarGeometry

Method = Literal["classical_profile", "unet"]
Mode = Literal["trained", "unavailable", "classical_baseline"]


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class IntensityProfile(BaseModel):
    """The along-range mean intensity profile used to find the shadow.

    Present so the measurement is inspectable/plottable, not just a number.
    """

    values: list[float] = Field(description="Mean intensity per range-axis pixel across the padded ROI.")
    seabed_mean: float
    seabed_std: float
    threshold: float = Field(description="seabed_mean - k * seabed_std, the level the shadow must fall below.")
    peak_index: int = Field(description="Index of the highlight peak within `values`.")


class ShadowMeasurement(BaseModel):
    """Classical acoustic-shadow measurement from the ROI intensity profile.

    This never requires a trained model. `available=False` only if the ROI
    profile itself could not be analyzed (e.g. no clear shadow run found) —
    that is independent of whether height could subsequently be computed.
    """

    available: bool
    reason: Optional[str] = None
    method: Method = "classical_profile"
    shadow_start_px: Optional[int] = None
    shadow_end_px: Optional[int] = None
    shadow_length_px: Optional[float] = None
    profile: Optional[IntensityProfile] = None
    profile_plot_url: Optional[str] = Field(
        default=None, description="Rendered intensity-profile plot with the shadow span bracketed."
    )


class HeightEstimate(BaseModel):
    """Object height above the seabed, from shadow length by similar triangles.

    G = sqrt(R^2 - H^2)        ground range from slant range and altitude
    h = H * Ls / (G + Ls)      object height from ground-range shadow length

    Computed only if the caller supplied towfish_altitude_m, slant_range_m,
    and across_track_resolution_m_per_px, AND a shadow was measured. If any
    of those is missing, `available=False` and `reason` names exactly which
    input is absent. Never a placeholder, never a default.
    """

    available: bool
    reason: Optional[str] = None
    method: Optional[Method] = None
    height_m: Optional[float] = None
    ground_range_m: Optional[float] = None
    shadow_length_m: Optional[float] = None
    assumptions: list[str] = Field(default_factory=list)


class Segmentation(BaseModel):
    """3-class U-Net segmentation (background / highlight / shadow) of the ROI.

    Unavailable whenever no checkpoint is loaded. Never faked.
    """

    available: bool
    reason: Optional[str] = None
    method: Literal["unet"] = "unet"
    mask_png_url: Optional[str] = None
    highlight_area_px: Optional[int] = None
    shadow_area_px: Optional[int] = None


class ShadowCrossCheck(BaseModel):
    """Agreement between the two independent shadow measurements.

    The classical intensity profile and the U-Net shadow mask measure the same
    physical quantity by different means. Where they agree, confidence is high.
    Where they diverge, the detection belongs in an analyst review queue.

    This is deliberately a validation signal and not a merged number: we do not
    average the two into a single "better" estimate, because that would hide the
    disagreement that makes the check useful. It also requires no ground truth,
    which matters because none exists for this data.
    """

    available: bool
    reason: Optional[str] = None
    classical_length_px: Optional[float] = None
    unet_length_px: Optional[float] = None
    absolute_difference_px: Optional[float] = None
    relative_difference: Optional[float] = Field(
        default=None, description="|classical - unet| / mean(classical, unet)."
    )
    agrees: Optional[bool] = Field(
        default=None, description="True when relative_difference is within the configured tolerance."
    )


class FullFrameAnalysis(BaseModel):
    """Shadow measurement over the whole image, with no detector involved.

    Requested explicitly via `analyze_full_frame`. This is NOT a detection:
    there is no class and no detection confidence, and the response says so.
    It exists so the shadow-geometry chain can be exercised and verified
    independently of whether any detector is loaded or fires.
    """

    requested: bool = False
    note: str = "Full-frame shadow analysis. Not a detection - no class or detection confidence."
    shadow: Optional[ShadowMeasurement] = None
    height: Optional[HeightEstimate] = None


class Detection(BaseModel):
    id: int
    class_name: str
    confidence: float = Field(description="YOLO detection confidence. Never blended with segmentation metrics.")
    detector: str = Field(default="unknown", description="Which detector model produced this detection.")
    bbox: BBox
    roi_bbox: BBox = Field(description="Padded ROI bbox actually used for shadow/segmentation analysis.")
    shadow: ShadowMeasurement = Field(
        description="The shadow measurement used for the height estimate. Its `method` names the source."
    )
    shadow_classical: Optional[ShadowMeasurement] = Field(
        default=None,
        description="Classical profile measurement, always attempted. Same object as `shadow` when U-Net is unavailable.",
    )
    shadow_unet: Optional[ShadowMeasurement] = Field(
        default=None, description="Shadow measured from the U-Net mask. None when no checkpoint is loaded."
    )
    cross_check: ShadowCrossCheck = Field(
        default_factory=lambda: ShadowCrossCheck(
            available=False, reason="requires both a classical and a U-Net shadow measurement"
        )
    )
    height: HeightEstimate
    segmentation: Segmentation
    spatial: dict = Field(
        default_factory=lambda: {"available": False, "reason": "module not implemented"},
        description="lat/lon/depth. Not computed by this service — see PLAN.md 6.",
    )


class ModelEntry(BaseModel):
    loaded: bool
    reason: Optional[str] = None
    path: Optional[str] = None


class ModelStatus(BaseModel):
    mode: Mode
    trained_model: bool
    device: str
    yolo: ModelEntry
    unet: ModelEntry
    thresholds: dict


class PreprocessingInfo(BaseModel):
    applied: bool
    reason: Optional[str] = None


class InferenceResponse(BaseModel):
    mode: Mode
    trained_model: bool
    detections: list[Detection]
    geometry: SonarGeometry
    preprocessing: PreprocessingInfo
    annotated_image_url: Optional[str] = None
    full_frame: FullFrameAnalysis = Field(default_factory=FullFrameAnalysis)
    timings_ms: dict
    spatial_localization: dict = Field(
        default_factory=lambda: {"available": False, "reason": "module not implemented"}
    )
