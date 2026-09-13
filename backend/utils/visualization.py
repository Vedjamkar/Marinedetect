"""
Annotated output rendering — boxes, labels, shadow extent, mask overlay, and
the per-detection intensity-profile plot (the explainability asset: it shows
the shadow measurement actually being made, not just its numeric result).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from backend.services.shadow_service import axis_and_sign

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from backend.services.fusion_service import FusedDetection

# Palette matches the interface: amber = acoustic return, slate = acoustic
# shadow. Saturated green/magenta reads as an alert and fights the sonar
# imagery it sits on, which is the opposite of what an instrument display
# should do.
BOX_COLOR = (240, 176, 92)            # amber - detection box
ROI_COLOR = (120, 140, 150)           # muted slate - padded ROI bounds
SHADOW_BAND_COLOR = (131, 174, 194)   # slate - measured shadow span
HIGHLIGHT_MASK_COLOR = (224, 152, 59) # amber - highlight pixels
SHADOW_MASK_COLOR = (131, 174, 194)   # slate - shadow pixels


def _shadow_band_abs_coords(
    roi_shape: tuple[int, int], roi_bbox: tuple[int, int, int, int], shadow_direction: str, start_idx: int, end_idx: int
) -> tuple[int, int, int, int]:
    """Map canonical shadow-profile indices back to an absolute-image rectangle."""
    axis, sign = axis_and_sign(shadow_direction)
    roi_x1, roi_y1, roi_x2, roi_y2 = roi_bbox
    length_along_axis = roi_shape[axis]

    if sign == 1:
        raw_start, raw_end = start_idx, end_idx
    else:
        raw_start = length_along_axis - 1 - end_idx
        raw_end = length_along_axis - 1 - start_idx

    if axis == 0:  # rows: shadow band is horizontal, spans full ROI width
        return roi_x1, roi_y1 + raw_start, roi_x2, roi_y1 + raw_end
    else:  # cols: shadow band is vertical, spans full ROI height
        return roi_x1 + raw_start, roi_y1, roi_x1 + raw_end, roi_y2


def _alpha_blend_rect(image: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: tuple, alpha: float = 0.35) -> None:
    x1, x2 = sorted((max(0, x1), min(image.shape[1], x2)))
    y1, y2 = sorted((max(0, y1), min(image.shape[0], y2)))
    if x2 <= x1 or y2 <= y1:
        return
    overlay = image[y1:y2, x1:x2].astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)
    blended = overlay * (1 - alpha) + color_arr * alpha
    image[y1:y2, x1:x2] = blended.astype(np.uint8)


def draw_annotations(rgb: np.ndarray, fused: list["FusedDetection"], shadow_direction: str) -> np.ndarray:
    """Returns a new annotated RGB uint8 array. Does not mutate the input."""
    out = rgb.copy()

    for item in fused:
        det = item.detection
        b = det.bbox
        rb = det.roi_bbox

        # Mask overlay (highlight/shadow classes), drawn first so boxes sit on top.
        if item.class_map is not None:
            roi_x1, roi_y1 = int(rb.x1), int(rb.y1)
            class_map = item.class_map
            highlight_mask = class_map == 1
            shadow_mask = class_map == 2 if class_map.max() >= 2 else np.zeros_like(class_map, dtype=bool)
            region = out[roi_y1: roi_y1 + class_map.shape[0], roi_x1: roi_x1 + class_map.shape[1]]
            if region.shape[:2] == class_map.shape[:2]:
                region[highlight_mask] = (
                    region[highlight_mask].astype(np.float32) * 0.5
                    + np.array(HIGHLIGHT_MASK_COLOR, dtype=np.float32) * 0.5
                ).astype(np.uint8)
                region[shadow_mask] = (
                    region[shadow_mask].astype(np.float32) * 0.5
                    + np.array(SHADOW_MASK_COLOR, dtype=np.float32) * 0.5
                ).astype(np.uint8)

        # Padded ROI box (thin).
        cv2.rectangle(out, (int(rb.x1), int(rb.y1)), (int(rb.x2), int(rb.y2)), ROI_COLOR, 1)

        # Shadow extent band, mapped back into absolute image coordinates.
        if det.shadow.available and det.shadow.shadow_start_px is not None and det.shadow.shadow_end_px is not None:
            band = _shadow_band_abs_coords(
                item.roi_gray.shape[:2],
                (int(rb.x1), int(rb.y1), int(rb.x2), int(rb.y2)),
                shadow_direction,
                det.shadow.shadow_start_px,
                det.shadow.shadow_end_px,
            )
            _alpha_blend_rect(out, *band, SHADOW_BAND_COLOR, alpha=0.35)

        # Detection box + label.
        cv2.rectangle(out, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), BOX_COLOR, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        # Name the source model: two detectors run on every frame and a
        # viewer must never have to guess which one drew a box.
        if getattr(det, "detector", None) and det.detector != "unknown":
            label = f"[{det.detector}] " + label
        if det.height.available and det.height.height_m is not None:
            label += f" | h={det.height.height_m:.2f}m"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(int(b.y1) - 6, th + 2)
        cv2.rectangle(out, (int(b.x1), ty - th - 4), (int(b.x1) + tw + 4, ty + 2), BOX_COLOR, -1)
        cv2.putText(out, label, (int(b.x1) + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (18, 22, 24), 1, cv2.LINE_AA)

    return out


def render_profile_figure(item: "FusedDetection", index: int) -> "Figure":
    """Intensity profile plot for one detection, shadow span bracketed.
    This is the explainability asset — it shows the measurement, not just
    the resulting number.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    det = item.detection
    fig, ax = plt.subplots(figsize=(6, 3), dpi=110)

    profile = det.shadow.profile
    if profile is None:
        ax.text(0.5, 0.5, "no profile available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"Detection {index}: {det.class_name} — no shadow profile")
        fig.tight_layout()
        return fig

    values = profile.values
    xs = list(range(len(values)))
    ax.plot(xs, values, color="#1f77b4", linewidth=1.2, label="mean intensity")
    ax.axhline(profile.threshold, color="#d62728", linestyle="--", linewidth=1, label="shadow threshold")
    ax.axvline(profile.peak_index, color="#2ca02c", linestyle=":", linewidth=1, label="highlight peak")

    if det.shadow.available and det.shadow.shadow_start_px is not None and det.shadow.shadow_end_px is not None:
        ax.axvspan(
            det.shadow.shadow_start_px,
            det.shadow.shadow_end_px,
            color="magenta",
            alpha=0.2,
            label="measured shadow",
        )

    title = f"Detection {index}: {det.class_name} (conf {det.confidence:.2f}) — classical_profile"
    if det.height.available and det.height.height_m is not None:
        title += f"\nheight = {det.height.height_m:.3f} m"
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("range-axis index (down-range →)")
    ax.set_ylabel("mean intensity")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    return fig


# Class colours for the segmentation panel, RGB. These match the demo UI's
# palette so a class reads the same colour everywhere in the product.
SEG_HIGHLIGHT_RGB = (224, 152, 59)    # amber - acoustic return off the target
SEG_SHADOW_RGB = (131, 174, 194)      # slate - acoustic shadow behind it
_PANEL_BG = 22
_LABEL_H = 22
_TARGET_LONG_EDGE = 560


def _label_strip(width: int, text: str) -> "np.ndarray":
    import cv2 as _cv2
    import numpy as _np

    strip = _np.full((_LABEL_H, width, 3), _PANEL_BG, dtype=_np.uint8)
    _cv2.putText(strip, text, (6, 15), _cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                 (150, 168, 172), 1, _cv2.LINE_AA)
    return strip


def render_segmentation_panel(roi_gray: "np.ndarray", class_map: "np.ndarray") -> "np.ndarray":
    """Three labelled views of one detection: the raw ROI, the U-Net overlay,
    and the class map on its own.

    Layout adapts to the ROI's shape. Padded sonar ROIs are often much wider
    than they are tall, and stacking those side by side produces an unreadable
    sliver, so wide ROIs stack vertically and tall ones stack horizontally.

    The class map is shown separately as well as overlaid because the overlay
    alone makes it hard to see where the model actually drew the boundary -
    which is the thing worth judging, since the shadow boundary is what the
    height calculation consumes.

    Returns RGB, or None if the inputs do not correspond.
    """
    import cv2 as _cv2
    import numpy as _np

    if roi_gray is None or class_map is None:
        return None
    if roi_gray.ndim != 2 or class_map.shape != roi_gray.shape:
        return None

    h, w = roi_gray.shape
    if h < 2 or w < 2:
        return None

    # Normalize scale so small ROIs are legible and large ones do not dominate.
    scale = _TARGET_LONG_EDGE / max(h, w)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    interp = _cv2.INTER_NEAREST if scale > 1 else _cv2.INTER_AREA

    base = _cv2.cvtColor(_cv2.resize(roi_gray, (nw, nh), interpolation=interp),
                         _cv2.COLOR_GRAY2RGB)
    cm = _cv2.resize(class_map, (nw, nh), interpolation=_cv2.INTER_NEAREST)

    overlay = base.copy()
    flat = _np.full_like(base, 28)          # dark ground for the standalone map
    for cls_id, colour in ((1, SEG_HIGHLIGHT_RGB), (2, SEG_SHADOW_RGB)):
        mask = cm == cls_id
        if not mask.any():
            continue
        overlay[mask] = (0.40 * base[mask]
                         + 0.60 * _np.array(colour, dtype=_np.float32)).astype(_np.uint8)
        flat[mask] = colour
        contours, _ = _cv2.findContours(mask.astype(_np.uint8),
                                        _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
        _cv2.drawContours(overlay, contours, -1, colour, 1)

    views = [("SONAR ROI", base), ("U-NET OVERLAY", overlay), ("CLASS MAP", flat)]
    blocks = [_np.vstack([_label_strip(nw, name), img]) for name, img in views]

    # Side by side, so the three views can be compared directly against each
    # other rather than by scrolling between them.
    gap_px = 6
    gap = _np.full((blocks[0].shape[0], gap_px, 3), _PANEL_BG, dtype=_np.uint8)
    out = blocks[0]
    for b in blocks[1:]:
        out = _np.hstack([out, gap, b])

    return _cv2.copyMakeBorder(out, 8, 8, 8, 8, _cv2.BORDER_CONSTANT,
                               value=(_PANEL_BG, _PANEL_BG, _PANEL_BG))
