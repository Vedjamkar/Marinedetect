"""
Classical acoustic-shadow measurement and shadow-geometry height computation.

THE KEY COMPONENT (see BACKEND.md step 6). Requires no trained model at all:
it works on the raw intensity profile of a padded ROI. This is what makes
the demo real before any weights exist, and it stays fully explainable
after they do.

Algorithm (BACKEND.md step 6):
  1. Take the padded ROI. Compute a mean intensity profile along the range
     axis (averaging across the along-track axis suppresses speckle).
  2. The target highlight is the bright peak. The shadow is the dark run
     immediately down-range of it.
  3. shadow_start = first index after the peak where intensity drops below
     seabed_mean - k*seabed_std. shadow_end = where it recovers above that
     threshold for a sustained run.
  4. shadow_length_px = shadow_end - shadow_start.

Height, only if the caller supplied the required geometry:
  G = sqrt(R^2 - H^2)             ground range from slant range and altitude
  Ls = shadow_length_px * across_track_resolution_m_per_px
  h = H * Ls / (G + Ls)           object height from shadow length

If any required input is missing, HeightEstimate.available is False with a
reason naming exactly which input is absent. Never a default, never a guess.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from backend.schemas.geometry import SonarGeometry
from backend.schemas.responses import ShadowCrossCheck, HeightEstimate, IntensityProfile, ShadowMeasurement

# Consecutive above-threshold samples required to call the shadow "recovered"
# (a single noisy bright pixel inside the shadow should not end it early).
SHADOW_MIN_RECOVERY_RUN = 3

# Below this a 'shadow' is noise, not a measurable acoustic shadow.
MIN_SHADOW_LENGTH_PX = 3

# A range row counts as shadow if it holds at least this fraction of the
# strongest shadow row's pixel count.
MIN_SHADOW_ROW_FRACTION = 0.35

# ...and must in any case span at least this fraction of the ROI's along-track
# width. A handful of scattered columns is speckle, not an acoustic shadow,
# however strong they are relative to an otherwise-empty mask.
MIN_SHADOW_ROW_ABS_FRACTION = 0.25

ASSUMPTIONS = ["locally flat seabed", "shadow measured in ground range"]


def axis_and_sign(shadow_direction: str) -> Tuple[int, int]:
    """Map SHADOW_DIRECTION to (profile_axis, sign).

    profile_axis: 0 = rows (vertical), 1 = columns (horizontal) — the axis
    the intensity profile is taken along (the "range axis").
    sign: +1 if down-range corresponds to increasing index along that axis,
    -1 if down-range corresponds to decreasing index.

    "none" has no known orientation; we default to axis=0 (rows), sign=+1
    and this is documented in the response so it is never a silent guess.
    """
    mapping = {
        "down": (0, 1),
        "up": (0, -1),
        "right": (1, 1),
        "left": (1, -1),
        "none": (0, 1),
    }
    return mapping.get(shadow_direction, (0, 1))


SHADOW_CLASS_ID = 2  # pinned 3-class convention: 0 background, 1 highlight, 2 shadow


def compute_intensity_profile(roi_gray: np.ndarray, shadow_direction: str) -> Optional[np.ndarray]:
    """Mean intensity per index along the range axis, canonicalized so that
    increasing index = down-range. Averaging across the along-track axis
    suppresses speckle per BACKEND.md step 6.1. Returns None if the ROI is
    too small to profile.
    """
    if roi_gray.size == 0 or roi_gray.shape[0] < 2 or roi_gray.shape[1] < 2:
        return None

    axis, sign = axis_and_sign(shadow_direction)
    along_track_axis = 1 - axis
    profile = roi_gray.astype(np.float64).mean(axis=along_track_axis)

    if profile.shape[0] < 2:
        return None

    if sign == -1:
        profile = profile[::-1]
    return profile


def measure_shadow(roi_gray: np.ndarray, shadow_direction: str, k: float) -> ShadowMeasurement:
    """Classical shadow measurement from the ROI intensity profile. Requires
    no trained model. `available=False` only when the profile itself can't
    be analyzed (empty ROI, or no shadow run found down-range of the peak).
    """
    profile = compute_intensity_profile(roi_gray, shadow_direction)
    if profile is None:
        return ShadowMeasurement(
            available=False,
            reason="ROI too small along the range axis to compute an intensity profile",
        )

    peak_index = int(np.argmax(profile))

    # Seabed reference statistics: the up-range side of the peak, which is
    # unaffected by both the highlight and the (down-range) shadow.
    seabed_region = profile[:peak_index]
    if seabed_region.size < 3:
        # Not enough up-range context (peak near the ROI edge) — fall back to
        # the whole profile excluding a small window around the peak.
        margin = 2
        mask = np.ones(profile.shape[0], dtype=bool)
        mask[max(0, peak_index - margin): peak_index + margin + 1] = False
        seabed_region = profile[mask] if mask.any() else profile

    seabed_mean = float(np.mean(seabed_region))
    seabed_std = float(np.std(seabed_region))
    threshold = seabed_mean - k * seabed_std

    profile_schema = IntensityProfile(
        values=profile.tolist(),
        seabed_mean=seabed_mean,
        seabed_std=seabed_std,
        threshold=threshold,
        peak_index=peak_index,
    )

    # shadow_start: first index after the peak where intensity drops below
    # threshold for a sustained run (a single noisy sample dipping below
    # threshold is not a real acoustic shadow, which is an extended dark
    # region — symmetric with the recovery condition for shadow_end below).
    shadow_start = None
    for i in range(peak_index + 1, profile.shape[0]):
        if profile[i] < threshold:
            run_end = min(i + SHADOW_MIN_RECOVERY_RUN, profile.shape[0])
            if np.all(profile[i:run_end] < threshold):
                shadow_start = i
                break

    if shadow_start is None:
        return ShadowMeasurement(
            available=False,
            reason=(
                f"no acoustic shadow found: intensity never sustained a drop below "
                f"seabed_mean - {k}*seabed_std ({threshold:.2f}) down-range of the highlight peak "
                f"(index {peak_index})"
            ),
            profile=profile_schema,
        )

    # shadow_end: first index at/after shadow_start where intensity recovers
    # above threshold for a sustained run of SHADOW_MIN_RECOVERY_RUN samples.
    shadow_end = profile.shape[0] - 1  # default: shadow runs to the ROI edge (truncated)
    i = shadow_start
    while i < profile.shape[0]:
        if profile[i] >= threshold:
            run_end = min(i + SHADOW_MIN_RECOVERY_RUN, profile.shape[0])
            if np.all(profile[i:run_end] >= threshold):
                shadow_end = i
                break
        i += 1

    shadow_length_px = float(shadow_end - shadow_start)

    # A start that recovers immediately is a threshold artefact, not a shadow.
    # Reporting it as a zero-length measurement propagates into a height of
    # 0.00 m, which reads as a real answer and is not one.
    if shadow_length_px < MIN_SHADOW_LENGTH_PX:
        return ShadowMeasurement(
            available=False,
            reason=(
                f"shadow run too short to measure ({shadow_length_px:.0f} px, "
                f"minimum {MIN_SHADOW_LENGTH_PX}) - likely a threshold artefact rather "
                f"than an acoustic shadow"
            ),
            profile=profile_schema,
        )

    return ShadowMeasurement(
        available=True,
        method="classical_profile",
        shadow_start_px=shadow_start,
        shadow_end_px=shadow_end,
        shadow_length_px=shadow_length_px,
        profile=profile_schema,
    )


def measure_shadow_from_mask(class_map: np.ndarray, shadow_direction: str) -> ShadowMeasurement:
    """Shadow extent taken from the U-Net class map instead of the intensity profile.

    Class ids follow the pinned 3-class convention: 0 background, 1 highlight,
    2 shadow. We reduce the mask to a per-range-index count of shadow pixels and
    take the longest contiguous run where shadow is the majority along-track.
    Using the longest run rather than the total count keeps a few stray shadow
    pixels elsewhere in the ROI from inflating the length.

    This measures the same physical quantity as `measure_shadow`, by an
    independent route, so the two can be cross-checked against each other.
    """
    if class_map is None or class_map.size == 0 or class_map.ndim != 2:
        return ShadowMeasurement(
            available=False, method="unet", reason="no U-Net class map available for this ROI"
        )

    axis, sign = axis_and_sign(shadow_direction)
    along_track_axis = 1 - axis

    shadow_counts = (class_map == SHADOW_CLASS_ID).sum(axis=along_track_axis)
    along_track_extent = class_map.shape[along_track_axis]
    if along_track_extent == 0:
        return ShadowMeasurement(available=False, method="unet", reason="ROI has zero along-track extent")

    # A strict >50% majority is too demanding for a real segmentation mask:
    # a U-Net that labels a genuine shadow band sparsely would be reported as
    # finding nothing, contradicting its own pixel counts. Threshold instead
    # on a fraction of the strongest row, which adapts to mask quality.
    peak_count = float(shadow_counts.max())
    if peak_count <= 0:
        return ShadowMeasurement(
            available=False, method="unet", reason="U-Net segmented no shadow pixels in this ROI"
        )
    min_count = max(MIN_SHADOW_ROW_FRACTION * peak_count, MIN_SHADOW_ROW_ABS_FRACTION * along_track_extent)
    is_shadow = shadow_counts >= min_count
    if sign == -1:
        is_shadow = is_shadow[::-1]

    if not is_shadow.any():
        return ShadowMeasurement(
            available=False,
            method="unet",
            reason="U-Net segmented no shadow region in this ROI",
        )

    # Longest contiguous True run.
    best_len = best_start = 0
    cur_len = 0
    for i, flag in enumerate(is_shadow):
        if flag:
            cur_len += 1
            if cur_len > best_len:
                best_len, best_start = cur_len, i - cur_len + 1
        else:
            cur_len = 0

    return ShadowMeasurement(
        available=True,
        method="unet",
        shadow_start_px=int(best_start),
        shadow_end_px=int(best_start + best_len),
        shadow_length_px=float(best_len),
    )


def cross_check_shadows(
    classical: ShadowMeasurement, unet: Optional[ShadowMeasurement], tolerance: float
) -> ShadowCrossCheck:
    """Compare the two independent shadow measurements.

    Deliberately does not merge them into one number — averaging would conceal
    the disagreement, and the disagreement is the useful signal.
    """
    if unet is None or not unet.available:
        reason = "no U-Net shadow measurement" if unet is None else unet.reason
        return ShadowCrossCheck(available=False, reason=reason)
    if not classical.available:
        return ShadowCrossCheck(available=False, reason=f"no classical measurement: {classical.reason}")

    a = float(classical.shadow_length_px or 0.0)
    b = float(unet.shadow_length_px or 0.0)
    mean = (a + b) / 2.0
    if mean <= 0:
        return ShadowCrossCheck(available=False, reason="both measurements are zero length")

    diff = abs(a - b)
    rel = diff / mean
    return ShadowCrossCheck(
        available=True,
        classical_length_px=a,
        unet_length_px=b,
        absolute_difference_px=diff,
        relative_difference=rel,
        agrees=bool(rel <= tolerance),
    )


def compute_height(shadow: ShadowMeasurement, geometry: SonarGeometry) -> HeightEstimate:
    """Object height above the seabed from shadow length by similar triangles.
    Never populated unless every required input is present and physically valid.
    """
    if not shadow.available:
        return HeightEstimate(
            available=False,
            reason=f"no shadow measurement available: {shadow.reason}",
        )

    if not shadow.shadow_length_px or shadow.shadow_length_px <= 0:
        return HeightEstimate(
            available=False,
            reason="shadow length is zero - no height can be derived from it",
        )

    missing = []
    if geometry.towfish_altitude_m is None:
        missing.append("towfish_altitude_m")
    if geometry.slant_range_m is None:
        missing.append("slant_range_m")
    if geometry.across_track_resolution_m_per_px is None:
        missing.append("across_track_resolution_m_per_px")

    if missing:
        return HeightEstimate(
            available=False,
            reason=f"missing required geometry input(s): {', '.join(missing)}",
        )

    H = geometry.towfish_altitude_m
    R = geometry.slant_range_m
    res = geometry.across_track_resolution_m_per_px

    if H <= 0 or R <= 0 or res <= 0:
        return HeightEstimate(
            available=False,
            reason=(
                "geometry inputs must be positive: "
                f"towfish_altitude_m={H}, slant_range_m={R}, "
                f"across_track_resolution_m_per_px={res}"
            ),
        )

    if R <= H:
        return HeightEstimate(
            available=False,
            reason=(
                f"slant_range_m ({R}) must exceed towfish_altitude_m ({H}) "
                "for ground_range = sqrt(R^2 - H^2) to be real"
            ),
        )

    ground_range = math.sqrt(R * R - H * H)
    shadow_length_m = shadow.shadow_length_px * res
    height_m = H * shadow_length_m / (ground_range + shadow_length_m)

    return HeightEstimate(
        available=True,
        method=shadow.method,
        height_m=height_m,
        ground_range_m=ground_range,
        shadow_length_m=shadow_length_m,
        assumptions=list(ASSUMPTIONS),
    )
