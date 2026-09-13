"""
The U-Net shadow path and the classical/U-Net cross-check.

These exercise `measure_shadow_from_mask` and `cross_check_shadows` with
synthetic class maps, so they prove the merged pipeline works without needing
a trained U-Net checkpoint. Hermetic — no weights on disk.
"""
import numpy as np
import pytest

from backend.schemas.geometry import SonarGeometry
from backend.schemas.responses import ShadowMeasurement
from backend.services.shadow_service import (
    compute_height,
    cross_check_shadows,
    measure_shadow_from_mask,
)

BACKGROUND, HIGHLIGHT, SHADOW = 0, 1, 2


def make_class_map(rows: int, cols: int, shadow_rows: slice, along_axis_is_rows: bool = True):
    """Class map with a shadow band spanning the full along-track extent."""
    cm = np.full((rows, cols), BACKGROUND, dtype=np.int64)
    if along_axis_is_rows:
        cm[shadow_rows, :] = SHADOW
    else:
        cm[:, shadow_rows] = SHADOW
    return cm


def test_recovers_shadow_length_from_mask():
    cm = make_class_map(40, 20, slice(10, 25))  # 15-row shadow band
    m = measure_shadow_from_mask(cm, "down")
    assert m.available
    assert m.method == "unet"
    assert m.shadow_length_px == 15.0
    assert m.shadow_start_px == 10


def test_direction_up_reverses_the_range_axis():
    """'up' means down-range is decreasing row index, so the reported start
    must be measured from the far end."""
    cm = make_class_map(40, 20, slice(10, 25))
    m = measure_shadow_from_mask(cm, "up")
    assert m.available
    assert m.shadow_length_px == 15.0
    # 40 rows, band occupies 10..25; reversed that is 15..30.
    assert m.shadow_start_px == 15


def test_longest_run_wins_over_stray_pixels():
    """Scattered shadow pixels elsewhere must not inflate the length."""
    cm = make_class_map(60, 20, slice(20, 32))  # real 12-row shadow
    cm[5, :] = SHADOW   # stray single row far up-range
    cm[50, :] = SHADOW  # another stray
    m = measure_shadow_from_mask(cm, "down")
    assert m.available
    assert m.shadow_length_px == 12.0, "should take the longest contiguous run, not the total count"


def test_no_shadow_class_reports_unavailable_with_reason():
    cm = np.full((30, 20), BACKGROUND, dtype=np.int64)
    cm[10:14, :] = HIGHLIGHT  # a target but no shadow
    m = measure_shadow_from_mask(cm, "down")
    assert not m.available
    assert m.method == "unet"
    assert "no shadow" in m.reason.lower()


def test_minority_shadow_along_track_is_not_counted():
    """A few shadow pixels per row is not a shadow band."""
    cm = np.full((30, 20), BACKGROUND, dtype=np.int64)
    cm[10:20, 0:3] = SHADOW  # only 3 of 20 columns — below the 50% majority
    m = measure_shadow_from_mask(cm, "down")
    assert not m.available


def test_empty_map_is_handled():
    assert not measure_shadow_from_mask(np.array([]), "down").available
    assert not measure_shadow_from_mask(None, "down").available


# --- cross-check -----------------------------------------------------------

def _sm(length, method="classical_profile", available=True, reason=None):
    return ShadowMeasurement(
        available=available, method=method, reason=reason,
        shadow_start_px=0 if available else None,
        shadow_end_px=int(length) if available else None,
        shadow_length_px=float(length) if available else None,
    )


def test_cross_check_agrees_when_close():
    cc = cross_check_shadows(_sm(20.0), _sm(22.0, "unet"), tolerance=0.25)
    assert cc.available and cc.agrees
    assert cc.classical_length_px == 20.0
    assert cc.unet_length_px == 22.0
    assert cc.absolute_difference_px == pytest.approx(2.0)
    assert cc.relative_difference == pytest.approx(2.0 / 21.0)


def test_cross_check_disagrees_when_far_apart():
    cc = cross_check_shadows(_sm(10.0), _sm(40.0, "unet"), tolerance=0.25)
    assert cc.available
    assert cc.agrees is False, "a 3x discrepancy must be flagged, not averaged away"


def test_cross_check_does_not_merge_the_two_values():
    """The check must report both numbers, never a blended one — the
    disagreement is the signal."""
    cc = cross_check_shadows(_sm(10.0), _sm(40.0, "unet"), tolerance=0.25)
    assert cc.classical_length_px == 10.0 and cc.unet_length_px == 40.0
    assert not hasattr(cc, "merged_length_px")


def test_cross_check_unavailable_without_unet():
    cc = cross_check_shadows(_sm(20.0), None, tolerance=0.25)
    assert not cc.available
    assert "u-net" in cc.reason.lower()


def test_cross_check_unavailable_when_unet_found_nothing():
    unet = _sm(0, "unet", available=False, reason="U-Net segmented no shadow region in this ROI")
    cc = cross_check_shadows(_sm(20.0), unet, tolerance=0.25)
    assert not cc.available
    assert "no shadow" in cc.reason.lower()


# --- height from the U-Net measurement -------------------------------------

def test_height_from_unet_mask_names_unet_as_the_method():
    """A height derived from the U-Net mask must not be labelled classical."""
    cm = make_class_map(200, 40, slice(60, 120))  # 60 px shadow
    shadow = measure_shadow_from_mask(cm, "down")
    geom = SonarGeometry(
        towfish_altitude_m=12.0,
        slant_range_m=40.0,
        across_track_resolution_m_per_px=0.05,
    )
    h = compute_height(shadow, geom)
    assert h.available
    assert h.method == "unet", "method must reflect the actual source of the measurement"

    # Ls = 60 px * 0.05 = 3.0 m; G = sqrt(40^2 - 12^2) = 38.158
    # h = 12 * 3 / (38.158 + 3) = 0.8746
    assert h.shadow_length_m == pytest.approx(3.0)
    assert h.ground_range_m == pytest.approx(38.158, abs=1e-2)
    assert h.height_m == pytest.approx(0.8746, abs=1e-3)


def test_height_from_unet_still_refuses_without_geometry():
    cm = make_class_map(200, 40, slice(60, 120))
    shadow = measure_shadow_from_mask(cm, "down")
    h = compute_height(shadow, SonarGeometry(towfish_altitude_m=12.0))
    assert not h.available
    assert "slant_range_m" in h.reason
    assert "across_track_resolution_m_per_px" in h.reason
    assert h.height_m is None
