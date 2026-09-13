"""
THE critical test (BACKEND.md step 10): plant a synthetic target with a
shadow corresponding to a KNOWN height, at KNOWN geometry, and assert the
recovered height matches the planted value within tolerance.

This proves the full physics chain end to end —
  intensity profile -> classical shadow measurement -> shadow-geometry height
G = sqrt(R^2 - H^2); h = H * Ls / (G + Ls)
— with zero trained models involved anywhere in the path.
"""
import math

import pytest

from backend.schemas.geometry import SonarGeometry
from backend.services.shadow_service import compute_height, measure_shadow


def _shadow_px_for_target_height(h_true: float, H: float, R: float, res: float) -> int:
    """Inverse of the height formula: given a desired true height, compute
    the shadow length in pixels that would produce it. Used only to *plant*
    the fixture — the test never calls this to derive the expected answer,
    it derives shadow_length_px from `h_true` and checks the pipeline
    recovers `h_true` back out.
    """
    G = math.sqrt(R**2 - H**2)
    Ls = h_true * G / (H - h_true)
    return round(Ls / res)


@pytest.mark.parametrize("shadow_direction", ["down", "up", "left", "right"])
@pytest.mark.parametrize(
    "H,R,res,h_true",
    [
        (5.0, 20.0, 0.05, 1.2),
        (8.0, 30.0, 0.10, 2.0),
        (3.0, 12.0, 0.02, 0.5),
    ],
)
def test_recovers_planted_height_end_to_end(synthetic_roi_factory, shadow_direction, H, R, res, h_true):
    shadow_length_px = _shadow_px_for_target_height(h_true, H, R, res)
    assert shadow_length_px > 0  # sanity: fixture parameters must be physically valid

    roi = synthetic_roi_factory(shadow_length_px=shadow_length_px, shadow_direction=shadow_direction)

    shadow = measure_shadow(roi, shadow_direction, k=1.0)
    assert shadow.available, f"shadow measurement failed: {shadow.reason}"
    assert shadow.method == "classical_profile"

    geometry = SonarGeometry(towfish_altitude_m=H, slant_range_m=R, across_track_resolution_m_per_px=res)
    height = compute_height(shadow, geometry)

    assert height.available, f"height computation failed: {height.reason}"
    assert height.method == "classical_profile"
    assert height.assumptions == ["locally flat seabed", "shadow measured in ground range"]

    # Tolerance covers pixel-quantization of the planted shadow and the
    # threshold-crossing edge effect in measure_shadow (a few px either way).
    assert height.height_m == pytest.approx(h_true, rel=0.20)


def test_height_never_fabricated_without_geometry(synthetic_roi_factory):
    roi = synthetic_roi_factory(shadow_length_px=30, shadow_direction="down")
    shadow = measure_shadow(roi, "down", k=1.0)
    assert shadow.available

    height = compute_height(shadow, SonarGeometry())  # nothing supplied
    assert height.available is False
    assert height.height_m is None
    assert "towfish_altitude_m" in height.reason
    assert "slant_range_m" in height.reason
    assert "across_track_resolution_m_per_px" in height.reason


@pytest.mark.parametrize(
    "field",
    ["towfish_altitude_m", "slant_range_m", "across_track_resolution_m_per_px"],
)
def test_height_names_exactly_the_missing_field(synthetic_roi_factory, field):
    roi = synthetic_roi_factory(shadow_length_px=30, shadow_direction="down")
    shadow = measure_shadow(roi, "down", k=1.0)
    assert shadow.available

    full = {"towfish_altitude_m": 5.0, "slant_range_m": 20.0, "across_track_resolution_m_per_px": 0.05}
    del full[field]
    geometry = SonarGeometry(**full)

    height = compute_height(shadow, geometry)
    assert height.available is False
    assert field in height.reason
    # No other required field should be reported missing when only one is absent.
    other_fields = {"towfish_altitude_m", "slant_range_m", "across_track_resolution_m_per_px"} - {field}
    for other in other_fields:
        assert other not in height.reason


def test_height_unavailable_when_no_shadow(uniform_seabed_roi_factory):
    roi = uniform_seabed_roi_factory()
    shadow = measure_shadow(roi, "down", k=1.0)
    assert shadow.available is False

    geometry = SonarGeometry(towfish_altitude_m=5.0, slant_range_m=20.0, across_track_resolution_m_per_px=0.05)
    height = compute_height(shadow, geometry)
    assert height.available is False
    assert height.height_m is None
    assert "no shadow measurement available" in height.reason


def test_height_rejects_slant_range_not_exceeding_altitude(synthetic_roi_factory):
    roi = synthetic_roi_factory(shadow_length_px=30, shadow_direction="down")
    shadow = measure_shadow(roi, "down", k=1.0)
    assert shadow.available

    geometry = SonarGeometry(towfish_altitude_m=20.0, slant_range_m=10.0, across_track_resolution_m_per_px=0.05)
    height = compute_height(shadow, geometry)
    assert height.available is False
    assert "must exceed" in height.reason
