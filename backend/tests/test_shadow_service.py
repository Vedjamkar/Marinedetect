"""Unit tests for the classical shadow measurement itself (no geometry, no
height involved yet) — purely the intensity-profile analysis."""
import numpy as np
import pytest

from backend.services.shadow_service import measure_shadow


@pytest.mark.parametrize("shadow_direction", ["down", "up", "left", "right", "none"])
def test_measures_known_shadow_length(synthetic_roi_factory, shadow_direction):
    roi = synthetic_roi_factory(shadow_length_px=30, shadow_direction=shadow_direction)
    result = measure_shadow(roi, shadow_direction, k=1.0)

    assert result.available, result.reason
    assert result.method == "classical_profile"
    # Small slack for threshold-crossing edge effects at the shadow boundary.
    assert abs(result.shadow_length_px - 30) <= 3
    assert result.profile is not None
    assert len(result.profile.values) == roi.shape[0 if shadow_direction in ("down", "up", "none") else 1]


def test_no_shadow_on_flat_seabed(uniform_seabed_roi_factory):
    roi = uniform_seabed_roi_factory()
    result = measure_shadow(roi, "down", k=1.0)

    assert result.available is False
    assert result.reason is not None
    assert "no acoustic shadow found" in result.reason
    # honesty: never populate a length when unavailable
    assert result.shadow_length_px is None
    assert result.shadow_start_px is None
    assert result.shadow_end_px is None


def test_empty_roi_reports_unavailable_not_crash():
    empty = np.zeros((0, 0), dtype=np.uint8)
    result = measure_shadow(empty, "down", k=1.0)
    assert result.available is False
    assert "too small" in result.reason


def test_speckle_is_suppressed_by_along_track_averaging(synthetic_roi_factory):
    """A profile built from a wide along-track strip with independent
    per-pixel speckle should still recover the shadow cleanly, proving the
    along-track mean actually suppresses noise rather than just working on
    a single row/column."""
    roi = synthetic_roi_factory(shadow_length_px=25, shadow_direction="down", along_track=80, speckle_std=15.0)
    result = measure_shadow(roi, "down", k=1.0)
    assert result.available, result.reason
    assert abs(result.shadow_length_px - 25) <= 4
