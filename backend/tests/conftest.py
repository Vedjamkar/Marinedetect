"""
Hermetic test fixtures. Every test in this suite must pass with zero model
weights on disk — fixtures are synthetic, generated with numpy.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Make `backend` importable when pytest is run from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Point the weight paths at a location that cannot exist, BEFORE backend.config
# is imported and reads the environment.
#
# Several tests assert the zero-weights behaviour: model-status must report
# unavailable with a precise reason, and inference must return a clean 503
# rather than a stack trace. Without this the suite passes on a fresh clone and
# fails the moment someone trains a model, which trains people to ignore red
# tests. Isolating here makes the result independent of what is on disk.
_NOWHERE = Path(__file__).resolve().parent / "_no_such_weights_dir"
os.environ["YOLO_MODEL_PATH"] = str(_NOWHERE / "best.pt")
# The detector pool reads the plural form; both must point nowhere or the
# zero-weights tests silently start passing against real weights.
os.environ["YOLO_MODEL_PATHS"] = str(_NOWHERE / "best.pt")
os.environ["UNET_MODEL_PATH"] = str(_NOWHERE / "unet.pth")

from backend.services.shadow_service import axis_and_sign  # noqa: E402


def _build_canonical_profile_1d(
    seabed_before: int,
    highlight_width: int,
    shadow_length_px: int,
    seabed_after: int,
    rng: np.random.Generator,
    seabed_mean: float,
    seabed_std: float,
    highlight_value: float,
    shadow_value: float,
) -> tuple[np.ndarray, int, int]:
    n = seabed_before + highlight_width + shadow_length_px + seabed_after
    profile = rng.normal(seabed_mean, seabed_std, n)
    profile[seabed_before: seabed_before + highlight_width] = highlight_value
    s0 = seabed_before + highlight_width
    s1 = s0 + shadow_length_px
    profile[s0:s1] = shadow_value
    return profile, s0, s1


def make_synthetic_roi(
    shadow_length_px: int,
    shadow_direction: str = "down",
    along_track: int = 40,
    seabed_before: int = 15,
    highlight_width: int = 8,
    seabed_after: int = 15,
    seabed_mean: float = 130.0,
    seabed_std: float = 4.0,
    highlight_value: float = 235.0,
    shadow_value: float = 15.0,
    speckle_std: float = 2.0,
    seed: int = 0,
) -> np.ndarray:
    """Synthetic side-scan ROI: gradient/noisy seabed + bright highlight blob
    + dark shadow of exactly `shadow_length_px` immediately down-range of it
    (down-range direction per `shadow_direction`, same convention as
    shadow_service.axis_and_sign). Along-track axis carries independent
    per-pixel speckle so mean-profile averaging is genuinely exercised.
    """
    rng = np.random.default_rng(seed)
    canonical, _, _ = _build_canonical_profile_1d(
        seabed_before, highlight_width, shadow_length_px, seabed_after,
        rng, seabed_mean, seabed_std, highlight_value, shadow_value,
    )
    axis, sign = axis_and_sign(shadow_direction)
    raw_1d = canonical if sign == 1 else canonical[::-1]

    if axis == 0:
        base = np.tile(raw_1d.reshape(-1, 1), (1, along_track))
    else:
        base = np.tile(raw_1d.reshape(1, -1), (along_track, 1))

    noise = rng.normal(0.0, speckle_std, base.shape)
    arr = np.clip(base + noise, 0, 255).astype(np.uint8)
    return arr


def make_uniform_seabed_roi(
    along_track: int = 40,
    range_len: int = 60,
    seabed_mean: float = 130.0,
    seabed_std: float = 4.0,
    seed: int = 0,
) -> np.ndarray:
    """A flat seabed ROI with no target/shadow at all — used to assert the
    shadow measurement honestly reports unavailable rather than inventing one.
    """
    rng = np.random.default_rng(seed)
    arr = np.clip(rng.normal(seabed_mean, seabed_std, (range_len, along_track)), 0, 255).astype(np.uint8)
    return arr


@pytest.fixture
def synthetic_roi_factory():
    return make_synthetic_roi


@pytest.fixture
def uniform_seabed_roi_factory():
    return make_uniform_seabed_roi
