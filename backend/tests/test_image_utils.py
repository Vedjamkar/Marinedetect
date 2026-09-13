"""Safe image loading, streaming upload size enforcement, and padded ROI
crop with down-range-direction padding."""
import io

import numpy as np
import pytest
from PIL import Image

from backend.utils.image_utils import (
    ImageLoadError,
    UploadTooLargeError,
    crop_padded_roi,
    load_image_safe,
    save_upload_streaming,
)


def test_upload_streaming_aborts_mid_write(tmp_path):
    big = io.BytesIO(b"x" * (5 * 1024 * 1024))  # 5 MB
    dest = tmp_path / "upload.bin"
    with pytest.raises(UploadTooLargeError):
        save_upload_streaming(big, dest, max_bytes=1 * 1024 * 1024)
    # Partial file must be cleaned up, not left on disk.
    assert not dest.exists()


def test_upload_streaming_succeeds_under_limit(tmp_path):
    data = io.BytesIO(b"y" * (1024))
    dest = tmp_path / "upload.bin"
    written = save_upload_streaming(data, dest, max_bytes=1024 * 1024)
    assert written == 1024
    assert dest.exists()
    assert dest.stat().st_size == 1024


def test_16bit_tiff_is_percentile_normalized_not_naively_cast(tmp_path):
    # A 16-bit image whose values live entirely in a narrow high band. A naive
    # cast to uint8 (>>8 or clip) would crush this to near-uniform gray;
    # percentile normalization should spread it across the 0-255 range.
    arr16 = np.random.default_rng(0).integers(60000, 65000, size=(50, 50), dtype=np.uint16)
    path = tmp_path / "sonar16.tiff"
    Image.fromarray(arr16, mode="I;16").save(path)

    gray, rgb = load_image_safe(path, max_pixels=80_000_000)
    assert gray.dtype == np.uint8
    assert gray.shape == (50, 50)
    # Normalized output should use a wide dynamic range, not be crushed near one value.
    assert gray.max() - gray.min() > 100
    assert rgb.shape == (50, 50, 3)


def test_decompression_bomb_is_rejected(tmp_path):
    path = tmp_path / "huge.png"
    Image.new("L", (2000, 2000)).save(path)
    with pytest.raises(ImageLoadError):
        load_image_safe(path, max_pixels=1000)  # far below 2000*2000


def test_unreadable_file_reports_clean_error(tmp_path):
    path = tmp_path / "not_an_image.png"
    path.write_bytes(b"this is not image data")
    with pytest.raises(ImageLoadError):
        load_image_safe(path, max_pixels=80_000_000)


@pytest.mark.parametrize(
    "shadow_direction,expected_extra_side",
    [("down", "bottom"), ("up", "top"), ("right", "right"), ("left", "left")],
)
def test_roi_padding_extra_on_down_range_side(shadow_direction, expected_extra_side):
    gray = np.zeros((200, 200), dtype=np.uint8)
    bbox = (80.0, 80.0, 120.0, 120.0)  # 40x40 box, centered
    pad_ratio = 0.25

    roi_none, bbox_none = crop_padded_roi(gray, bbox, pad_ratio, "none")
    roi_dir, bbox_dir = crop_padded_roi(gray, bbox, pad_ratio, shadow_direction)

    nx1, ny1, nx2, ny2 = bbox_none
    dx1, dy1, dx2, dy2 = bbox_dir

    if expected_extra_side == "bottom":
        assert dy2 > ny2  # more room below
        assert dy1 == ny1 and dx1 == nx1 and dx2 == nx2
    elif expected_extra_side == "top":
        assert dy1 < ny1
        assert dy2 == ny2 and dx1 == nx1 and dx2 == nx2
    elif expected_extra_side == "right":
        assert dx2 > nx2
        assert dx1 == nx1 and dy1 == ny1 and dy2 == ny2
    elif expected_extra_side == "left":
        assert dx1 < nx1
        assert dx2 == nx2 and dy1 == ny1 and dy2 == ny2


def test_roi_padding_clamped_to_image_bounds():
    gray = np.zeros((50, 50), dtype=np.uint8)
    bbox = (0.0, 0.0, 10.0, 10.0)
    roi, (x1, y1, x2, y2) = crop_padded_roi(gray, bbox, 0.5, "down")
    assert x1 >= 0 and y1 >= 0
    assert x2 <= 50 and y2 <= 50
    assert roi.shape == (y2 - y1, x2 - x1)
