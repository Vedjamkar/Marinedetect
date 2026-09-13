"""
Safe image loading and ROI cropping.

- Safe load for PNG/JPG/TIFF with a decompression-bomb pixel cap.
- 16-bit imagery (common for sonar TIFFs) is percentile-normalized to 8-bit,
  never naively cast (naive cast to uint8 clips/crushes most of the dynamic
  range of 16-bit sonar returns).
- Upload size is enforced while the file streams to disk, aborting the write
  mid-stream, never after the whole file has already landed.
- Padded ROI crop pads more heavily on the down-range side (SHADOW_DIRECTION)
  because that is where the acoustic shadow lives (PLAN.md 1.2) — cropping
  it off makes object height uncomputable.
"""
from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError


class UploadTooLargeError(Exception):
    def __init__(self, max_bytes: int, received_bytes: int):
        self.max_bytes = max_bytes
        self.received_bytes = received_bytes
        super().__init__(
            f"Upload exceeds max size of {max_bytes} bytes (received at least {received_bytes})."
        )


class ImageLoadError(Exception):
    """Raised for any image that cannot be safely decoded/loaded."""


def save_upload_streaming(file_obj: BinaryIO, dest_path: Path, max_bytes: int, chunk_size: int = 1024 * 1024) -> int:
    """Copy a file-like object to dest_path, aborting mid-write if it exceeds max_bytes.

    Deletes the partial file on abort. Returns total bytes written on success.
    """
    total = 0
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(dest_path, "wb") as out:
            while True:
                chunk = file_obj.read(chunk_size)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise UploadTooLargeError(max_bytes, total)
                out.write(chunk)
    except UploadTooLargeError:
        dest_path.unlink(missing_ok=True)
        raise
    except Exception:
        dest_path.unlink(missing_ok=True)
        raise
    return total


def load_image_safe(path: Path, max_pixels: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load an image file, guarding against decompression bombs.

    Returns (gray_uint8, rgb_uint8) as numpy arrays: gray is HxW (used for
    intensity-profile analysis), rgb is HxWx3 (used for annotated output).

    16-bit single-channel images (mode I / I;16*) are percentile-normalized
    (2nd/98th percentile -> 0-255) rather than naively truncated, so the
    dynamic range of real sonar returns survives.
    """
    prior_cap = Image.MAX_IMAGE_PIXELS
    try:
        Image.MAX_IMAGE_PIXELS = max_pixels
        try:
            with Image.open(path) as im:
                im.load()
                mode = im.mode
                if mode in ("I", "I;16", "I;16B", "I;16L", "I;16N"):
                    arr = np.asarray(im).astype(np.float64)
                    lo, hi = np.percentile(arr, [2.0, 98.0])
                    if hi <= lo:
                        hi = lo + 1.0
                    norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
                    gray = (norm * 255.0).astype(np.uint8)
                    rgb = np.stack([gray, gray, gray], axis=-1)
                else:
                    rgb = np.asarray(im.convert("RGB")).astype(np.uint8)
                    gray = np.asarray(im.convert("L")).astype(np.uint8)
                return gray, rgb
        except Image.DecompressionBombError as exc:
            raise ImageLoadError(f"Image exceeds the {max_pixels}-pixel safety cap: {exc}") from exc
        except UnidentifiedImageError as exc:
            raise ImageLoadError(f"Could not identify image format for {path.name}: {exc}") from exc
    finally:
        Image.MAX_IMAGE_PIXELS = prior_cap


def crop_padded_roi(
    gray: np.ndarray,
    bbox: Tuple[float, float, float, float],
    pad_ratio: float,
    shadow_direction: str = "none",
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Crop a bbox out of `gray` with symmetric padding, plus extra padding on
    the down-range side named by `shadow_direction` (down/up/right/left/none).

    Returns (roi_array, (x1, y1, x2, y2)) in absolute image pixel coordinates,
    clamped to image bounds.
    """
    H, W = gray.shape[:2]
    x1, y1, x2, y2 = bbox
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)
    pad_x = w * pad_ratio
    pad_y = h * pad_ratio

    left, right, top, bottom = pad_x, pad_x, pad_y, pad_y
    if shadow_direction == "down":
        bottom += pad_y
    elif shadow_direction == "up":
        top += pad_y
    elif shadow_direction == "right":
        right += pad_x
    elif shadow_direction == "left":
        left += pad_x
    # "none" -> symmetric padding only, no extra.

    nx1 = int(max(0, np.floor(x1 - left)))
    ny1 = int(max(0, np.floor(y1 - top)))
    nx2 = int(min(W, np.ceil(x2 + right)))
    ny2 = int(min(H, np.ceil(y2 + bottom)))

    roi = gray[ny1:ny2, nx1:nx2]
    return roi, (nx1, ny1, nx2, ny2)
