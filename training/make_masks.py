"""
Generate weakly-supervised 3-class segmentation masks from crab-pot detection boxes.

WHY THIS EXISTS
No public side-scan dataset ships highlight/shadow segmentation masks, and no
published model segments those classes (the sonar segmentation literature targets
seafloor SUBSTRATE - rock/sand/mud - which is a different problem). So there is
nothing to download and nothing to fine-tune from. We derive labels from the
detection boxes we do have.

METHOD - and its limits, stated plainly
For each annotated box we take a padded ROI and label:
    class 0  background / seabed reverberation
    class 1  highlight   - the bright target return, inside the box
    class 2  shadow      - the dark run immediately down-range of the box

Thresholds are per-ROI percentiles of the seabed statistics, so they adapt to
local gain rather than assuming a global intensity scale.

These are WEAK labels. They are produced by thresholding, not by a human, and
they will be wrong on low-contrast targets, on overlapping shadows, and wherever
the seabed itself is dark. A model trained on them inherits those errors. This
is a bootstrap to get a working segmenter, not a substitute for annotation, and
anything reporting results from it must say "weakly supervised".

Output layout (PNG masks with pixel values 0/1/2):
    data/unet_masks/{train,val}/images/*.png     ROI crop, greyscale
    data/unet_masks/{train,val}/masks/*.png      class map
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "crabpot_yolo"   # overridden by --src
DST = ROOT / "data" / "unet_masks"

BACKGROUND, HIGHLIGHT, SHADOW = 0, 1, 2

ROI_PAD = 0.35          # more than inference padding: we must contain the shadow
SHADOW_EXTENT = 2.5     # search this many box-heights down-range for shadow
OUT_SIZE = 256

# A target must be this much brighter than the local seabed to count as
# highlight, and a shadow this much darker. Expressed in standard deviations
# so they adapt to local gain.
HIGHLIGHT_K = 1.0
SHADOW_K = 1.0

MIN_HIGHLIGHT_PX = 20   # reject ROIs where thresholding found essentially nothing
MIN_SHADOW_PX = 20


def load_boxes(label_path: Path, W: int, H: int) -> list[tuple[int, int, int, int]]:
    """YOLO label file -> absolute xyxy boxes."""
    if not label_path.exists():
        return []
    out = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, xc, yc, bw, bh = (float(p) for p in parts)
        x1 = int(round((xc - bw / 2) * W))
        y1 = int(round((yc - bh / 2) * H))
        x2 = int(round((xc + bw / 2) * W))
        y2 = int(round((yc + bh / 2) * H))
        if x2 > x1 and y2 > y1:
            out.append((x1, y1, x2, y2))
    return out


def build_mask(gray: np.ndarray, box: tuple[int, int, int, int]):
    """Return (roi_crop, class_map) or None if the ROI yields no usable labels.

    Down-range is taken as +y (increasing row), matching SHADOW_DIRECTION=down.
    """
    H, W = gray.shape
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1

    padx = int(bw * ROI_PAD)
    # Asymmetric in y: little above the target, a lot below it, because that is
    # where the shadow lives. A symmetric crop routinely cuts the shadow off.
    pad_up = int(bh * ROI_PAD)
    pad_dn = int(bh * SHADOW_EXTENT)

    rx1 = max(0, x1 - padx)
    ry1 = max(0, y1 - pad_up)
    rx2 = min(W, x2 + padx)
    ry2 = min(H, y2 + pad_dn)
    if rx2 - rx1 < 16 or ry2 - ry1 < 16:
        return None

    roi = gray[ry1:ry2, rx1:rx2]
    cls = np.full(roi.shape, BACKGROUND, dtype=np.uint8)

    # Seabed reference: the ROI minus the target box and minus the down-range
    # strip, i.e. the parts we believe are plain seabed.
    bx1, by1 = x1 - rx1, y1 - ry1
    bx2, by2 = x2 - rx1, y2 - ry1
    ref = np.ones(roi.shape, dtype=bool)
    ref[by1:by2, bx1:bx2] = False
    ref[by2:, bx1:bx2] = False
    if ref.sum() < 50:
        return None

    mean = float(roi[ref].mean())
    std = float(roi[ref].std()) or 1.0

    # Highlight: bright pixels inside the detection box.
    box_region = roi[by1:by2, bx1:bx2]
    hl = box_region > (mean + HIGHLIGHT_K * std)
    cls[by1:by2, bx1:bx2][hl] = HIGHLIGHT

    # Shadow: dark pixels in the down-range strip beneath the box.
    strip = roi[by2:, bx1:bx2]
    if strip.size:
        sh = strip < (mean - SHADOW_K * std)
        cls[by2:, bx1:bx2][sh] = SHADOW

    if (cls == HIGHLIGHT).sum() < MIN_HIGHLIGHT_PX:
        return None
    if (cls == SHADOW).sum() < MIN_SHADOW_PX:
        return None

    # Clean up speckle: opening removes isolated pixels, closing fills pinholes.
    k = np.ones((3, 3), np.uint8)
    for c in (HIGHLIGHT, SHADOW):
        m = (cls == c).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        cls[cls == c] = BACKGROUND
        cls[m.astype(bool)] = c

    roi_out = cv2.resize(roi, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_LINEAR)
    cls_out = cv2.resize(cls, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_NEAREST)
    return roi_out, cls_out


def process_split(split: str, limit: int | None) -> dict:
    img_dir = SRC / split / "images"
    lbl_dir = SRC / split / "labels"
    out_img = DST / split / "images"
    out_msk = DST / split / "masks"
    out_img.mkdir(parents=True, exist_ok=True)
    out_msk.mkdir(parents=True, exist_ok=True)

    made = rejected = 0
    hl_frac: list[float] = []
    sh_frac: list[float] = []

    files = sorted(img_dir.glob("*.jpg"))
    for img_path in files:
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        H, W = gray.shape
        for i, box in enumerate(load_boxes(lbl_dir / f"{img_path.stem}.txt", W, H)):
            built = build_mask(gray, box)
            if built is None:
                rejected += 1
                continue
            roi, cls = built
            stem = f"{img_path.stem}_{i}"
            cv2.imwrite(str(out_img / f"{stem}.png"), roi)
            cv2.imwrite(str(out_msk / f"{stem}.png"), cls)
            made += 1
            hl_frac.append(float((cls == HIGHLIGHT).mean()))
            sh_frac.append(float((cls == SHADOW).mean()))
            if limit and made >= limit:
                break
        if limit and made >= limit:
            break

    return {
        "split": split,
        "made": made,
        "rejected": rejected,
        "mean_highlight_frac": round(float(np.mean(hl_frac)), 4) if hl_frac else 0.0,
        "mean_shadow_frac": round(float(np.mean(sh_frac)), 4) if sh_frac else 0.0,
    }


def main() -> None:
    global SRC, HIGHLIGHT_K, SHADOW_K
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap masks per split (for a quick pass)")
    ap.add_argument("--src", default=None, help="YOLO-format dataset dir (default: crabpot_yolo)")
    ap.add_argument("--hl-k", type=float, default=HIGHLIGHT_K)
    ap.add_argument("--sh-k", type=float, default=SHADOW_K)
    args = ap.parse_args()

    if args.src:
        SRC = ROOT / "data" / args.src
    HIGHLIGHT_K, SHADOW_K = args.hl_k, args.sh_k
    print(f"source={SRC.name}  highlight_k={HIGHLIGHT_K}  shadow_k={SHADOW_K}")

    for s in ("train", "val"):
        r = process_split(s, args.limit)
        print(
            f"{r['split']:<6} made={r['made']:<6} rejected={r['rejected']:<6} "
            f"mean highlight={r['mean_highlight_frac']:.3f}  mean shadow={r['mean_shadow_frac']:.3f}"
        )
    print(f"\nwrote {DST}")
    print("NOTE: these are WEAK labels from thresholding, not human annotation.")


if __name__ == "__main__":
    main()
