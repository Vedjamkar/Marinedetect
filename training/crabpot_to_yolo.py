"""
Convert the crab-pot side-scan dataset (HF imagefolder + metadata.jsonl) to YOLO format.

Source layout:
    data/raw/crabpot/{train,valid,test}/
        *.jpg
        metadata.jsonl   one row per image:
            {"file_name": "...", "objects": {"bbox": [[x, y, w, h], ...],
                                             "category": ["Crab-Pot", ...],
                                             "area": [...]}}

bbox is COCO-style: absolute pixels, [x_min, y_min, width, height].
YOLO wants: <cls> <x_center> <y_center> <width> <height>, all normalized to
the image's real dimensions (read from the file, never trusted from metadata).

The dataset ships its own train/valid/test split, so unlike SCTD we do not
invent one — using the authors' split keeps our numbers comparable to theirs.

Attribution: PINGEcosystem/sss-crab-pot-detection-ds, CC-BY-SA-4.0. See CREDITS.md.
"""
from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "raw" / "crabpot"
DST = ROOT / "data" / "crabpot_yolo"

# Fixed class order. Index is the YOLO class id and must not be reordered
# later without retraining — the ids are baked into every label file.
# The dataset documents a "Maybe-Crab-Pot" class, but a full pass over all three
# splits found ZERO instances of it. Declaring an unused class makes per-class mAP
# report NaN for it and drags the mean, so this is treated as single-class.
CLASSES = ["Crab-Pot"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

# The dataset's own split names -> the names Ultralytics expects.
SPLIT_MAP = {"train": "train", "valid": "val", "test": "test"}


def convert_split(src_split: str, dst_split: str) -> dict:
    src_dir = SRC / src_split
    meta_path = src_dir / "metadata.jsonl"
    if not meta_path.exists():
        raise FileNotFoundError(f"missing {meta_path}")

    img_out = DST / dst_split / "images"
    lbl_out = DST / dst_split / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    counts: Counter = Counter()
    n_images = 0
    n_boxes = 0
    skipped_degenerate = 0
    skipped_missing = 0
    unknown_categories: Counter = Counter()

    with meta_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            fname = row["file_name"]
            src_img = src_dir / fname
            if not src_img.exists():
                skipped_missing += 1
                continue

            # Read the real dimensions off the file rather than trusting metadata.
            with Image.open(src_img) as im:
                W, H = im.size
            if W <= 0 or H <= 0:
                skipped_missing += 1
                continue

            objects = row.get("objects") or {}
            bboxes = objects.get("bbox") or []
            cats = objects.get("category") or []

            lines: list[str] = []
            for bbox, cat in zip(bboxes, cats):
                if cat not in CLASS_TO_ID:
                    unknown_categories[cat] += 1
                    continue
                x, y, w, h = (float(v) for v in bbox)

                # Clamp to the image before normalizing; a box running off the
                # edge is real data, a box of zero area is not.
                x1 = max(0.0, min(x, W))
                y1 = max(0.0, min(y, H))
                x2 = max(0.0, min(x + w, W))
                y2 = max(0.0, min(y + h, H))
                bw, bh = x2 - x1, y2 - y1
                if bw <= 1e-6 or bh <= 1e-6:
                    skipped_degenerate += 1
                    continue

                xc = (x1 + x2) / 2.0 / W
                yc = (y1 + y2) / 2.0 / H
                lines.append(
                    f"{CLASS_TO_ID[cat]} {xc:.6f} {yc:.6f} {bw / W:.6f} {bh / H:.6f}"
                )
                counts[cat] += 1
                n_boxes += 1

            stem = Path(fname).stem
            shutil.copy(src_img, img_out / fname)
            # An image with no boxes still gets an empty label file — YOLO reads
            # that as a valid background image, which is useful negative signal.
            (lbl_out / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
            n_images += 1

    return {
        "split": dst_split,
        "images": n_images,
        "boxes": n_boxes,
        "per_class": dict(counts),
        "skipped_degenerate": skipped_degenerate,
        "skipped_missing": skipped_missing,
        "unknown_categories": dict(unknown_categories),
    }


def main() -> None:
    if DST.exists():
        shutil.rmtree(DST)

    summaries = [convert_split(s, d) for s, d in SPLIT_MAP.items()]

    yaml_path = ROOT / "training" / "crabpot.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {DST.as_posix()}",
                "train: train/images",
                "val: val/images",
                "test: test/images",
                "",
                f"nc: {len(CLASSES)}",
                "names:",
                *[f"  {i}: {c}" for i, c in enumerate(CLASSES)],
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"{'split':<8}{'images':>8}{'boxes':>8}   per-class")
    for s in summaries:
        print(f"{s['split']:<8}{s['images']:>8}{s['boxes']:>8}   {s['per_class']}")
        if s["skipped_degenerate"]:
            print(f"         skipped {s['skipped_degenerate']} zero-area boxes")
        if s["skipped_missing"]:
            print(f"         skipped {s['skipped_missing']} rows with no image file")
        if s["unknown_categories"]:
            print(f"         UNKNOWN CATEGORIES: {s['unknown_categories']}")
    print(f"\nwrote {yaml_path}")


if __name__ == "__main__":
    main()
