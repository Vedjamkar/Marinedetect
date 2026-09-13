"""
Convert the SCTD Pascal VOC annotations into YOLO format, with a seeded
stratified 80/20 train/val split.

Source:  data/raw/SCTD/SCTD/{JPEGImages,Annotations}/
Output:  data/yolo/{train,val}/{images,labels}/

Notes / traps handled (see TRAINING.md):
 - Only <object><name> is read as a class label. <owner><name> (e.g.
   "ChaojieZhu") is NOT a class and must not be picked up by a naive
   <name> grep.
 - <folder>/<source> "UAV autolanding" boilerplate is ignored.
 - Image dimensions are read from the actual JPEG file (via OpenCV), not
   trusted from the XML <size> block; mismatches are logged.
 - Boxes are clamped to [0, 1] after normalization; degenerate boxes
   (zero width or height post-clamp) are skipped and logged.
 - The train/val split is stratified on each image's *dominant* class
   (the class with the most instances in that image), seeded with
   random_state=42, so the rare 'human' class (35 instances total) is
   represented in both splits rather than risking exclusion from val
   under a purely random image-level split.
"""
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import cv2

SEED = 42
VAL_FRACTION = 0.2

CLASSES = ["ship", "aircraft", "human"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}

SRC_ROOT = ROOT / "data" / "raw" / "SCTD" / "SCTD"
IMAGES_DIR = SRC_ROOT / "JPEGImages"
ANNOTATIONS_DIR = SRC_ROOT / "Annotations"

OUT_ROOT = ROOT / "data" / "yolo"


def parse_annotation(xml_path: Path):
    """Return (list of (class_name, xmin, ymin, xmax, ymax), xml_width, xml_height)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_el = root.find("size")
    xml_w = int(size_el.findtext("width")) if size_el is not None else None
    xml_h = int(size_el.findtext("height")) if size_el is not None else None

    objects = []
    # Only iterate direct <object> children of <annotation>; only read
    # <name> that lives inside <object>. This deliberately does NOT do a
    # blanket root.iter("name") which would also match <owner><name>.
    for obj in root.findall("object"):
        cls_name = obj.findtext("name")
        if cls_name is None:
            continue
        cls_name = cls_name.strip()
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        xmin = float(bnd.findtext("xmin"))
        ymin = float(bnd.findtext("ymin"))
        xmax = float(bnd.findtext("xmax"))
        ymax = float(bnd.findtext("ymax"))
        objects.append((cls_name, xmin, ymin, xmax, ymax))

    return objects, xml_w, xml_h


def main():
    random.seed(SEED)

    xml_files = sorted(ANNOTATIONS_DIR.glob("*.xml"))
    print(f"Found {len(xml_files)} annotation files")

    skipped_boxes = []
    unknown_classes = Counter()
    dim_mismatches = []

    # image_id -> list of (cls_id, xc, yc, w, h)
    per_image_labels = {}
    # image_id -> jpg path
    per_image_path = {}
    # class counts per image, for dominant-class stratification
    per_image_class_counts = {}

    total_obj_seen = 0
    total_obj_kept = 0

    for xml_path in xml_files:
        image_id = xml_path.stem
        img_path = IMAGES_DIR / f"{image_id}.jpg"
        if not img_path.exists():
            print(f"WARNING: no image for annotation {xml_path.name}, skipping")
            continue

        objects, xml_w, xml_h = parse_annotation(xml_path)

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"WARNING: could not read image {img_path}, skipping")
            continue
        real_h, real_w = img.shape[:2]

        if xml_w is not None and xml_h is not None and (xml_w != real_w or xml_h != real_h):
            dim_mismatches.append((image_id, (xml_w, xml_h), (real_w, real_h)))

        yolo_lines = []
        class_counts = Counter()

        for cls_name, xmin, ymin, xmax, ymax in objects:
            total_obj_seen += 1
            if cls_name not in CLASS_TO_ID:
                unknown_classes[cls_name] += 1
                print(f"WARNING: unknown class '{cls_name}' in {xml_path.name}, skipping object")
                continue

            # Normalize against the REAL image dimensions.
            xmin_n = xmin / real_w
            xmax_n = xmax / real_w
            ymin_n = ymin / real_h
            ymax_n = ymax / real_h

            # Clamp to [0, 1]
            xmin_n = min(max(xmin_n, 0.0), 1.0)
            xmax_n = min(max(xmax_n, 0.0), 1.0)
            ymin_n = min(max(ymin_n, 0.0), 1.0)
            ymax_n = min(max(ymax_n, 0.0), 1.0)

            w_n = xmax_n - xmin_n
            h_n = ymax_n - ymin_n

            if w_n <= 0 or h_n <= 0:
                skipped_boxes.append((xml_path.name, cls_name, (xmin, ymin, xmax, ymax)))
                print(
                    f"SKIP degenerate box in {xml_path.name}: class={cls_name} "
                    f"raw=({xmin},{ymin},{xmax},{ymax}) -> normalized w={w_n:.4f} h={h_n:.4f}"
                )
                continue

            xc_n = xmin_n + w_n / 2
            yc_n = ymin_n + h_n / 2

            cls_id = CLASS_TO_ID[cls_name]
            yolo_lines.append((cls_id, xc_n, yc_n, w_n, h_n))
            class_counts[cls_name] += 1
            total_obj_kept += 1

        if not yolo_lines:
            # Image has no valid objects after filtering; still possible
            # to include as a background image, but with 357 images we
            # keep it simple and only include images with >=1 valid box.
            print(f"NOTE: {image_id} has no valid objects after filtering, excluding from dataset")
            continue

        per_image_labels[image_id] = yolo_lines
        per_image_path[image_id] = img_path
        per_image_class_counts[image_id] = class_counts

    print()
    print(f"Objects seen: {total_obj_seen}, kept: {total_obj_kept}, "
          f"skipped degenerate: {len(skipped_boxes)}, unknown-class: {sum(unknown_classes.values())}")
    if dim_mismatches:
        print(f"\n{len(dim_mismatches)} image(s) with XML size != real image size:")
        for image_id, xml_dim, real_dim in dim_mismatches:
            print(f"  {image_id}: xml={xml_dim} real={real_dim}")
    if unknown_classes:
        print(f"\nUnknown class labels encountered: {dict(unknown_classes)}")

    image_ids = sorted(per_image_labels.keys())
    print(f"\nUsable images: {len(image_ids)}")

    # --- Stratified split on dominant class per image ---
    def dominant_class(image_id):
        counts = per_image_class_counts[image_id]
        # Deterministic tie-break: prefer fixed CLASSES order.
        best_cls = None
        best_count = -1
        for cls_name in CLASSES:
            c = counts.get(cls_name, 0)
            if c > best_count:
                best_count = c
                best_cls = cls_name
        return best_cls

    by_dominant = defaultdict(list)
    for image_id in image_ids:
        by_dominant[dominant_class(image_id)].append(image_id)

    train_ids = []
    val_ids = []
    rng = random.Random(SEED)

    for cls_name in CLASSES:
        ids = sorted(by_dominant.get(cls_name, []))
        rng.shuffle(ids)
        n_val = max(1, round(len(ids) * VAL_FRACTION)) if ids else 0
        val_ids.extend(ids[:n_val])
        train_ids.extend(ids[n_val:])

    train_ids = sorted(train_ids)
    val_ids = sorted(val_ids)

    print(f"\nSplit: {len(train_ids)} train images, {len(val_ids)} val images "
          f"(stratified on dominant class, seed={SEED})")

    # --- Write output ---
    for split_name, ids in (("train", train_ids), ("val", val_ids)):
        img_out = OUT_ROOT / split_name / "images"
        lbl_out = OUT_ROOT / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for image_id in ids:
            src_img = per_image_path[image_id]
            dst_img = img_out / src_img.name
            shutil.copyfile(src_img, dst_img)

            lbl_path = lbl_out / f"{image_id}.txt"
            with open(lbl_path, "w") as f:
                for cls_id, xc, yc, w, h in per_image_labels[image_id]:
                    f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    # --- Summary table: per-class instance counts in train/val ---
    def instance_counts(ids):
        counts = Counter()
        for image_id in ids:
            for cls_id, *_ in per_image_labels[image_id]:
                counts[CLASSES[cls_id]] += 1
        return counts

    train_counts = instance_counts(train_ids)
    val_counts = instance_counts(val_ids)

    print("\n=== Per-class instance counts ===")
    header = f"{'class':<10} {'train':>8} {'val':>8} {'total':>8}"
    print(header)
    print("-" * len(header))
    grand_train = grand_val = 0
    for cls_name in CLASSES:
        t = train_counts.get(cls_name, 0)
        v = val_counts.get(cls_name, 0)
        grand_train += t
        grand_val += v
        print(f"{cls_name:<10} {t:>8} {v:>8} {t + v:>8}")
    print("-" * len(header))
    print(f"{'TOTAL':<10} {grand_train:>8} {grand_val:>8} {grand_train + grand_val:>8}")

    print(f"\nImages -> train: {len(train_ids)}, val: {len(val_ids)}, total: {len(train_ids) + len(val_ids)}")
    print(f"Output written to: {OUT_ROOT}")


if __name__ == "__main__":
    main()
