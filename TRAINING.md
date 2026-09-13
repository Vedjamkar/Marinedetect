# YOLO Training — Handover Spec

Owner: Sonnet agent. Architecture and constraints fixed by this document; do not renegotiate them.

## Ground truth about the environment (verified 2026-09-01)

| Item | Value |
|---|---|
| GPU | GTX 1650 Ti, **4 GB VRAM**, driver 610.47 |
| venv | `D:\Marinedetect\.venv` (Python 3.11.15, created with uv) |
| venv python | `D:/Marinedetect/.venv/Scripts/python.exe` |
| torch | installing from `https://download.pytorch.org/whl/cu124` — verify before use |
| ultralytics / opencv | **not installed** — install into the venv |
| `python` on PATH | Hermes agent venv, no pip. **Never use it.** Always the explicit venv path. |

Install remaining deps with:

```bash
VIRTUAL_ENV=/d/Marinedetect/.venv uv pip install --python .venv/Scripts/python.exe ultralytics opencv-python
```

Verify CUDA is actually live before training:

```bash
/d/Marinedetect/.venv/Scripts/python.exe -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If `cuda.is_available()` is False, stop and report. Do not silently train on CPU.

## The data

`D:\Marinedetect\data\raw\SCTD\SCTD\` — Pascal VOC layout, already extracted.

- `JPEGImages/` — 357 `.jpg`
- `Annotations/` — 357 `.xml`
- `ImageSets/Main/trainval.txt` — all ids, **no train/val split provided**

Verified contents: **357 images, 363 objects, 3 classes** — `ship` 271, `aircraft` 57, `human` 35. Median image size 512×244.

Two traps in the XML:
1. `<owner><name>ChaojieZhu</name></owner>` — a naive `<name>` grep picks this up as a class. Only read `<name>` **inside `<object>`**.
2. `<folder>` and `<source>` carry "UAV autolanding" boilerplate from a template. Ignore them.

## What to build

### 1. `training/voc_to_yolo.py`

Convert VOC → YOLO format. Requirements:

- Parse only `object/name` for classes. Class order fixed: `["ship", "aircraft", "human"]` → ids 0, 1, 2.
- YOLO label line: `<cls> <xc> <yc> <w> <h>`, all normalized to image width/height.
- **Clamp** boxes to `[0, 1]` and skip degenerate ones (zero width or height) — log every skip with the filename.
- Read actual image dimensions from the file, not from the XML `<size>` block. Verify they agree; log mismatches.
- **Stratified** 80/20 train/val split on the image's dominant class, seeded (`random_state=42`) so it is reproducible. With `human` at only 35 instances, a random split can starve the val set.
- Write to `data/yolo/{train,val}/{images,labels}/`.
- Print a summary table: per-class instance counts in train and in val.

### 2. `training/data.yaml`

Standard Ultralytics format pointing at `data/yolo`, `nc: 3`, the class names in the order above.

### 3. `training/train.py`

```
model:    yolov8n.pt        (COCO-pretrained starting point)
imgsz:    640
batch:    8                 drop to 4 if CUDA OOM — do not raise it
epochs:   150               tiny dataset, needs the passes
patience: 40                early stop
amp:      True              required at 4 GB
device:   0
project:  runs/  name: sctd_yolov8n
seed:     42
```

Augmentation — sonar-appropriate, not photographic defaults:

- `fliplr: 0.5` — port/starboard mirroring is physically meaningful
- `flipud: 0.0` — **must stay 0.** Vertical flip inverts the range axis and puts shadows up-range, which is physically impossible. Do not enable it.
- `hsv_h: 0.0`, `hsv_s: 0.0` — imagery is effectively greyscale intensity; hue/saturation jitter is meaningless
- `hsv_v: 0.4` — brightness jitter stands in for gain and TVG variation
- `degrees: 5.0` — small only; sonar geometry is not rotation-invariant
- `scale: 0.5`, `translate: 0.1`
- `mosaic: 1.0` with `close_mosaic: 20`

### 4. Evaluate and report

Run validation on the held-out split. Write `training/RESULTS.md` containing:

- mAP@0.5 and mAP@0.5:0.95, overall and **per class**
- precision, recall, and the confusion matrix
- number of train/val images and per-class instance counts
- exact training config used, and wall-clock time
- the run directory path

Copy best weights to `weights/yolo/best.pt`.

## Non-negotiable constraints

1. **Report whatever the numbers actually are.** With 357 images and 35 `human` instances, a weak result is the expected outcome, not a failure to hide. Never round up, never quote a best-epoch number as if it were the final validation result, never omit a bad class.
2. **`human` will likely score poorly.** Say so explicitly rather than reporting only the mean.
3. The val split is ~72 images. **State that metrics on a split this small are noisy** and should not be quoted to three significant figures.
4. These classes are **ship, aircraft, human — not marine debris.** Anything you write must call the model a seabed-object detector. Do not describe its output as debris detection.
5. If CUDA is unavailable, or OOM persists at batch 4, stop and report rather than falling back to CPU silently.
6. Do not `git add -A` anywhere. This repo is not initialised; do not initialise it.

## Out of scope

Do not train U-Net, do not build the FastAPI backend, do not touch `brief.html` or `PLAN.md`. Detection training and its honest evaluation only.
