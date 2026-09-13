"""
Crab-pot detector, small-object configuration (P2 head).

WHY THIS EXISTS
The baseline yolov8n run reached mAP@0.5 = 0.465 with mAP@0.5:0.95 = 0.157 at
epoch 12. That gap - boxes found but poorly localized - is the signature of a
small-object problem, and a size audit confirmed it:

    crab-pot median box side   44 px   (33% "small" by COCO, <32px)
    SCTD     median box side  271 px   (94% "large")

YOLOv8's detection heads use strides 8 / 16 / 32 (P3/P4/P5). At a 44 px median
object the P5 head sees ~1.4 grid cells, so the deepest head contributes almost
nothing, and localization on the remaining heads is coarse relative to the box.

`yolov8-p2.yaml` adds a stride-4 (P2) head. The same median object then spans
~11 cells instead of 5.5, which is the standard fix for this failure mode.

COST
The P2 head operates at 160x160 for a 640 input, so it is markedly heavier in
both compute and memory. batch is 4 here, not 8, because 4 GB will not hold the
P2 feature maps at batch 8.

Compare against runs/crabpot_baseline_ep12 at matched epochs, not against a
different epoch count.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = str(ROOT / "training" / "crabpot.yaml")
PROJECT_DIR = str(ROOT / "runs")
RUN_NAME = "crabpot_p2"

BATCH = 4          # P2 feature maps are large; 8 does not fit in 4 GB
MIN_BATCH = 2


def check_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("FATAL: CUDA unavailable. Refusing to train on CPU silently.")
    print(f"CUDA OK: {torch.cuda.get_device_name(0)}")


def train(batch_size: int):
    # Start from the COCO-pretrained nano weights; the P2 architecture takes what
    # it can from them and initializes the extra head fresh.
    model = YOLO("yolov8-p2.yaml").load("yolov8n.pt")
    return model.train(
        data=DATA_YAML,
        imgsz=640,
        batch=batch_size,
        epochs=25,
        patience=6,
        amp=True,
        device=0,
        project=PROJECT_DIR,
        name=RUN_NAME,
        seed=42,
        workers=0,          # mandatory on Windows
        # --- sonar-appropriate augmentation (identical to the baseline run,
        #     so the P2 head is the only variable) ---
        fliplr=0.5,
        flipud=0.0,         # a vertical flip puts the shadow up-range: impossible
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.4,
        degrees=5.0,
        scale=0.5,
        translate=0.1,
        mosaic=1.0,
        close_mosaic=5,
        exist_ok=True,
    )


def main() -> None:
    check_cuda()
    t0 = time.perf_counter()
    batch = BATCH
    try:
        results = train(batch)
    except torch.cuda.OutOfMemoryError:
        print(f"CUDA OOM at batch={batch}; retrying once at batch={MIN_BATCH}")
        torch.cuda.empty_cache()
        batch = MIN_BATCH
        results = train(batch)

    print(f"\nFinished in {(time.perf_counter()-t0)/60:.1f} min at batch={batch}")
    try:
        b = results.box
        print(f"mAP50={b.map50:.4f}  mAP50-95={b.map:.4f}  P={b.mp:.4f}  R={b.mr:.4f}")
        print("\nBaseline (yolov8n, no P2) at epoch 12: mAP50=0.465  mAP50-95=0.157")
    except Exception as exc:
        print(f"could not read summary metrics: {exc}")


if __name__ == "__main__":
    main()
