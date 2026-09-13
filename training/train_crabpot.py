"""
Fine-tune YOLOv8n on the crab-pot side-scan dataset (derelict fishing gear).

Unlike the SCTD run, this is real marine debris in real side-scan survey
imagery, which is the project's actual problem statement.

Sizing notes for the GTX 1650 Ti (4 GB):
  - Images are already 640x640, so imgsz=640 means no resampling.
  - batch=8 is the ceiling at 640 on 4 GB. Do not raise it.
  - workers=0 is mandatory on Windows; the spawn-based dataloader has been
    observed killing the run silently after the label-scan phase.

Epoch budget: 5,721 training images is ~20x SCTD, so far fewer epochs are
needed. 25 epochs here is roughly 143k image-presentations, against SCTD's
150 x 286 = 43k. Ultralytics writes best.pt every epoch, so the run can be
used for a demo at any point without waiting for it to finish.

Attribution: PINGEcosystem/sss-crab-pot-detection-ds, CC-BY-SA-4.0. See CREDITS.md.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = str(ROOT / "training" / "crabpot.yaml")
PROJECT_DIR = str(ROOT / "runs")
RUN_NAME = "crabpot_yolov8n"

BATCH = 8
MIN_BATCH = 4


def check_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("FATAL: CUDA unavailable. Refusing to fall back to CPU silently.")
    print(f"CUDA OK: {torch.cuda.get_device_name(0)}")


def train(batch_size: int):
    model = YOLO("yolov8n.pt")
    results = model.train(
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
        workers=0,  # mandatory on Windows, see module docstring
        # --- sonar-appropriate augmentation ---
        fliplr=0.5,     # port/starboard mirroring is physically meaningful
        flipud=0.0,     # MUST stay 0: a vertical flip puts the shadow up-range,
                        # which is physically impossible in side-scan geometry
        hsv_h=0.0,      # imagery is greyscale acoustic intensity
        hsv_s=0.0,      # -- hue/saturation jitter is meaningless here
        hsv_v=0.4,      # brightness jitter stands in for gain/TVG variation
        degrees=5.0,    # small only; sonar geometry is not rotation-invariant
        scale=0.5,
        translate=0.1,
        mosaic=1.0,
        close_mosaic=5,
        exist_ok=True,
    )
    return model, results


def main() -> None:
    check_cuda()
    t0 = time.perf_counter()
    batch = BATCH
    try:
        model, results = train(batch)
    except torch.cuda.OutOfMemoryError:
        print(f"CUDA OOM at batch={batch}; retrying once at batch={MIN_BATCH}")
        torch.cuda.empty_cache()
        batch = MIN_BATCH
        model, results = train(batch)

    mins = (time.perf_counter() - t0) / 60
    print(f"\nTraining finished in {mins:.1f} min at batch={batch}")

    # Report whatever the numbers actually are.
    try:
        box = results.box
        print(f"mAP50={box.map50:.4f}  mAP50-95={box.map:.4f}  P={box.mp:.4f}  R={box.mr:.4f}")
    except Exception as exc:
        print(f"could not read summary metrics off the results object: {exc}")

    best = Path(PROJECT_DIR) / RUN_NAME / "weights" / "best.pt"
    print(f"best weights: {best}  exists={best.exists()}")


if __name__ == "__main__":
    main()
