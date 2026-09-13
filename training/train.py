"""
Train a YOLOv8n seabed-object detector (ship / aircraft / human) on the
SCTD side-scan sonar dataset.

Config is fixed by TRAINING.md — do not raise batch size above 8, do not
enable flipud, do not fall back to CPU on CUDA failure.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

import torch
from ultralytics import YOLO

DATA_YAML = str(ROOT / "training" / "data.yaml")
PROJECT_DIR = str(ROOT / "runs")
RUN_NAME = "sctd_yolov8n"
WEIGHTS_OUT = ROOT / "weights" / "yolo" / "best.pt"

BATCH = 8
MIN_BATCH = 4


def check_cuda():
    if not torch.cuda.is_available():
        print("FATAL: CUDA is not available. Refusing to fall back to CPU training.")
        sys.exit(1)
    print(f"CUDA OK: {torch.cuda.get_device_name(0)}")


def train_with_batch(batch_size: int):
    model = YOLO("yolov8n.pt")
    results = model.train(
        data=DATA_YAML,
        imgsz=640,
        batch=batch_size,
        epochs=150,
        patience=40,
        amp=True,
        device=0,
        project=PROJECT_DIR,
        name=RUN_NAME,
        seed=42,
        # Windows: dataloader worker processes use spawn and have been observed
        # killing this run silently after the label-scan phase. 286 small images
        # load fast enough single-process that we lose nothing here.
        workers=0,
        # Sonar-appropriate augmentation (see TRAINING.md)
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.4,
        degrees=5.0,
        scale=0.5,
        translate=0.1,
        mosaic=1.0,
        close_mosaic=20,
        exist_ok=True,
    )
    return model, results


def main():
    check_cuda()

    start = time.time()
    batch_size = BATCH
    model = None
    results = None

    try:
        model, results = train_with_batch(batch_size)
    except torch.cuda.OutOfMemoryError:
        print(f"CUDA OOM at batch={batch_size}. Retrying once at batch={MIN_BATCH}.")
        torch.cuda.empty_cache()
        batch_size = MIN_BATCH
        try:
            model, results = train_with_batch(batch_size)
        except torch.cuda.OutOfMemoryError:
            print(f"FATAL: CUDA OOM persists at batch={MIN_BATCH}. Stopping per spec "
                  f"(no CPU fallback).")
            sys.exit(1)

    elapsed = time.time() - start
    print(f"\nTraining finished in {elapsed / 60:.1f} minutes, batch={batch_size}")

    # Locate the actual save_dir Ultralytics used (don't assume a path —
    # observed to sometimes nest differently than project/name would
    # suggest). Fall back to the trainer's save_dir, then to the naive
    # project/name guess only as a last resort.
    run_dir = None
    if results is not None and getattr(results, "save_dir", None):
        run_dir = Path(results.save_dir)
    elif model is not None and getattr(model, "trainer", None) is not None:
        run_dir = Path(model.trainer.save_dir)
    else:
        run_dir = Path(PROJECT_DIR) / RUN_NAME

    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"WARNING: expected best weights at {best_pt}, not found")
    else:
        WEIGHTS_OUT.parent.mkdir(parents=True, exist_ok=True)
        WEIGHTS_OUT.write_bytes(best_pt.read_bytes())
        print(f"Copied best weights to {WEIGHTS_OUT}")

    print(f"Run directory: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
