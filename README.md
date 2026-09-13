# Marinedetect

Automated detection of marine debris and seabed anomalies in **side-scan sonar**, with
object height derived from acoustic shadow geometry.

The system is deliberately split into three layers, and only the first two are implemented:

| Layer | Produces | Status |
|---|---|---|
| **Perception** — YOLO + U-Net | object class, highlight and shadow pixel extents | implemented |
| **Acoustic geometry** | slant range, ground range, **object height from shadow length** | implemented |
| **Spatial reference** — GPS/INS, bathymetry | latitude, longitude, seabed depth | **not implemented, by design** |

Seabed depth is never derived from imagery. It comes from an echosounder or a bathymetric
survey. Any field the system cannot compute is reported as absent with the reason, never
filled with a placeholder.

---

## Quick start

> **New to this project?** Start with **[QUICKSTART.md](QUICKSTART.md)** — a five-minute
> step-by-step guide with troubleshooting. This file is the full reference.

Requires **Python 3.10+** (3.11 recommended). One command does everything:

```bash
python scripts/setup.py
```

It creates `.venv`, detects whether you have an NVIDIA GPU, installs the matching torch
build, installs the rest, then verifies the install and runs the tests. It prints exactly
what works and what doesn't.

Options:

```bash
python scripts/setup.py --cpu      # force the CPU-only torch build
python scripts/setup.py --check    # verify an existing install, install nothing
```

Then run the service:

```bash
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/> for the demo interface.

### Manual setup, if you prefer

torch is deliberately **not** in `requirements.txt`, because the correct build depends on
your machine. Install it first:

```bash
# GPU, CUDA 12.4
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# CPU only
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
```

then

```bash
pip install -r requirements.txt
```

---

## Does it work without model weights?

**Yes, by design, and this is worth knowing before you go looking for a bug.**

- `GET /health` and `GET /api/v1/model-status` always work. Model-status reports each model
  as loaded or unavailable *with the exact path it looked in*.
- `POST /api/v1/inference` with no detector returns a clean **503** naming the missing file,
  not a stack trace.
- **The shadow measurement and height calculation need no model at all.** They work on the
  raw intensity profile. This is the core of the system and it runs on a fresh clone.

Weights are gitignored (they are large binaries). Without them you get a working API and a
working physics pipeline; you just get no detections.

### Getting weights

Either obtain the trained checkpoints from the team and place them at:

```
backend/weights/yolo/best.pt      # detector
backend/weights/unet/unet.pth     # 3-class segmentation
```

or train your own — see **Training** below. A GPU is required for detector training.

---

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness |
| `GET /api/v1/model-status` | which models are loaded, device, active thresholds |
| `POST /api/v1/inference` | run the pipeline on one image |
| `GET /` | demo interface |

### Example

```bash
curl -X POST http://127.0.0.1:8000/api/v1/inference \
  -F "image=@sonar.png" \
  -F 'geometry={"towfish_altitude_m":12.0,"slant_range_m":40.0,"across_track_resolution_m_per_px":0.05}' \
  -F "analyze_full_frame=true"
```

`geometry` is optional. Without it, detections and shadow measurements still come back, but
height is reported as unavailable naming exactly which input was missing — it is never
estimated from a default.

`analyze_full_frame=true` additionally measures the shadow across the whole image with no
detector involved. Useful for verifying the physics independently of detection.

### Response shape, and the honesty rules baked into it

Every field that could be absent carries `available` plus a `reason`. Every measurement
carries a `method` naming what produced it — `classical_profile` or `unet`. Detection
confidence and segmentation metrics are **never** blended into a combined score, because
that number would not mean anything.

```jsonc
{
  "mode": "trained",
  "trained_model": true,
  "detections": [{
    "class_name": "ship",
    "confidence": 0.75,
    "bbox": {...},
    "shadow":           { "available": true, "method": "classical_profile", "shadow_length_px": 34.0, "profile": {...} },
    "shadow_classical": { ... },
    "shadow_unet":      { ... },
    "cross_check":      { "available": true, "agrees": false, "relative_difference": 0.46 },
    "height":           { "available": true, "height_m": 0.51, "ground_range_m": 38.16, "method": "classical_profile",
                          "assumptions": ["locally flat seabed", "shadow measured in ground range"] },
    "segmentation":     { "available": true, "highlight_area_px": 16829, "shadow_area_px": 20118 },
    "spatial":          { "available": false, "reason": "module not implemented" }
  }],
  "timings_ms": { "preprocess": 12, "detect": 50, "fuse": 12, "render": 24, "total": 98 }
}
```

---

## The physics

```
R  = c * t / 2                 slant range from two-way travel time
G  = sqrt(R^2 - H^2)           ground range   (H = towfish altitude, a MEASURED input)
h  = H * Ls / (G + Ls)         object height from ground-range shadow length Ls
```

Two things that are commonly got wrong:

1. **`H` is an input, not an unknown.** It comes from the towfish altimeter. Solving the
   second equation for `H` yields a "depth" that is not the target's depth.
2. **Height is robust to sound-speed error.** `Ls` and `G` scale together with assumed sound
   speed, so a 3% error in `c` moves the height by roughly 0.2%. Range estimates carry that
   3%; height does not.

Verified end to end: a synthetic target planted at 1.600 m is recovered at **1.61 m (0.4%
error)** through the real code path, with no model involved. See
`backend/tests/test_shadow_physics.py`.

---

## Two shadow measurements, cross-checked

The classical intensity profile and the U-Net shadow mask measure the same physical quantity
by independent routes. Where they agree, confidence is high; where they diverge, the
detection belongs in a review queue. **This is a validation signal that needs no ground
truth** — which matters, because none exists for this data.

They are never averaged into one number. Averaging would conceal the disagreement, and the
disagreement is the point.

Currently the **classical method drives the height and U-Net only cross-checks it.** That is
a measurement, not a preference: classical recovers a planted height to 0.4%, while the
current U-Net checkpoint scores shadow IoU 0.196 on weakly-labelled data. Revisit once the
U-Net is trained on real annotations.

---

## Tests

```bash
pytest backend/tests -q
```

61 tests, fully hermetic — they pass whether or not model weights are on disk (the fixtures
point the weight paths at a nonexistent directory, so results do not depend on your machine's
state).

---

## Training

**A GPU is required for detector training.** The pipeline was developed on a 4 GB GTX 1650 Ti,
which is the source of several constraints below.

```bash
# 1. Detector — SCTD (ship / aircraft / human)
python training/voc_to_yolo.py        # VOC -> YOLO, stratified split
python training/train.py

# 2. Detector — crab-pot (derelict fishing gear, actual marine debris)
python training/crabpot_to_yolo.py
python training/train_crabpot.py

# 3. Segmentation masks, then U-Net
python training/make_masks.py --src yolo --hl-k 0.6 --sh-k 0.6
python training/train_unet.py --epochs 45
```

### Constraints that are not negotiable

- **`workers=0` on Windows.** Ultralytics' spawn-based dataloader silently kills the run
  after the label-scan phase. Do not raise it.
- **`flipud=0.0`.** A vertical flip puts the acoustic shadow up-range of its target, which is
  physically impossible in side-scan geometry and teaches the model a false prior.
- **`batch=8` at 640 px** is the ceiling on 4 GB.
- **`hsv_h=0, hsv_s=0`.** The imagery is greyscale acoustic intensity; hue and saturation
  jitter are meaningless.

### Measured results

| Model | Data | Result |
|---|---|---|
| YOLOv8n | SCTD, 286 train / 71 val | mAP@0.5 **0.794**, mAP@0.5:0.95 0.477, P 0.871, R 0.682 |
| YOLOv8n | crab-pot, 5721 train / 555 val | mAP@0.5 **0.465**, mAP@0.5:0.95 0.157 @ epoch 12 |
| YOLOv8-p2 | crab-pot, same split | small-object config, training |
| U-Net (3-class) | 269 train / 68 val, **weak labels** | foreground mean IoU **0.273** (highlight 0.328, shadow 0.196) |

The validation split for SCTD is 71 images, which is small enough that its metrics are noisy.
Do not quote a single best epoch.

**The U-Net labels are weak.** They were generated by thresholding intensity inside and below
each annotated box, not by human annotation. Any result from this model must be described as
weakly supervised.

---

## Why crab-pot accuracy is lower than SCTD, and what to do about it

This is the first question anyone asks, so here is the actual diagnosis rather than a guess.

### It is a small-object problem, not a training problem

A size audit of the two datasets (both at 640 px):

| Dataset | Median box side | COCO "small" (<32 px) | mAP@0.5 |
|---|---|---|---|
| SCTD (wrecks, aircraft) | **271 px** | 0% | 0.794 |
| Crab-pot (fishing gear) | **44 px** | 33% | 0.465 |

SCTD scores well because its targets are enormous — 94% are "large" by COCO's definition.
Crab pots are an order of magnitude smaller.

The tell is the **gap between mAP@0.5 (0.465) and mAP@0.5:0.95 (0.157)**. That pattern means
objects are being *found* but *poorly localized*. On a 44 px box, being 3 px out costs a large
slice of IoU; on a 271 px box the same error is negligible.

### Why the architecture makes it worse

YOLOv8 detects at strides 8 / 16 / 32 (heads P3 / P4 / P5). At the median crab-pot size:

| Head | Stride | Grid cells the median object spans |
|---|---|---|
| P3 | 8 | 5.5 |
| P4 | 16 | 2.7 |
| P5 | 32 | **1.4 — contributes essentially nothing** |

A third of the detector's capacity is wasted on a head that cannot resolve these targets.

### Fixes, in order of value

1. **Add a P2 head (stride 4).** The median object then spans ~11 cells instead of 5.5. This
   is the standard remedy for small-object detection. Run `training/train_crabpot_p2.py`.
   Costs memory — use `batch=4` on a 4 GB card, not 8.
2. **Train at higher `imgsz`.** Going 640 → 960 makes the median object 66 px instead of 44.
   Also costs memory; reduce batch accordingly.
3. **Use a larger model** (`yolov8s`). More capacity, but it does not fix the resolution
   mismatch, so expect less benefit than 1 or 2.
4. **Train longer.** The baseline was still improving at epoch 12 of 25 (0.269 → 0.465 across
   epochs 5-12, noisily). Do not conclude a config has plateaued from a short run.
5. **Tile the source imagery at higher resolution.** These 640 px tiles came from larger
   survey waterfalls. Tiling with more overlap and less downscaling makes the objects bigger
   in pixels, which is the root fix rather than a compensation for it.

### What will not help

- **More augmentation.** The pipeline already applies sonar-appropriate augmentation, and the
  failure mode is resolution, not overfitting.
- **Lowering the confidence threshold.** It raises recall and destroys precision. The
  threshold is 0.40 because below that the output is mostly noise on this data.
- **Using NKSID or SeabedObjects-KLSG as extra detection data.** Both are
  classification-only — whole-image labels, no boxes — and NKSID is forward-looking sonar,
  not side-scan. Neither can train a detector. (NKSID *could* pretrain a backbone on sonar
  imagery to reduce the domain gap from COCO weights, which is a genuine idea, but that is a
  separate piece of work.)

### On the U-Net's IoU of 0.273

Low, and expected. It was trained on **269 weakly-labelled images**, where the labels came
from thresholding rather than a human. Shadow is the harder class (0.196) because it is only
2.2% of pixels. The route to improving it is better labels, not more epochs:

1. Hand-correct the generated masks — they are a starting point, not an endpoint.
2. Generate masks from more source images.
3. Only then increase model capacity or training length.

Until its shadow IoU beats the classical profile method, `fusion_service` deliberately keeps
the classical measurement as the primary height source. That ordering is set by measurement
and should be revisited when the numbers change.

---

## Known limitations

- **The shadow-height method needs substantial targets.** Crab pots (median box 20×20 px,
  target-vs-seabed contrast median +0.43σ, p25 −0.03σ) do not cast resolvable shadows. It was
  developed against wrecks and aircraft. Small debris may produce nothing measurable.
- **The classical method assumes TVG-corrected imagery.** If seabed brightness falls off with
  range, the seabed past the shadow can sit below the threshold derived from before it, and
  the shadow appears never to end.
- **Raster input only.** Operational sonar arrives as XTF / JSF / SEG-Y with per-ping
  altitude, heading and position. A PNG has been stripped of all of it, so geometry must be
  supplied by the caller. The typed interface for that metadata already exists; an XTF parser
  would populate it.
- **A flat seabed is assumed.** On a slope the shadow stretches or shortens and the height
  estimate carries that error.
- **No position, no depth, no bathymetry.** Deliberate — see the layer table at the top.

---

## Layout

```
backend/            FastAPI service
  api/                health, model-status, inference endpoints
  services/
    shadow_service.py   classical shadow measurement + height. Needs no model.
    fusion_service.py   YOLO -> padded ROI -> classical + U-Net -> cross-check
    yolo_service.py     lazy load-once, refuses COCO checkpoints
    unet_service.py     shape-validating loader
  ml/unet_arch.py     3-class U-Net
  schemas/            Pydantic models; the honesty rules live here
  static/index.html   demo interface
  tests/              61 hermetic tests
training/           dataset conversion and training scripts
scripts/setup.py    one-command environment setup and verification
```

## Documentation map

| File | Read it when |
|---|---|
| **QUICKSTART.md** | you just want it running |
| **README.md** (this file) | you need the API, the physics, or training details |
| **PROJECT_STATE.md** | you are picking the work up mid-stream |
| **PLAN.md** | you want the architecture and *why* each decision was made |
| **CREDITS.md** | before publishing or redistributing anything |
| **TRAINING.md** / **BACKEND.md** | the original build specs the code was written against |

---

## Attribution

Third-party datasets and models are credited in `CREDITS.md`. The crab-pot dataset and the
GhostVision baseline models are CC-BY-SA-4.0, which carries attribution **and share-alike**
obligations on derivatives — check that file before publishing or redistributing trained
weights.
