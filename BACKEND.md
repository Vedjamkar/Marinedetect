# Backend Build — Handover Spec

Owner: Sonnet agent. Read `PLAN.md` first for architecture rationale; this file is the build order and the parts that changed.

## Goal

A runnable FastAPI service that takes a side-scan sonar image and returns detections, an annotated image, and — this is the important part — **a measured object height**, with every number traceable to a named source.

It must run and be demonstrable **before any model weights exist**, and degrade honestly when they don't.

## Environment (verified)

- venv: `D:\Marinedetect\.venv`, python at `D:/Marinedetect/.venv/Scripts/python.exe`
- Installed already: torch 2.6.0+cu124 (CUDA works), ultralytics, opencv-python, huggingface_hub
- **Never use the `python` on PATH** — it is a Hermes agent venv with no pip.
- GPU: GTX 1650 Ti, 4 GB. Training may be running concurrently — do not start GPU work.

## Build order

Work top to bottom. Each step must run before you start the next.

### 1. `backend/config.py`
pydantic-settings. All paths resolve from `BASE_DIR = Path(__file__).resolve().parent`, never cwd. Env vars per PLAN.md §4. Provide a `.env.example`.

### 2. `backend/schemas/`
Pydantic models: `Detection`, `Segmentation`, `HeightEstimate`, `SonarGeometry`, `InferenceResponse`, `ModelStatus`.

`SonarGeometry` accepts and echoes back: `towfish_altitude_m`, `slant_range_m`, `sound_speed_mps`, `latitude`, `longitude`, `heading_deg`, `across_track_resolution_m_per_px`. All optional. **Never invent values for these.**

### 3. `backend/main.py` + `api/`
- `GET /health`
- `GET /api/v1/model-status` — reports per-model loaded/unavailable with a specific reason, plus device and active thresholds. **Must not crash when weights are absent.**
- `POST /api/v1/inference` — multipart image upload, optional `geometry` JSON form field.
- Mount `outputs/` as static. Server-generated UUID filenames only — never user-derived (path traversal).
- Handlers are sync `def` (FastAPI threadpools them). Guard model inference with a `threading.Semaphore(1)` — 4 GB will not take concurrent inference.

Server must boot with zero weights on disk. Verify this before moving on.

### 4. `backend/utils/image_utils.py`
Safe load (PNG/JPG/TIFF), 16-bit percentile normalization, `Image.MAX_IMAGE_PIXELS` cap, upload size enforced **while streaming to disk** (not after). Padded ROI crop: `ROI_PAD_RATIO` default 0.25, extra padding on the down-range side per `SHADOW_DIRECTION`.

### 5. `backend/services/yolo_service.py`
Lazy load-once from `YOLO_MODEL_PATH`. Report unavailable with a reason if missing. **COCO guard:** if the checkpoint's class names look like COCO (person/car/dog/etc.), refuse to load in trained mode and say why.

### 6. `backend/services/shadow_service.py` — **THE KEY COMPONENT**

This is new and is not in PLAN.md. It makes the demo work tonight with **no trained segmentation model at all**, and it is fully explainable.

For each detection, measure the acoustic shadow classically:

1. Take the padded ROI. Compute a mean intensity profile **along the range axis** (average across the along-track axis to suppress speckle).
2. The target highlight is the bright peak. The shadow is the dark run immediately **down-range** of it.
3. Find shadow start = first index after the highlight peak where intensity drops below `seabed_mean - k*seabed_std` (k configurable, default 1.0). Shadow end = where it recovers above that threshold for a sustained run.
4. Return `shadow_length_px`, plus the profile itself so it can be plotted.

Then compute height **only if the caller supplied the required geometry**:

```
G = sqrt(R**2 - H**2)
h = H * Ls / (G + Ls)
```

where `Ls = shadow_length_px * across_track_resolution_m_per_px`.

**If any required geometry input is missing, return `available: false` with `reason` naming exactly which input is absent. Never substitute a default and never estimate.** Always return `assumptions: ["locally flat seabed", "shadow measured in ground range"]` alongside any height.

### 7. `backend/services/unet_service.py`
3-class U-Net (background / highlight / shadow), softmax + argmax. Shape-validating loader that names the exact mismatched tensor and both shapes on failure. With no checkpoint it reports unavailable — expected, do not fake it.

### 8. `backend/services/fusion_service.py`
Bind each detection to its box, masks (if any), shadow measurement, and height (if computable). **No blended "combined confidence" score** — detection confidence and segmentation metrics stay separate fields.

### 9. `backend/utils/visualization.py`
Annotated output: boxes, class + confidence labels, shadow extent marked, mask overlay when available. Also render the **intensity profile plot** for each detection with the shadow span bracketed — this is the explainability asset, it shows the measurement being made.

### 10. `backend/tests/`
Hermetic — must pass with **zero weights on disk**. Synthetic sonar fixture generated with numpy (gradient seabed + bright blob + dark shadow at a known offset).

Critical test: feed the synthetic fixture with known geometry and assert the recovered height matches the planted value within tolerance. **This proves the physics chain end to end without any trained model.**

## Honesty constraints — the point of the project

1. Never populate depth, latitude, longitude, or height with a placeholder. Missing input → field absent with a stated reason.
2. Never present classical shadow measurement as a neural network result. Response says which produced it: `"method": "classical_profile"` or `"unet"`.
3. Every response carries `"mode"` and `"trained_model": true|false`.
4. No fabricated metrics anywhere.

## Out of scope

No frontend. No database. No map. Do not touch `brief.html`, `PLAN.md`, `TRAINING.md`, `training/`, or `runs/`. Do not start GPU training.

## Report back

What runs, what the tests prove, the exact curl command to demo it, and anything you could not complete.
