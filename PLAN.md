# Marinedetect — Backend Build Plan

Side-scan sonar marine debris / anomaly detection. SIH Round 2 prototype.
Scope of this plan: **backend AI inference pipeline only** (YOLO detection → U-Net segmentation → fusion).
No frontend, no Ministry dashboard, no depth/GPS/bathymetry math.

Status: planning complete, scaffolding not yet written.

---

## 0. Environment facts (verified 2026-09-01)

| Item | Value |
|---|---|
| GPU | NVIDIA GTX 1650 Ti, 4 GB VRAM, driver 610.47 |
| Python to use | 3.11 via `uv` (uv 0.11.30 installed) |
| `python` on PATH | **hermes-agent venv, no pip — do not use** |
| Python 3.14 on PATH via `py` | too new for reliable torch/ultralytics wheels |
| Repo state | empty, not a git repo |

Setup command (Windows):

```bash
uv venv --python 3.11 .venv
```

4 GB VRAM constraints:
- YOLOv8n/s inference: fine. YOLOv8n training at 640px, batch 4–8: fine.
- U-Net at 256px on ROI crops: fine.
- Concurrent GPU inference: **not** fine. Serialize with a semaphore.
- `uvicorn --reload` spawns two processes and loads weights twice. Dev only, with lazy load. Demo runs without `--reload`.

---

## 1. Corrections applied to the original spec

These deviate deliberately from the ChatGPT draft. Reasons recorded so they are not "fixed" back later.

### 1.1 Preprocessing must match training exactly
Applying CLAHE / denoise / normalization at inference when the detector was trained on raw tiles causes
train–test skew and silently degrades mAP.

**Rule:** each weight file ships a sidecar `preprocess.json` describing the exact pipeline used during training.
Inference loads the sidecar and replays it. No free-floating env toggle decides this.

```
weights/yolo/best.pt
weights/yolo/preprocess.json   <- {"grayscale": true, "clahe": {"clip": 2.0, "grid": 8}, ...}
```

If the sidecar is missing, preprocessing defaults to identity (pass-through) and the API response reports
`"preprocessing": {"applied": false, "reason": "no sidecar; defaulting to identity to avoid train/test skew"}`.

### 1.2 ROI padding must preserve the acoustic shadow
In side-scan sonar the **acoustic shadow** behind a target carries its height and is often the strongest
classification cue. A tight YOLO box hugs the highlight and cuts the shadow off.

**Rule:** pad the ROI before segmentation. `ROI_PAD_RATIO=0.25` default, padded asymmetrically so the
down-range side (away from the nadir/track line) gets more padding — that is where the shadow lies.

Nadir side is determined from image geometry (port/starboard channel split) once metadata exists;
until then, configurable `SHADOW_DIRECTION=down|up|right|left|none`.

This is **load-bearing**, not cosmetic — see §1.2b. Crop the shadow off and object height becomes
uncomputable.

### 1.2b U-Net segments highlight AND shadow — 3 classes, not binary

Supersedes the earlier binary-segmentation decision.

```
UNET_NUM_CLASSES=3   # 0=background, 1=highlight (target return), 2=acoustic shadow
```

Rationale: with the shadow segmented, its along-range length is **measured from the mask**, which makes
object height a computed quantity rather than a typed-in placeholder. Without this, U-Net only draws a
prettier outline than the YOLO box and does no physical work.

Head convention: 3 output channels, softmax, argmax. Loader shape-checks the final conv `out_channels`
and names the exact mismatched tensor on failure.

Consequence for §2.2: mask annotation must label two foreground classes, not one. Slightly more
annotation effort, disproportionately more defensible output.

### 1.3 Three explicit model modes, none of them dishonest

| Mode | Trigger | Behaviour |
|---|---|---|
| `trained` | sonar-trained weights present + valid | normal inference |
| `unavailable` | weights missing/incompatible | `503` with precise reason. Never a stack trace. |
| `classical_baseline` | `ALLOW_CLASSICAL_FALLBACK=true` (opt-in, never default) | deterministic threshold + shadow-pair CV detector |

Every response carries `"mode"` and `"trained_model": true|false`.
Generic COCO weights are **never** relabelled as sonar classes. If a COCO checkpoint is detected by its
class-name list, the service refuses to start in `trained` mode.

### 1.4 Blocking inference
Endpoint handlers are sync `def` (FastAPI runs them in a threadpool). GPU section guarded by
`asyncio.Semaphore(1)` / `threading.Semaphore(1)`. Never `async def` around a torch call.

### 1.5 Directory naming
The draft's `models/` collides three concepts. Split:

- `schemas/` — Pydantic request/response models
- `ml/` — PyTorch `nn.Module` architectures (U-Net), checkpoint loaders
- `weights/` — checkpoint files + preprocess sidecars

### 1.6 Segmentation head convention — pinned
The original spec's `UNET_NUM_CLASSES=2` is ambiguous between 1-channel+sigmoid and 2-channel+softmax;
a wrong guess loads a mismatched head and produces garbage silently.

**Pinned:** N channels + softmax + argmax for `N >= 2`. Per §1.2b the project uses `N=3`
(background / highlight / shadow). The 1-channel+sigmoid path is supported for compatibility with
externally-trained binary checkpoints, selected by `UNET_NUM_CLASSES=1` with `UNET_MASK_THRESHOLD=0.5`.

Loader asserts the checkpoint's final conv `out_channels` matches the configured value, and reports the
exact tensor name and both shapes when it does not.

### 1.7 Paths and uploads
- All paths resolve from `BASE_DIR = Path(__file__).resolve().parent`, never from cwd.
- Upload size enforced **while streaming to disk**, aborting mid-write. Not checked after the file lands.
- `Image.MAX_IMAGE_PIXELS` capped (decompression bomb).
- `outputs/` filenames are server-generated UUIDs only — never derived from user input (path traversal).
- Retention sweep deletes outputs older than `OUTPUT_RETENTION_HOURS`.

### 1.8 Per-stage timing
```json
"timings_ms": {"preprocess": 12, "detect": 240, "segment": 190, "fuse": 3, "render": 37, "total": 482}
```

### 1.9 Input format reality
PNG/JPG/TIFF is a proxy. Real side-scan data arrives as **XTF / JSF / SEG-Y** with per-ping metadata.
16-bit TIFF must be percentile-normalized, never naive-cast to uint8. XTF ingest (`pyxtf`) is a documented
future module, not built now.

### 1.9b Sonar physics — corrections to the team's working notes

Round 1 judges challenged the team specifically on depth/3D. The prior planning notes contain one
inverted formula and one significant omission. Recorded here so the wrong version is not presented.

**Slant range from two-way travel time** — correct as written:

```
slant_range = (sound_speed × two_way_time) / 2
```

**Slant-range to ground-range correction** — the prior notes wrote `D = sqrt(R² - H²)` and labelled `D`
a "vertical depth difference". That solves for the wrong unknown. `H` (towfish altitude above seabed) is a
*measured* input from the altimeter. The equation yields **ground range**, the horizontal distance across
the seabed:

```
ground_range = sqrt(slant_range² − towfish_altitude²)
```

**Target depth is not derived from side-scan geometry.** For debris resting on the seabed —
essentially all of it — target depth ≈ seabed depth at that ground position, which comes from the
echosounder or a bathymetric grid. Side-scan contributes horizontal **position**, not depth. Any claim
that side-scan imagery yields absolute depth on its own is indefensible.

**Object height above seabed — the quantity side-scan uniquely provides.** From shadow length by
similar triangles:

```
height = (shadow_length × towfish_altitude) / (slant_range + shadow_length)
```

Assumptions to state openly whenever this is shown: locally flat seabed, shadow length measured in
**ground range** (so slant-range correction must be applied first), and shadow fully captured in the ROI
(§1.2). This is the formula that makes §1.2b worth the annotation cost.

**Bathymetry resolution honesty.** GEBCO global bathymetry is a ~450 m grid — a regional depth reference,
incapable of resolving a metre-scale object. High-resolution multibeam coverage for Indian coastal waters
is not casually obtainable. Any bathymetry layer shown must be labelled with its actual grid resolution
and described as a coarse reference, with co-located multibeam named as the operational requirement.

None of these formulas are implemented in this task (§16 of the original spec stands). They are recorded
so the future `spatial_service.py` implements the correct ones and so the team's verbal answer is right.

### 1.10 Kept from the original spec, unchanged
- No depth, GPS, bathymetry, or 3D values computed. (§16)
- Sonar geometry accepted as an optional input, stored and echoed back untouched, with
  `"spatial_localization": {"available": false, "reason": "module not implemented"}`. (§17)
- No fabricated mAP / IoU / accuracy anywhere. Unevaluated means `"not evaluated yet"`. (§24)

---

## 2. Critical path — weights, not FastAPI

The API is a weekend. **The demo dies without weights.** This is the real risk.

### 2.1 Detection data
Candidate public side-scan datasets (all bbox-level, no masks):
- SeabedObjects-KLSG (wrecks, aircraft, drowning victims, seafloor)
- NKSID
- SCTD (Sonar Common Target Detection)

Plan: fine-tune YOLOv8n at 640px on the merged set, batch 4–8, on the 1650 Ti.

### 2.2 Segmentation masks — the actual gap
None of the above ship segmentation masks, and per §1.2b we need **two** foreground classes
(highlight, shadow). Options, in order of preference:

1. **Hand-annotate ~150 ROI crops** (bbox crops from the detection set), painting highlight and shadow
   separately. Highlight/shadow are high-contrast against the seabed return, so this is faster than
   general-purpose segmentation labelling. Roughly one afternoon. Highest quality, defensible to judges.
2. **Weak masks**: Otsu / GrabCut seeded inside each padded box — shadow is the dark mode, highlight the
   bright mode, seabed the middle. Manually reject the failures. Fast, noisier, must be described
   honestly as weakly supervised.
3. Bootstrap with option 2, refine on option 1.

Whichever is used, the README states it plainly. No claim of a mask-annotated sonar corpus we do not have.

### 2.3 Acceptance for "demo-ready"
- YOLO detects on held-out sonar tiles with a reported number, measured, not invented.
- U-Net produces a plausible mask on those detections.
- Metrics reported only if actually computed on a held-out split.

---

## 3. Target structure

```text
Marinedetect/
├── PLAN.md
├── .gitignore                  # weights/ uploads/ outputs/ .venv/ .env
└── backend/
    ├── main.py                 # app factory, router mounts, static /outputs, lifespan
    ├── config.py               # pydantic-settings, BASE_DIR-relative paths
    ├── requirements.txt
    ├── .env.example
    ├── README.md
    │
    ├── api/
    │   ├── health.py           # GET /health
    │   ├── status.py           # GET /api/v1/model-status  (mode, device, thresholds)
    │   └── inference.py        # POST /api/v1/inference
    │
    ├── services/
    │   ├── preprocessing.py    # sidecar-driven, identity by default
    │   ├── yolo_service.py     # load-once, COCO-checkpoint guard
    │   ├── unet_service.py     # shape-validated loader, segment(roi)
    │   ├── classical_service.py# opt-in CV baseline detector
    │   └── fusion_service.py   # detection + mask, no invented combined score
    │
    ├── ml/
    │   └── unet_arch.py        # baseline U-Net nn.Module, documented as baseline
    │
    ├── schemas/
    │   ├── responses.py        # InferenceResponse, Detection, ModelStatus
    │   └── geometry.py         # SonarGeometry (accepted, echoed, never used in math)
    │
    ├── utils/
    │   ├── image_utils.py      # safe load, 16-bit normalize, roi crop+pad, size guards
    │   └── visualization.py    # boxes + labels + mask overlay
    │
    ├── weights/{yolo,unet}/README.md
    ├── uploads/  outputs/
    └── tests/
        ├── conftest.py         # synthetic sonar fixture (gradient + blob + shadow)
        ├── test_health.py
        ├── test_preprocessing.py
        ├── test_roi_padding.py
        ├── test_unet_loader.py # asserts precise error on shape mismatch
        └── test_inference.py   # weights-absent path must 503, not crash
```

Tests must be hermetic — green with **zero** weights on disk, via numpy-generated sonar-like fixtures.

---

## 4. `.env.example`

```text
# --- models ---
YOLO_MODEL_PATH=weights/yolo/best.pt
UNET_MODEL_PATH=weights/unet/unet.pth
YOLO_CONFIDENCE=0.25
YOLO_IOU=0.45
UNET_INPUT_SIZE=256
UNET_NUM_CLASSES=3          # 0=background, 1=highlight, 2=shadow

# --- pipeline ---
ROI_PAD_RATIO=0.25
SHADOW_DIRECTION=none
ALLOW_CLASSICAL_FALLBACK=false

# --- runtime ---
DEVICE=auto
HOST=127.0.0.1
PORT=8000
LOG_LEVEL=INFO

# --- limits ---
MAX_UPLOAD_MB=20
MAX_IMAGE_PIXELS=80000000
OUTPUT_RETENTION_HOURS=24
```

---

## 5. Build order (tomorrow)

1. `uv venv --python 3.11`, `git init`, `.gitignore`, requirements pinned.
2. `config.py` + `schemas/` + `/health` + `/api/v1/model-status`. Server boots with **no weights present**.
3. `utils/image_utils.py` — safe load, 16-bit handling, padded ROI crop. Tests.
4. `services/preprocessing.py` — sidecar-driven, identity default. Tests.
5. `ml/unet_arch.py` + `unet_service.py` — shape-validating loader. Tests for the mismatch error.
6. `services/yolo_service.py` — lazy load-once, COCO guard.
7. `fusion_service.py` + `visualization.py`.
8. `POST /api/v1/inference` wiring all stages + per-stage timings + semaphore.
9. `README.md` with curl example, honest limitations table, future-module plug points.

Steps 1–9 are all runnable and testable **without any weights**. Weight acquisition (§2) runs in parallel
and is the schedule risk, not this list.

---

## 6. Where future modules plug in

```
detections[]  (bbox + highlight mask + shadow mask)
        +  SonarGeometry  (altitude, slant range, sound speed, GPS, heading)
        +  bathymetry raster
                     ↓
        services/spatial_service.py      (NOT BUILT)
                     ↓
        ground_range = sqrt(slant_range² − altitude²)
        height       = shadow_len × altitude / (slant_range + shadow_len)
        lat/lon      from platform position + heading + across-track offset
        depth        from bathymetry lookup at (lat, lon)
                     ↓
        response["detections"][i]["spatial"] = {lat, lon, depth_m, height_m, ...}
```

Note the division of labour, which is the defensible answer to the Round 1 depth challenge:

| Quantity | Source | Available now? |
|---|---|---|
| object class, image-space location | YOLO | after training |
| highlight + shadow pixel extents | U-Net | after training |
| ground range | sonar geometry from slant range + altitude | needs metadata |
| **object height above seabed** | **shadow length × geometry** | needs metadata + masks |
| lat / lon | GPS/INS + heading + across-track offset | needs metadata |
| **seabed depth** | **bathymetry / echosounder — never from imagery** | needs bathy data |

Until `spatial_service.py` exists these fields are present and explicitly empty. Never populated with a
guess, and never with a hard-coded demo value.

## 6b. Scope boundary vs the wider team plan

The team's broader planning covers a React Ministry dashboard, PostgreSQL persistence, Leaflet mapping,
a 3D seabed view, alerting and report generation. **None of that is in this task.** This repository is the
inference backend only. Building the dashboard first inverts the priority: Round 1 judges challenged the
physics, not the UI.

Specific items deliberately excluded here: frontend, database, map, 3D view, alerts, reports,
multi-pass fusion, XTF ingest, spatial/depth computation.

---

## 7. Honest limitations (to mirror into README.md)

- No trained sonar-specific weights yet → detection/segmentation quality unmeasured.
- No mAP / IoU / precision / recall reported. Status: **not evaluated yet**.
- U-Net architecture is a documented baseline, not a checkpoint-derived one.
- Depth, position, and geospatial extent are **not** computed and cannot be derived from imagery alone.
  Seabed depth requires bathymetry or an echosounder; position requires GPS/INS. No demo placeholder
  values are printed for these fields anywhere in the API or the UI.
- Object height from shadow length is designed for but not implemented; it additionally requires
  towfish altitude and slant range from sonar metadata, and assumes a locally flat seabed.
- Input is raster imagery; native sonar formats (XTF/JSF) are not yet ingested.
- Single-image, synchronous, single-GPU. No batching, no queue, no persistence.
```

