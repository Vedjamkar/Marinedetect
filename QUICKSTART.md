# Quickstart

Get it running in about five minutes. For the full documentation see [README.md](README.md).

---

## 1. Check you have Python

```bash
python --version
```

You need **3.10, 3.11, 3.12 or 3.13**. If that command fails, shows something older, or shows
**3.14** (the pinned PyTorch has no 3.14 wheels), install Python 3.12 from
[python.org](https://www.python.org/downloads/). The setup script searches for a compatible
version on its own, so having 3.14 installed *as well* is fine — it just can't be the only one.

> **Windows note:** if `python` opens the Microsoft Store, use `py -3.11` instead everywhere
> below, or install Python from python.org and tick "Add python.exe to PATH".

---

## 2. The one-click way

**Windows:** double-click `START_DEMO.bat`
**Mac / Linux:** `chmod +x start_demo.sh` then `./start_demo.sh`

It sets everything up on first run, starts the server, and opens your browser.
Skip to step 4 if you use this.

> **Do the first run before the day you present**, on the presentation laptop, on good wifi.
> It downloads ~2.5 GB of PyTorch. Every run after that is offline and starts in seconds.

## 2b. Or run the setup script yourself

From the project folder:

```bash
python scripts/setup.py
```

That's the whole install. It will:

1. create a `.venv` folder (an isolated Python environment, so nothing touches your system)
2. check whether you have an NVIDIA graphics card and install the matching PyTorch build
3. install everything else
4. verify it worked and run the tests

It prints a line per component. You want to see `[ ok ]`. A `[warn]` is not fatal — it means
something optional is missing and it tells you what.

**Takes 5-15 minutes**, mostly downloading PyTorch (~2.5 GB with GPU support).

No NVIDIA card? It installs the CPU build automatically. Everything works except training.

---

## 3. Start it

**Windows**

```bash
.venv\Scripts\activate
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

**macOS / Linux**

```bash
source .venv/bin/activate
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open **<http://127.0.0.1:8000/>**

---

## 4. Try it

1. Click a thumbnail under **Sample images** — the green **VERIFY** tile is the best one to
   start with.
2. It fills in the sonar geometry for you.
3. Click **Run inference**.

You should see an annotated image, an intensity-profile chart, and a height calculation.

On the VERIFY sample specifically, you'll see:

```
planted ground truth  1.60 m
recovered             1.61 m
error                 0.4%
```

That's a synthetic target with a shadow of known size. The system measured the shadow and
worked back to the height without being told the answer.

---

## Things that look broken but aren't

**Grey panels saying "not computed"**
These are correct. The system refuses to invent numbers. If it doesn't have the input for
something, it says so and names what's missing. A panel reading *"Position & depth — by
design"* is the architecture working, not a failure.

**"No detector loaded" / a 503 from the inference endpoint**
Model weight files aren't in the repository — they're large binaries. The API, the shadow
measurement and the height calculation all still work without them; you just get no
detections. See "Getting weights" in [README.md](README.md).

**Two panels showing different shadow lengths**
Deliberate. Two independent methods measure the same shadow. When they disagree, the system
shows you both rather than averaging them into one confident-looking wrong number.

**Tests fail after you train a model**
They shouldn't — the suite is isolated from whatever weights are on disk. If you do see this,
it's a real bug, not expected behaviour.

---

## Common problems

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'backend'` | running from the wrong folder | run from the project root, not from inside `backend/` |
| `Port 8000 is in use` | something else is on that port | add `--port 8001` and open that instead |
| Setup says CUDA not available, but you have a GPU | wrong torch build, or a stale driver | update your NVIDIA driver, then `python scripts/setup.py --cpu` and re-run without `--cpu` |
| You edited the interface and see no change | the browser cached it | hard reload: `Ctrl+Shift+R` (`Cmd+Shift+R` on Mac) |
| `CUDA out of memory` during training | batch size too large for your card | lower `batch` in the training script; 4 GB needs `batch=8` at most, `batch=4` for the P2 config |

---

## What to read next

| File | What's in it |
|---|---|
| [README.md](README.md) | full documentation: API, physics, training, limitations |
| [PROJECT_STATE.md](PROJECT_STATE.md) | current status, measured results, next actions |
| [PLAN.md](PLAN.md) | architecture and the reasoning behind each decision |
| [CREDITS.md](CREDITS.md) | third-party data attribution and licence obligations |

---

## The one-paragraph version of what this does

It finds objects in side-scan sonar images and works out **how tall they stand off the
seabed** by measuring the acoustic shadow behind them. It does **not** work out how deep the
water is — that needs a bathymetric survey or an echosounder, and no amount of image
processing can substitute. Everything the system reports names the instrument it came from,
and anything it can't compute is shown as missing rather than guessed.
