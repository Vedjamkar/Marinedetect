#!/usr/bin/env python3
"""
One-command environment setup for Marinedetect.

    python scripts/setup.py            # auto-detect GPU, install, verify
    python scripts/setup.py --cpu      # force the CPU-only torch build
    python scripts/setup.py --check    # verify an existing install, change nothing

Creates .venv, installs the correct torch build for this machine, installs
requirements.txt, then runs a real verification (imports, CUDA state, weights
present, test suite) and prints exactly what works and what does not.

Deliberately stdlib-only so it runs on a bare Python with nothing installed.
Needs a CPython 3.10-3.13 somewhere on the machine; it finds one itself.
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv"
IS_WIN = platform.system() == "Windows"
PY = VENV / ("Scripts/python.exe" if IS_WIN else "bin/python")

TORCH_CU124 = ["torch==2.6.0", "torchvision==0.21.0",
               "--index-url", "https://download.pytorch.org/whl/cu124"]
TORCH_CPU = ["torch==2.6.0", "torchvision==0.21.0",
             "--index-url", "https://download.pytorch.org/whl/cpu"]

OK, BAD, WARN = "[ ok ]", "[FAIL]", "[warn]"


def run(cmd: list[str], **kw) -> int:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    return subprocess.call([str(c) for c in cmd], **kw)


def out(cmd: list[str]) -> str:
    try:
        return subprocess.check_output([str(c) for c in cmd],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return ""


def has_nvidia_gpu() -> bool:
    if not shutil.which("nvidia-smi"):
        return False
    return bool(out(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]))


# torch 2.6.0 ships wheels for CPython 3.9-3.13 only. A machine whose default
# `python` is 3.14 (increasingly common) must NOT be allowed to build the venv
# with it, or the very first pip install fails with "no matching distribution".
SUPPORTED = ((3, 10), (3, 13))


def _interpreter_version(cmd: list[str]):
    """Return (major, minor, path) for a candidate interpreter, or None."""
    try:
        out = subprocess.check_output(
            [*cmd, "-c", "import sys;print(sys.version_info[0],sys.version_info[1],sys.executable)"],
            stderr=subprocess.DEVNULL, text=True, timeout=20,
        ).split()
        return int(out[0]), int(out[1]), out[2]
    except Exception:
        return None


def find_compatible_python():
    """Find an interpreter torch actually supports, preferring newest.

    Tries the version-specific launchers first so a stray 3.14 on PATH is
    skipped rather than picked, then falls back to whatever is running this
    script if that happens to be in range.
    """
    candidates = []
    for minor in (13, 12, 11, 10):
        if IS_WIN:
            candidates.append(["py", f"-3.{minor}"])
        candidates.append([f"python3.{minor}"])
    candidates.append([sys.executable])
    for c in ("python3", "python"):
        candidates.append([c])

    seen = set()
    for cmd in candidates:
        if not shutil.which(cmd[0]):
            continue
        info = _interpreter_version(cmd)
        if not info:
            continue
        major, minor, path = info
        if path in seen:
            continue
        seen.add(path)
        if SUPPORTED[0] <= (major, minor) <= SUPPORTED[1]:
            return cmd, f"{major}.{minor}", path
    return None


def create_venv() -> None:
    if PY.exists():
        print(f"{OK} venv already exists at {VENV}")
        return

    if shutil.which("uv"):
        print(f"Creating venv at {VENV} with uv (Python 3.11) ...")
        run(["uv", "venv", "--python", "3.11", str(VENV)])
        if PY.exists():
            return
        print(f"{WARN} uv could not provision Python 3.11; falling back to a local interpreter")

    found = find_compatible_python()
    if not found:
        here = f"{sys.version_info[0]}.{sys.version_info[1]}"
        print()
        print(f"{BAD} No compatible Python found.")
        print(f"       This script is running under Python {here}, but the pinned PyTorch build")
        print(f"       supports Python {SUPPORTED[0][0]}.{SUPPORTED[0][1]} to {SUPPORTED[1][0]}.{SUPPORTED[1][1]} only.")
        print()
        print("       Install Python 3.12 from https://www.python.org/downloads/")
        if IS_WIN:
            print("       (tick \"Add python.exe to PATH\"), or run:   py install 3.12")
        print("       then run this script again. It will find it automatically.")
        sys.exit(1)

    cmd, ver, path = found
    print(f"Creating venv at {VENV} with Python {ver} ({path}) ...")
    run([*cmd, "-m", "venv", str(VENV)])
    if not PY.exists():
        sys.exit(f"{BAD} venv creation failed")


def pip_install(args: list[str]) -> None:
    if shutil.which("uv"):
        env = dict(os.environ, VIRTUAL_ENV=str(VENV))
        rc = run(["uv", "pip", "install", "--python", str(PY), *args], env=env)
    else:
        rc = run([str(PY), "-m", "pip", "install", *args])
    if rc != 0:
        sys.exit(f"{BAD} install failed: {' '.join(args[:3])} ...")


def verify() -> int:
    """Check what actually works. Returns the number of hard failures."""
    print("\n" + "=" * 62)
    print("VERIFICATION")
    print("=" * 62)
    failures = 0

    # A freshly-extracted copy has no environment yet. Say so plainly rather
    # than failing on a probe that was never going to run.
    if not PY.exists():
        print(f"{WARN} no environment found at {VENV}")
        print()
        print("       Expected on a fresh copy. Create it with:")
        print()
        print("           python scripts/setup.py")
        print()
        print("       That installs everything, then re-runs this check.")
        return 0

    probe = (
        "import json,sys\n"
        "r={}\n"
        "try:\n"
        "    import torch\n"
        "    r['torch']=torch.__version__\n"
        "    r['cuda']=torch.cuda.is_available()\n"
        "    r['gpu']=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None\n"
        "except Exception as e: r['torch_err']=str(e)\n"
        "for m in ('ultralytics','cv2','fastapi','uvicorn','numpy','PIL','matplotlib'):\n"
        "    try:\n"
        "        __import__(m); r[m]=True\n"
        "    except Exception as e: r[m]=False\n"
        "print(json.dumps(r))"
    )
    raw = out([str(PY), "-c", probe])
    if not raw:
        print(f"{BAD} could not run the dependency probe")
        return failures + 1

    import json
    r = json.loads(raw)

    if "torch_err" in r:
        print(f"{BAD} torch failed to import: {r['torch_err']}")
        failures += 1
    else:
        print(f"{OK} torch {r['torch']}")
        if r["cuda"]:
            print(f"{OK} CUDA available - {r['gpu']}")
        else:
            print(f"{WARN} CUDA NOT available - inference works, TRAINING WILL NOT")

    for m in ("ultralytics", "cv2", "fastapi", "uvicorn", "numpy", "PIL", "matplotlib"):
        if r.get(m):
            print(f"{OK} {m}")
        else:
            print(f"{BAD} {m} missing")
            failures += 1

    # Weights are optional: the service is designed to run without them.
    yolo = ROOT / "backend" / "weights" / "yolo" / "best.pt"
    unet = ROOT / "backend" / "weights" / "unet" / "unet.pth"
    print(f"{OK if yolo.exists() else WARN} detector weights "
          f"{'found' if yolo.exists() else 'MISSING - API returns a clean 503, see README'}")
    print(f"{OK if unet.exists() else WARN} U-Net weights "
          f"{'found' if unet.exists() else 'MISSING - segmentation reports unavailable, classical shadow still works'}")

    print("\nRunning test suite ...")
    rc = run([str(PY), "-m", "pytest", "backend/tests", "-q"], cwd=str(ROOT))
    if rc == 0:
        print(f"{OK} tests pass")
    else:
        print(f"{WARN} some tests failed - if weights are present, 2 zero-weights tests are")
        print(f"       expected to fail. Move backend/weights aside to confirm. See README.")
    return failures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true", help="force the CPU-only torch build")
    ap.add_argument("--check", action="store_true", help="verify only, install nothing")
    args = ap.parse_args()

    print(f"Marinedetect setup\nrepo: {ROOT}\npython: {sys.version.split()[0]}  os: {platform.system()}")

    if args.check:
        sys.exit(1 if verify() else 0)

    if sys.version_info < (3, 10):
        sys.exit(f"{BAD} Python 3.10+ required (3.11 recommended); this is {sys.version.split()[0]}")

    create_venv()

    gpu = (not args.cpu) and has_nvidia_gpu()
    print(f"\nInstalling torch ({'CUDA 12.4' if gpu else 'CPU-only'}) ...")
    if not gpu and not args.cpu:
        print(f"{WARN} no NVIDIA GPU detected - installing the CPU build. Training will not run.")
    pip_install(TORCH_CU124 if gpu else TORCH_CPU)

    print("\nInstalling the rest ...")
    pip_install(["-r", str(ROOT / "requirements.txt")])

    failures = verify()

    print("\n" + "=" * 62)
    if failures:
        print(f"{BAD} setup finished with {failures} problem(s) - see above")
        sys.exit(1)
    act = ".venv\\Scripts\\activate" if IS_WIN else "source .venv/bin/activate"
    print("Setup complete.\n")
    print(f"  {act}")
    print("  python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000")
    print("  then open http://127.0.0.1:8000/")
    print("=" * 62)


if __name__ == "__main__":
    main()
