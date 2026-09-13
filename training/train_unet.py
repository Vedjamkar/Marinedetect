"""
Train the 3-class U-Net (background / highlight / shadow) on weak masks.

Uses backend.ml.unet_arch.UNet directly, so the checkpoint this produces is
loadable by the backend's shape-validating loader with no conversion.

THE LABELS ARE WEAK. They come from training/make_masks.py, which thresholds
intensity inside and below each annotated detection box. They are not human
annotation. Anything reporting results from this model must say so.

Class balance is severe - roughly 92% background, 6% highlight, 2% shadow - so
the loss is inverse-frequency weighted. Without that the model collapses to
predicting background everywhere and reports a deceptively high pixel accuracy.

Runs on CPU by default because the GPU is usually occupied by detector training
and a 4 GB card cannot host both. The model is small and the dataset is ~270
images, so CPU is workable.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.ml.unet_arch import UNet  # noqa: E402

DATA = ROOT / "data" / "unet_masks"
OUT = ROOT / "backend" / "weights" / "unet" / "unet.pth"
NUM_CLASSES = 3
SIZE = 256


class MaskDataset(Dataset):
    def __init__(self, split: str, augment: bool) -> None:
        self.images = sorted((DATA / split / "images").glob("*.png"))
        self.masks = [DATA / split / "masks" / p.name for p in self.images]
        self.augment = augment
        if not self.images:
            raise SystemExit(f"no data in {DATA / split}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        img = cv2.imread(str(self.images[i]), cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(str(self.masks[i]), cv2.IMREAD_GRAYSCALE)
        if img.shape != (SIZE, SIZE):
            img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
            msk = cv2.resize(msk, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            # Horizontal flip only. A VERTICAL flip would move the shadow
            # up-range of its target, which is physically impossible in
            # side-scan geometry and teaches the model a false prior.
            if np.random.rand() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1])
                msk = np.ascontiguousarray(msk[:, ::-1])
            if np.random.rand() < 0.5:
                img = np.clip(img.astype(np.float32) * np.random.uniform(0.8, 1.2), 0, 255).astype(np.uint8)

        x = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        y = torch.from_numpy(msk.astype(np.int64))
        return x, y


def class_weights(ds: MaskDataset) -> torch.Tensor:
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for p in ds.masks:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        counts += np.bincount(m.ravel(), minlength=NUM_CLASSES)
    freq = counts / counts.sum()
    w = 1.0 / np.maximum(freq, 1e-6)
    w = w / w.sum() * NUM_CLASSES          # normalize so mean weight is 1
    print(f"class pixel freq: {np.round(freq, 4).tolist()}")
    print(f"loss weights:     {np.round(w, 3).tolist()}")
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """Per-class IoU. Reported separately, never averaged into one number —
    a model good at highlights and useless at shadows must be visible as such."""
    model.eval()
    inter = np.zeros(NUM_CLASSES)
    union = np.zeros(NUM_CLASSES)
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu().numpy()
        true = y.numpy()
        for c in range(NUM_CLASSES):
            p, t = pred == c, true == c
            inter[c] += np.logical_and(p, t).sum()
            union[c] += np.logical_or(p, t).sum()
    iou = inter / np.maximum(union, 1)
    return {"background": iou[0], "highlight": iou[1], "shadow": iou[2]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"device={device}  epochs={args.epochs}  batch={args.batch}")

    tr = MaskDataset("train", augment=True)
    va = MaskDataset("val", augment=False)
    print(f"train={len(tr)}  val={len(va)}")

    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=0)
    vl = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=0)

    model = UNet(in_channels=1, num_classes=NUM_CLASSES).to(device)
    crit = nn.CrossEntropyLoss(weight=class_weights(tr).to(device))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = -1.0
    t0 = time.perf_counter()
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for x, y in tl:
            opt.zero_grad()
            loss = crit(model(x.to(device)), y.to(device))
            loss.backward()
            opt.step()
            tot += float(loss)
        sched.step()

        iou = evaluate(model, vl, device)
        # Selection metric is the mean of the two FOREGROUND classes.
        # Including background would let a lazy model win on the 92% class.
        fg = (iou["highlight"] + iou["shadow"]) / 2
        flag = ""
        if fg > best:
            best = fg
            OUT.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), OUT)
            flag = "  <- saved"
        print(
            f"ep{ep:>3}  loss={tot/max(1,len(tl)):.4f}  "
            f"IoU bg={iou['background']:.3f} highlight={iou['highlight']:.3f} "
            f"shadow={iou['shadow']:.3f}  fg_mean={fg:.3f}{flag}"
        )

    print(f"\nbest foreground mean IoU: {best:.4f}")
    print(f"elapsed: {(time.perf_counter()-t0)/60:.1f} min")
    print(f"weights: {OUT}  exists={OUT.exists()}")
    print("\nLABELS WERE WEAK (thresholded, not human). Report this model as weakly supervised.")


if __name__ == "__main__":
    main()
