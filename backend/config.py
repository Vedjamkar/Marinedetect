"""
Application configuration.

All paths resolve relative to BASE_DIR (this file's directory), never the
process current working directory, so the server behaves the same no matter
where it is launched from.
"""
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- models ---
    yolo_model_path: str = "weights/yolo/best.pt"
    # Every detector to load. Comma-separated, relative to backend/. All of them
    # run on each frame and each detection records which one found it.
    yolo_model_paths: str = "weights/yolo/best.pt,weights/yolo/debris_crabpot.pt"
    unet_model_path: str = "weights/unet/unet.pth"
    # 0.40 rather than the usual 0.25: on a 286-image training set the tail
    # below 0.4 is mostly noise, and a spurious low-confidence box beside a
    # real one is worse than a miss when the output is being read by a human.
    yolo_confidence: float = 0.40
    yolo_iou: float = 0.45
    unet_input_size: int = 256
    unet_num_classes: int = 3  # 0=background, 1=highlight, 2=shadow
    unet_mask_threshold: float = 0.5  # only used when unet_num_classes == 1

    # --- pipeline ---
    roi_pad_ratio: float = 0.25
    shadow_direction: Literal["down", "up", "right", "left", "none"] = "none"
    allow_classical_fallback: bool = False
    # Relative difference below which the classical and U-Net shadow
    # measurements are considered to agree (validation signal, no ground truth needed).
    shadow_agreement_tolerance: float = 0.25
    shadow_threshold_k: float = 1.0  # k in: seabed_mean - k*seabed_std

    # --- runtime ---
    device: str = "auto"
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"

    # --- limits ---
    max_upload_mb: int = 20
    max_image_pixels: int = 80_000_000
    output_retention_hours: int = 24

    @property
    def yolo_model_abs_paths(self) -> list:
        """Absolute paths of every configured detector, de-duplicated, order kept."""
        out, seen = [], set()
        for raw in self.yolo_model_paths.split(","):
            raw = raw.strip()
            if not raw:
                continue
            p = Path(raw)
            p = p if p.is_absolute() else BASE_DIR / p
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out

    @property
    def yolo_model_abs_path(self) -> Path:
        p = Path(self.yolo_model_path)
        return p if p.is_absolute() else BASE_DIR / p

    @property
    def unet_model_abs_path(self) -> Path:
        p = Path(self.unet_model_path)
        return p if p.is_absolute() else BASE_DIR / p

    @property
    def uploads_dir(self) -> Path:
        return BASE_DIR / "uploads"

    @property
    def outputs_dir(self) -> Path:
        return BASE_DIR / "outputs"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


settings = Settings()

# Ensure runtime directories exist (safe, idempotent, no weights touched).
settings.uploads_dir.mkdir(parents=True, exist_ok=True)
settings.outputs_dir.mkdir(parents=True, exist_ok=True)
