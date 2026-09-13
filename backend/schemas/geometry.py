"""
Sonar platform / geometry metadata.

This is accepted from the caller and echoed back verbatim. Nothing in this
module invents, defaults, or estimates any of these values — they either
come from the caller or they are absent. Per PLAN.md 1.9b, side-scan
geometry is used only for the classical shadow-height computation
(shadow_service.py); it is never used to derive depth, latitude, or
longitude on its own (spatial_service.py is not implemented).
"""
from typing import Optional

from pydantic import BaseModel, Field


class SonarGeometry(BaseModel):
    """All fields optional. Never populate a field the caller did not supply."""

    towfish_altitude_m: Optional[float] = Field(
        default=None, description="Height of the towfish above the seabed (H)."
    )
    slant_range_m: Optional[float] = Field(
        default=None, description="Slant range from towfish to the target (R)."
    )
    sound_speed_mps: Optional[float] = Field(
        default=None, description="Speed of sound in water used for ranging, m/s."
    )
    latitude: Optional[float] = Field(default=None, description="Platform latitude, decimal degrees.")
    longitude: Optional[float] = Field(default=None, description="Platform longitude, decimal degrees.")
    heading_deg: Optional[float] = Field(default=None, description="Platform heading, degrees.")
    across_track_resolution_m_per_px: Optional[float] = Field(
        default=None,
        description="Ground-range metres represented by one across-track (range-axis) pixel.",
    )
