"""Application settings using pydantic."""

import json
import logging
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .camera_modes import DEFAULT_MODE

logger = logging.getLogger(__name__)


def get_config_dir() -> Path:
    """Get the configuration directory for DrillCam."""
    # Use XDG config directory on Linux, or home/.drillcam elsewhere
    import os
    import sys

    if sys.platform == "linux":
        config_home = os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")
        return Path(config_home) / "drillcam"
    else:
        return Path.home() / ".drillcam"


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_prefix="DRILLCAM_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Camera settings
    camera_mode: str = Field(default=DEFAULT_MODE, description="Active camera mode")
    buffer_count: int = Field(default=6, ge=4, le=16, description="Camera buffer count")

    # Recording settings
    output_dir: Path = Field(
        default_factory=lambda: Path.home() / "Videos" / "DrillCam",
        description="Directory for saved recordings",
    )
    use_ramdisk: bool = Field(
        default=True, description="Use /dev/shm for temporary recording buffer"
    )
    ramdisk_path: Path = Field(
        default=Path("/dev/shm/drillcam"), description="Ramdisk path for temp files"
    )
    pre_record_seconds: float = Field(
        default=2.0, ge=0, le=30, description="Seconds to keep in buffer before trigger"
    )

    # Display settings
    preview_fps: int = Field(
        default=30, ge=10, le=60, description="Target preview frame rate"
    )
    show_crosshair: bool = Field(default=True, description="Show crosshair overlay")
    show_info: bool = Field(default=True, description="Show info overlay")

    # Storage settings
    ssd_path: Optional[Path] = Field(
        default=None, description="Path to NVMe SSD for longer recordings"
    )

    # Calibration (pixels per mm)
    calibration_pixels_per_unit: float = Field(
        default=0.0, description="Calibration: pixels per unit"
    )
    calibration_unit: str = Field(default="mm", description="Calibration unit")

    # Window geometry
    window_width: int = Field(default=1024, description="Main window width")
    window_height: int = Field(default=768, description="Main window height")
    window_x: int = Field(default=100, description="Main window X position")
    window_y: int = Field(default=100, description="Main window Y position")

    def get_recording_path(self) -> Path:
        """Get the appropriate recording path based on settings."""
        if self.ssd_path and self.ssd_path.exists():
            return self.ssd_path / "drillcam_capture"
        if self.use_ramdisk:
            return self.ramdisk_path
        return self.output_dir / "capture_temp"

    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        recording_path = self.get_recording_path()
        recording_path.mkdir(parents=True, exist_ok=True)

        # Create config directory
        config_dir = get_config_dir()
        config_dir.mkdir(parents=True, exist_ok=True)

    def save(self) -> None:
        """Save settings to config file."""
        config_dir = get_config_dir()
        config_dir.mkdir(parents=True, exist_ok=True)

        config_file = config_dir / "settings.json"

        # Convert to dict, handling Path objects
        data = {}
        for key, value in self.model_dump().items():
            if isinstance(value, Path):
                data[key] = str(value)
            else:
                data[key] = value

        with open(config_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Settings saved to {config_file}")

    @classmethod
    def load(cls) -> "Settings":
        """Load settings from config file, falling back to defaults."""
        config_file = get_config_dir() / "settings.json"

        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    data = json.load(f)

                # Convert string paths back to Path objects
                path_fields = [
                    "output_dir",
                    "ramdisk_path",
                    "ssd_path",
                ]
                for field in path_fields:
                    if field in data and data[field] is not None:
                        data[field] = Path(data[field])

                logger.info(f"Settings loaded from {config_file}")
                return cls(**data)

            except Exception as e:
                logger.warning(f"Failed to load settings: {e}, using defaults")

        return cls()


class CalibrationData:
    """Manages calibration data persistence."""

    def __init__(self, config_dir: Optional[Path] = None) -> None:
        self._config_dir = config_dir or get_config_dir()
        self._calibrations: dict = {}
        self._load()

    def _calibration_file(self) -> Path:
        return self._config_dir / "calibrations.json"

    def _load(self) -> None:
        """Load calibrations from file."""
        cal_file = self._calibration_file()
        if cal_file.exists():
            try:
                with open(cal_file, "r") as f:
                    self._calibrations = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load calibrations: {e}")

    def save(self) -> None:
        """Save calibrations to file."""
        self._config_dir.mkdir(parents=True, exist_ok=True)
        cal_file = self._calibration_file()

        with open(cal_file, "w") as f:
            json.dump(self._calibrations, f, indent=2)

    def set_calibration(
        self,
        name: str,
        pixels_per_unit: float,
        unit: str,
        resolution: tuple,
    ) -> None:
        """
        Save a calibration.

        Args:
            name: Calibration name/identifier
            pixels_per_unit: Pixels per real-world unit
            unit: Unit of measurement
            resolution: (width, height) this calibration applies to
        """
        self._calibrations[name] = {
            "pixels_per_unit": pixels_per_unit,
            "unit": unit,
            "resolution": list(resolution),
        }
        self.save()

    def get_calibration(self, name: str) -> Optional[dict]:
        """Get a saved calibration by name."""
        return self._calibrations.get(name)

    def get_calibration_for_resolution(
        self,
        width: int,
        height: int,
    ) -> Optional[dict]:
        """Find calibration matching a resolution."""
        for cal in self._calibrations.values():
            res = cal.get("resolution", [])
            if len(res) == 2 and res[0] == width and res[1] == height:
                return cal
        return None

    def list_calibrations(self) -> list:
        """List all saved calibration names."""
        return list(self._calibrations.keys())
