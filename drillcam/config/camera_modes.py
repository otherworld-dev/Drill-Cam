"""Camera mode presets for OV9281 sensor."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CameraMode:
    """Defines a camera capture mode."""

    name: str
    width: int
    height: int
    fps: int
    description: str

    @property
    def frame_duration_us(self) -> int:
        """Frame duration in microseconds for FrameDurationLimits."""
        return int(1_000_000 / self.fps)

    @property
    def bytes_per_frame(self) -> int:
        """Bytes per frame (grayscale 8-bit)."""
        return self.width * self.height

    @property
    def data_rate_mbps(self) -> float:
        """Data rate in MB/s."""
        return (self.bytes_per_frame * self.fps) / (1024 * 1024)


# OV9281 sensor modes
# Note: Actual output dimensions may vary based on libcamera/sensor config.
# The frame buffer will auto-resize to match actual camera output.
CAMERA_MODES = {
    "high_speed": CameraMode(
        name="High Speed",
        width=640,
        height=400,
        fps=309,
        description="Maximum frame rate for vibration analysis",
    ),
    "high_resolution": CameraMode(
        name="High Resolution",
        width=1280,
        height=800,
        fps=120,
        description="Full resolution for detail inspection and measurement",
    ),
    "balanced": CameraMode(
        name="Balanced",
        width=1280,
        height=800,
        fps=60,
        description="Full resolution at moderate frame rate",
    ),
}

DEFAULT_MODE = "high_resolution"
