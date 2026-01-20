"""Camera controller wrapping picamera2 for OV9281 sensor."""

import logging
from dataclasses import dataclass
from typing import Callable, Optional, List
import time

import numpy as np

try:
    from picamera2 import Picamera2
    from libcamera import controls

    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    Picamera2 = None

from ..config.camera_modes import CameraMode, CAMERA_MODES, DEFAULT_MODE

logger = logging.getLogger(__name__)


@dataclass
class FrameData:
    """Container for a captured frame with metadata."""

    array: np.ndarray
    timestamp: float
    frame_number: int


class CameraController:
    """
    Controls the OV9281 camera via picamera2.

    Provides abstraction over picamera2 for:
    - Camera initialization and configuration
    - Mode switching (resolution/fps)
    - Frame callbacks with minimal overhead
    - Resource cleanup
    """

    def __init__(self) -> None:
        self._camera: Optional[Picamera2] = None
        self._current_mode: Optional[CameraMode] = None
        self._frame_callback: Optional[Callable[[FrameData], None]] = None
        self._frame_count: int = 0
        self._is_streaming: bool = False
        self._start_time: float = 0.0

    @property
    def is_available(self) -> bool:
        """Check if picamera2 is available."""
        return PICAMERA2_AVAILABLE

    @property
    def is_initialized(self) -> bool:
        """Check if camera is initialized."""
        return self._camera is not None

    @property
    def is_streaming(self) -> bool:
        """Check if camera is currently streaming."""
        return self._is_streaming

    @property
    def current_mode(self) -> Optional[CameraMode]:
        """Get the current camera mode."""
        return self._current_mode

    def get_available_modes(self) -> List[CameraMode]:
        """Get list of available camera modes."""
        return list(CAMERA_MODES.values())

    def initialize(self, camera_id: int = 0) -> bool:
        """
        Initialize the camera.

        Args:
            camera_id: Camera index (default 0 for single camera)

        Returns:
            True if initialization successful
        """
        if not PICAMERA2_AVAILABLE:
            logger.error("picamera2 not available - running on non-Pi system?")
            return False

        try:
            self._camera = Picamera2(camera_id)
            logger.info(f"Camera initialized: {self._camera.camera_properties}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            self._camera = None
            return False

    def configure(self, mode_name: str = DEFAULT_MODE) -> bool:
        """
        Configure camera for specified mode.

        Args:
            mode_name: Key from CAMERA_MODES dict

        Returns:
            True if configuration successful
        """
        if not self._camera:
            logger.error("Camera not initialized")
            return False

        if mode_name not in CAMERA_MODES:
            logger.error(f"Unknown mode: {mode_name}")
            return False

        mode = CAMERA_MODES[mode_name]

        try:
            # Stop streaming if active
            if self._is_streaming:
                self.stop_streaming()

            # Configure camera based on mode requirements
            # High-speed modes (>= 200 fps) need special configuration for performance
            # Always disable TDN (Temporal Denoise) to avoid Pi 5 backend crash
            if mode.fps >= 200:
                # High-speed: optimize for maximum frame rate
                logger.info(f"Configuring high-speed mode: {mode.fps}fps")
                config = self._camera.create_video_configuration(
                    main={"size": (mode.width, mode.height), "format": "YUV420"},
                    lores={"size": (mode.width, mode.height), "format": "YUV420"},
                    buffer_count=8,  # More buffers for high-speed throughput
                    queue=False,  # Get latest frame, don't queue (reduces latency)
                    controls={
                        "FrameDurationLimits": (
                            mode.frame_duration_us,
                            mode.frame_duration_us,
                        ),
                        "NoiseReductionMode": 0,  # Disable for speed and TDN fix
                    },
                )
            else:
                # Normal speed: standard configuration
                config = self._camera.create_video_configuration(
                    main={"size": (mode.width, mode.height), "format": "YUV420"},
                    buffer_count=6,
                    controls={
                        "FrameDurationLimits": (
                            mode.frame_duration_us,
                            mode.frame_duration_us,
                        ),
                        "NoiseReductionMode": 0,  # Always disable TDN for Pi 5
                    },
                )

            self._camera.configure(config)
            self._current_mode = mode

            # Log actual sensor mode selected for diagnostics
            sensor_mode = self._camera.camera_configuration().get("sensor", {})
            sensor_config = self._camera.camera_configuration()
            logger.info(
                f"Camera configured: {mode.name} ({mode.width}x{mode.height} @ {mode.fps}fps)"
            )
            logger.info(f"Sensor mode: {sensor_mode}")
            logger.info(f"Buffer count: {sensor_config.get('buffer_count', 'unknown')}")
            logger.info(f"Queue mode: {sensor_config.get('queue', 'unknown')}")

            # Validate sensor can achieve requested frame rate
            sensor_fps = sensor_mode.get("fps", 0) if sensor_mode else 0
            if sensor_fps > 0 and abs(sensor_fps - mode.fps) > mode.fps * 0.1:
                logger.warning(
                    f"Sensor mode FPS ({sensor_fps}) differs from requested "
                    f"({mode.fps}). Actual performance may vary."
                )

            return True

        except Exception as e:
            logger.error(f"Failed to configure camera: {e}", exc_info=True)
            return False

    def start_streaming(
        self, callback: Optional[Callable[[FrameData], None]] = None
    ) -> bool:
        """
        Start camera streaming.

        Args:
            callback: Optional callback for each frame. Called with FrameData.
                     Keep callback fast to avoid dropping frames.

        Returns:
            True if streaming started successfully
        """
        if not self._camera:
            logger.error("Camera not initialized")
            return False

        if self._is_streaming:
            logger.warning("Already streaming")
            return True

        try:
            self._frame_callback = callback
            self._frame_count = 0
            self._start_time = time.monotonic()

            # Start camera
            self._camera.start()
            self._is_streaming = True

            logger.info("Camera streaming started")
            return True

        except Exception as e:
            logger.error(f"Failed to start streaming: {e}")
            return False

    def stop_streaming(self) -> None:
        """Stop camera streaming."""
        if not self._camera or not self._is_streaming:
            return

        try:
            self._camera.stop()
            self._is_streaming = False

            elapsed = time.monotonic() - self._start_time
            if elapsed > 0:
                actual_fps = self._frame_count / elapsed
                logger.info(
                    f"Streaming stopped. Captured {self._frame_count} frames "
                    f"in {elapsed:.1f}s ({actual_fps:.1f} fps)"
                )

        except Exception as e:
            logger.error(f"Error stopping stream: {e}")

    def capture_frame(self) -> Optional[FrameData]:
        """
        Capture a single frame.

        Returns:
            FrameData with the captured frame, or None on error
        """
        if not self._camera or not self._is_streaming:
            return None

        try:
            # Use capture_array for simple access
            array = self._camera.capture_array("main")

            # Handle YUV420 format - extract Y plane (luminance/grayscale)
            # YUV420 from picamera2 can come in different layouts:
            # - Shape (h, w, 3) = interleaved, take channel 0
            # - Shape (h * 1.5, w) = planar, Y is first h rows
            # - Shape (h, w) = already grayscale
            if len(array.shape) == 3:
                # Interleaved format: take Y channel (first channel)
                gray = array[:, :, 0]
            elif len(array.shape) == 2:
                # Check if this is planar YUV420 (height is 1.5x normal)
                # If so, Y plane is the first 2/3 of the data
                h, w = array.shape
                # Try to detect if it's planar by checking if height seems too large
                if self._current_mode:
                    expected_h = self._current_mode.height
                    if h > expected_h * 1.2:  # Likely planar YUV420
                        gray = array[:expected_h, :]
                    else:
                        gray = array
                else:
                    # Assume it's already grayscale
                    gray = array
            else:
                gray = array

            self._frame_count += 1
            timestamp = time.monotonic() - self._start_time

            frame_data = FrameData(
                array=gray.copy(),  # Copy to avoid buffer reuse issues
                timestamp=timestamp,
                frame_number=self._frame_count,
            )

            if self._frame_callback:
                self._frame_callback(frame_data)

            return frame_data

        except Exception as e:
            logger.error(f"Frame capture error: {e}")
            return None

    def capture_still(self) -> Optional[np.ndarray]:
        """
        Capture a single still frame at full quality.

        Returns:
            Grayscale numpy array, or None on error
        """
        frame_data = self.capture_frame()
        return frame_data.array if frame_data else None

    def close(self) -> None:
        """Release camera resources."""
        if self._camera:
            self.stop_streaming()
            try:
                self._camera.close()
            except Exception as e:
                logger.error(f"Error closing camera: {e}")
            finally:
                self._camera = None
                self._current_mode = None
                logger.info("Camera closed")

    def __enter__(self) -> "CameraController":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class MockCameraController(CameraController):
    """
    Mock camera controller for testing without hardware.

    Generates synthetic grayscale frames at the configured rate.
    """

    def __init__(self) -> None:
        super().__init__()
        self._mock_enabled = True

    @property
    def is_available(self) -> bool:
        return True

    def initialize(self, camera_id: int = 0) -> bool:
        logger.info("Mock camera initialized")
        return True

    def configure(self, mode_name: str = DEFAULT_MODE) -> bool:
        if mode_name not in CAMERA_MODES:
            logger.error(f"Unknown mode: {mode_name}")
            return False

        self._current_mode = CAMERA_MODES[mode_name]
        logger.info(f"Mock camera configured: {self._current_mode.name}")
        return True

    def start_streaming(
        self, callback: Optional[Callable[[FrameData], None]] = None
    ) -> bool:
        self._frame_callback = callback
        self._frame_count = 0
        self._start_time = time.monotonic()
        self._is_streaming = True
        logger.info("Mock camera streaming started")
        return True

    def capture_frame(self) -> Optional[FrameData]:
        if not self._is_streaming or not self._current_mode:
            return None

        mode = self._current_mode
        self._frame_count += 1
        timestamp = time.monotonic() - self._start_time

        # Generate synthetic test pattern
        array = self._generate_test_frame(mode.width, mode.height, self._frame_count)

        frame_data = FrameData(
            array=array,
            timestamp=timestamp,
            frame_number=self._frame_count,
        )

        if self._frame_callback:
            self._frame_callback(frame_data)

        return frame_data

    def _generate_test_frame(
        self, width: int, height: int, frame_num: int
    ) -> np.ndarray:
        """Generate a synthetic test frame with moving pattern."""
        # Create gradient with moving element
        y = np.linspace(0, 255, height, dtype=np.uint8)
        frame = np.tile(y.reshape(-1, 1), (1, width))

        # Add moving vertical bar
        bar_pos = (frame_num * 3) % width
        bar_width = 20
        start = max(0, bar_pos - bar_width // 2)
        end = min(width, bar_pos + bar_width // 2)
        frame[:, start:end] = 255

        # Add frame counter text area
        frame[10:40, 10:150] = 0

        return frame

    def close(self) -> None:
        self._is_streaming = False
        self._current_mode = None
        logger.info("Mock camera closed")


def create_camera_controller(use_mock: bool = False) -> CameraController:
    """
    Factory function to create appropriate camera controller.

    Args:
        use_mock: Force mock controller even if real hardware available

    Returns:
        CameraController or MockCameraController instance
    """
    if use_mock or not PICAMERA2_AVAILABLE:
        logger.info("Using mock camera controller")
        return MockCameraController()
    return CameraController()
