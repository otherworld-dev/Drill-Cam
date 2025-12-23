"""Capture worker thread for continuous frame acquisition."""

import logging
import time
from typing import Optional

from PySide6.QtCore import QThread, Signal

from ..core.camera_controller import CameraController, FrameData
from ..core.frame_buffer import FrameRingBuffer

logger = logging.getLogger(__name__)


class CaptureWorker(QThread):
    """
    Worker thread for continuous camera capture.

    Runs capture loop on dedicated thread to avoid blocking UI.
    Writes frames to ring buffer and emits signals for UI updates.

    Signals:
        frame_captured(int): Emitted with frame number after each capture
        dropped_frames(int): Emitted when frames are dropped
        capture_error(str): Emitted on capture errors
        fps_updated(float): Emitted periodically with actual FPS
    """

    frame_captured = Signal(int)  # frame_number
    dropped_frames = Signal(int)  # count
    capture_error = Signal(str)  # error message
    fps_updated = Signal(float)  # actual fps

    def __init__(
        self,
        camera: CameraController,
        buffer: FrameRingBuffer,
        parent=None,
    ) -> None:
        """
        Initialize capture worker.

        Args:
            camera: Initialized camera controller
            buffer: Ring buffer to write frames to
            parent: Qt parent object
        """
        super().__init__(parent)

        self._camera = camera
        self._buffer = buffer

        self._running = False
        self._paused = False

        # Performance tracking
        self._frame_count = 0
        self._last_frame_number = -1
        self._fps_window_start = 0.0
        self._fps_window_frames = 0
        self._fps_update_interval = 1.0  # seconds

    @property
    def is_running(self) -> bool:
        """Check if capture loop is running."""
        return self._running

    @property
    def is_paused(self) -> bool:
        """Check if capture is paused."""
        return self._paused

    @property
    def frame_count(self) -> int:
        """Total frames captured in this session."""
        return self._frame_count

    def pause(self) -> None:
        """Pause capture (frames will be dropped)."""
        self._paused = True
        logger.info("Capture paused")

    def resume(self) -> None:
        """Resume capture after pause."""
        self._paused = False
        logger.info("Capture resumed")

    def stop(self) -> None:
        """Request capture loop to stop."""
        self._running = False
        logger.info("Capture stop requested")

    def run(self) -> None:
        """Main capture loop (runs in worker thread)."""
        logger.info("Capture worker starting")

        self._running = True
        self._frame_count = 0
        self._last_frame_number = -1
        self._fps_window_start = time.monotonic()
        self._fps_window_frames = 0

        # Start camera streaming
        if not self._camera.start_streaming():
            self.capture_error.emit("Failed to start camera streaming")
            self._running = False
            return

        consecutive_errors = 0
        max_consecutive_errors = 10

        try:
            while self._running:
                if self._paused:
                    # When paused, still capture but don't store
                    # This keeps the camera pipeline flowing
                    try:
                        self._camera.capture_frame()
                    except Exception:
                        pass
                    time.sleep(0.001)
                    continue

                # Capture frame with error tracking
                try:
                    frame_data = self._camera.capture_frame()
                except Exception as e:
                    consecutive_errors += 1
                    logger.warning(f"Frame capture exception: {e}")
                    if consecutive_errors >= max_consecutive_errors:
                        raise RuntimeError(f"Too many consecutive capture errors: {e}")
                    time.sleep(0.01)
                    continue

                if frame_data is None:
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        raise RuntimeError("Camera not returning frames")
                    time.sleep(0.001)
                    continue

                # Reset error counter on successful capture
                consecutive_errors = 0

                # Check for dropped frames
                if self._last_frame_number >= 0:
                    expected = self._last_frame_number + 1
                    if frame_data.frame_number > expected:
                        dropped = frame_data.frame_number - expected
                        self.dropped_frames.emit(dropped)
                        logger.warning(f"Dropped {dropped} frames")

                self._last_frame_number = frame_data.frame_number

                # Write to buffer
                self._buffer.write(
                    frame_data.array,
                    frame_data.timestamp,
                    frame_data.frame_number,
                )

                self._frame_count += 1
                self._fps_window_frames += 1

                # Emit frame captured signal (not every frame to reduce overhead)
                if self._frame_count % 5 == 0:
                    self.frame_captured.emit(frame_data.frame_number)

                # Update FPS periodically
                now = time.monotonic()
                elapsed = now - self._fps_window_start
                if elapsed >= self._fps_update_interval:
                    fps = self._fps_window_frames / elapsed
                    self.fps_updated.emit(fps)
                    self._fps_window_start = now
                    self._fps_window_frames = 0

        except Exception as e:
            logger.error(f"Capture error: {e}")
            self.capture_error.emit(str(e))

        finally:
            self._camera.stop_streaming()
            self._running = False
            logger.info(f"Capture worker stopped. Total frames: {self._frame_count}")


class PreviewWorker(QThread):
    """
    Worker thread for converting frames to display format.

    Decouples display rate from capture rate by processing frames
    at a fixed target FPS regardless of capture speed.

    Signals:
        frame_ready(object): Emitted with QImage ready for display
    """

    frame_ready = Signal(object)  # QImage

    def __init__(
        self,
        buffer: FrameRingBuffer,
        target_fps: int = 30,
        parent=None,
    ) -> None:
        """
        Initialize preview worker.

        Args:
            buffer: Ring buffer to read frames from
            target_fps: Target display frame rate
            parent: Qt parent object
        """
        super().__init__(parent)

        self._buffer = buffer
        self._target_fps = target_fps
        self._frame_interval = 1.0 / target_fps

        self._running = False
        self._last_frame_number = -1

    def stop(self) -> None:
        """Request worker to stop."""
        self._running = False

    def run(self) -> None:
        """Main preview loop."""
        from PySide6.QtGui import QImage

        logger.info(f"Preview worker starting at {self._target_fps} fps")

        self._running = True
        next_frame_time = time.monotonic()

        while self._running:
            now = time.monotonic()

            # Wait until it's time for next frame
            if now < next_frame_time:
                time.sleep(next_frame_time - now)
                continue

            next_frame_time = now + self._frame_interval

            # Get latest frame from buffer
            result = self._buffer.read_latest()
            if result is None:
                continue

            frame, meta = result

            # Skip if same frame as last time
            if meta.frame_number == self._last_frame_number:
                continue

            self._last_frame_number = meta.frame_number

            # Convert numpy array to QImage
            height, width = frame.shape
            bytes_per_line = width

            qimage = QImage(
                frame.data,
                width,
                height,
                bytes_per_line,
                QImage.Format.Format_Grayscale8,
            ).copy()  # Copy to detach from numpy buffer

            self.frame_ready.emit(qimage)

        logger.info("Preview worker stopped")
