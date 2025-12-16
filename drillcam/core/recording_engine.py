"""Recording engine for two-phase capture and encoding."""

import logging
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, List
from enum import Enum, auto

import numpy as np
from PySide6.QtCore import QThread, Signal, QObject

from .frame_buffer import FrameRingBuffer, FrameMetadata
from ..utils.video_io import FFmpegEncoder, encode_raw_frames, EncodingProgress

logger = logging.getLogger(__name__)


class RecordingState(Enum):
    """Recording state machine."""

    IDLE = auto()
    CAPTURING = auto()
    ENCODING = auto()
    COMPLETE = auto()
    ERROR = auto()


@dataclass
class RecordingInfo:
    """Information about a recording."""

    start_time: datetime
    end_time: Optional[datetime]
    frame_count: int
    fps: int
    width: int
    height: int
    raw_path: Path
    output_path: Optional[Path]
    state: RecordingState
    error_message: Optional[str] = None

    @property
    def duration_seconds(self) -> float:
        """Recording duration in seconds."""
        return self.frame_count / self.fps if self.fps > 0 else 0

    @property
    def file_size_mb(self) -> float:
        """Estimated file size in MB."""
        if self.output_path and self.output_path.exists():
            return self.output_path.stat().st_size / (1024 * 1024)
        return 0


class RecordingEngine(QObject):
    """
    Manages the two-phase recording process.

    Phase 1: During capture - write raw frames to fast storage
    Phase 2: After capture - encode to FFV1 lossless video
    """

    # Signals
    state_changed = Signal(RecordingState)
    capture_progress = Signal(int, int)  # current_frame, total_frames
    encoding_progress = Signal(EncodingProgress)
    recording_complete = Signal(RecordingInfo)
    recording_error = Signal(str)

    def __init__(
        self,
        buffer: FrameRingBuffer,
        raw_storage_path: Path,
        output_dir: Path,
        parent=None,
    ) -> None:
        """
        Initialize recording engine.

        Args:
            buffer: Frame ring buffer to capture from
            raw_storage_path: Fast storage for raw frames (e.g., /dev/shm or SSD)
            output_dir: Final output directory for encoded videos
        """
        super().__init__(parent)

        self._buffer = buffer
        self._raw_storage_path = raw_storage_path
        self._output_dir = output_dir

        self._state = RecordingState.IDLE
        self._current_recording: Optional[RecordingInfo] = None
        self._encoder_thread: Optional[EncoderThread] = None

        # Ensure directories exist
        self._raw_storage_path.mkdir(parents=True, exist_ok=True)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def state(self) -> RecordingState:
        """Current recording state."""
        return self._state

    @property
    def is_recording(self) -> bool:
        """Check if actively capturing or encoding."""
        return self._state in (RecordingState.CAPTURING, RecordingState.ENCODING)

    @property
    def current_recording(self) -> Optional[RecordingInfo]:
        """Get current recording info."""
        return self._current_recording

    def _set_state(self, state: RecordingState) -> None:
        """Update state and emit signal."""
        self._state = state
        self.state_changed.emit(state)

    def start_capture(
        self,
        fps: int,
        width: int,
        height: int,
        pre_roll_frames: int = 0,
    ) -> bool:
        """
        Start capturing frames.

        Args:
            fps: Capture frame rate
            width: Frame width
            height: Frame height
            pre_roll_frames: Number of buffered frames to include before trigger

        Returns:
            True if capture started successfully
        """
        if self._state != RecordingState.IDLE:
            logger.warning(f"Cannot start capture in state {self._state}")
            return False

        # Create timestamped directory for this recording
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_path = self._raw_storage_path / f"capture_{timestamp}"
        raw_path.mkdir(parents=True, exist_ok=True)

        self._current_recording = RecordingInfo(
            start_time=datetime.now(),
            end_time=None,
            frame_count=0,
            fps=fps,
            width=width,
            height=height,
            raw_path=raw_path,
            output_path=None,
            state=RecordingState.CAPTURING,
        )

        # Mark recording start in buffer (includes pre-roll)
        self._buffer.start_recording(pre_roll_frames)

        self._set_state(RecordingState.CAPTURING)
        logger.info(f"Capture started: {raw_path} (pre-roll: {pre_roll_frames} frames)")

        return True

    def stop_capture(self) -> Optional[RecordingInfo]:
        """
        Stop capturing and begin encoding.

        Returns:
            RecordingInfo if successful, None on error
        """
        if self._state != RecordingState.CAPTURING:
            logger.warning(f"Cannot stop capture in state {self._state}")
            return None

        if not self._current_recording:
            return None

        # Get recording range from buffer
        start_frame, end_frame = self._buffer.stop_recording()
        frame_count = end_frame - start_frame + 1

        self._current_recording.end_time = datetime.now()
        self._current_recording.frame_count = frame_count

        logger.info(f"Capture stopped: {frame_count} frames")

        # Dump frames to disk
        self._set_state(RecordingState.ENCODING)
        self._dump_frames_and_encode(start_frame, end_frame)

        return self._current_recording

    def _dump_frames_and_encode(self, start_frame: int, end_frame: int) -> None:
        """Dump buffered frames to disk and start encoding."""
        if not self._current_recording:
            return

        recording = self._current_recording
        raw_path = recording.raw_path

        # Get actual frame dimensions from buffer (may have changed due to auto-resize)
        actual_width = self._buffer.width
        actual_height = self._buffer.height

        # Update recording info with actual dimensions
        recording.width = actual_width
        recording.height = actual_height

        # Dump frames from buffer to raw storage
        logger.info(f"Dumping frames {start_frame} to {end_frame} (size: {actual_width}x{actual_height})...")

        frame_idx = 0
        for frame, meta in self._buffer.get_recording_frames(start_frame, end_frame):
            frame_path = raw_path / f"frame_{frame_idx:08d}.npy"
            np.save(frame_path, frame)
            frame_idx += 1

            if frame_idx % 100 == 0:
                self.capture_progress.emit(frame_idx, recording.frame_count)

        # Update actual frame count
        recording.frame_count = frame_idx
        logger.info(f"Dumped {frame_idx} frames to {raw_path}")

        if frame_idx == 0:
            self._on_encoding_error("No frames captured")
            return

        # Generate output filename
        timestamp = recording.start_time.strftime("%Y%m%d_%H%M%S")
        output_path = self._output_dir / f"drillcam_{timestamp}.mkv"
        recording.output_path = output_path

        # Start encoding in background thread
        self._encoder_thread = EncoderThread(
            raw_path=raw_path,
            output_path=output_path,
            width=actual_width,
            height=actual_height,
            fps=recording.fps,
        )
        self._encoder_thread.progress.connect(self._on_encoding_progress)
        self._encoder_thread.finished_signal.connect(self._on_encoding_finished)
        self._encoder_thread.error.connect(self._on_encoding_error)
        self._encoder_thread.start()

    def _on_encoding_progress(self, progress: EncodingProgress) -> None:
        """Handle encoding progress update."""
        self.encoding_progress.emit(progress)

    def _on_encoding_finished(self, output_path: str) -> None:
        """Handle encoding completion."""
        if self._current_recording:
            self._current_recording.state = RecordingState.COMPLETE

            # Clean up raw frames
            if self._current_recording.raw_path.exists():
                shutil.rmtree(self._current_recording.raw_path)
                logger.info(f"Cleaned up raw frames: {self._current_recording.raw_path}")

            self.recording_complete.emit(self._current_recording)

        self._set_state(RecordingState.COMPLETE)
        self._encoder_thread = None

        # Return to idle after a moment
        self._set_state(RecordingState.IDLE)

    def _on_encoding_error(self, error_msg: str) -> None:
        """Handle encoding error."""
        if self._current_recording:
            self._current_recording.state = RecordingState.ERROR
            self._current_recording.error_message = error_msg

        self._set_state(RecordingState.ERROR)
        self.recording_error.emit(error_msg)
        self._encoder_thread = None

    def cancel(self) -> None:
        """Cancel current recording/encoding."""
        if self._encoder_thread and self._encoder_thread.isRunning():
            self._encoder_thread.cancel()
            self._encoder_thread.wait(5000)

        if self._state == RecordingState.CAPTURING:
            self._buffer.stop_recording()

        # Clean up raw frames if they exist
        if self._current_recording and self._current_recording.raw_path.exists():
            shutil.rmtree(self._current_recording.raw_path)

        self._current_recording = None
        self._set_state(RecordingState.IDLE)
        logger.info("Recording cancelled")


class EncoderThread(QThread):
    """Background thread for FFV1 encoding."""

    progress = Signal(EncodingProgress)
    finished_signal = Signal(str)  # output path
    error = Signal(str)

    def __init__(
        self,
        raw_path: Path,
        output_path: Path,
        width: int,
        height: int,
        fps: int,
    ) -> None:
        super().__init__()

        self._raw_path = raw_path
        self._output_path = output_path
        self._width = width
        self._height = height
        self._fps = fps
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation."""
        self._cancelled = True

    def run(self) -> None:
        """Run encoding in background."""
        try:
            success = encode_raw_frames(
                frame_dir=self._raw_path,
                output_path=self._output_path,
                width=self._width,
                height=self._height,
                fps=self._fps,
                progress_callback=self._on_progress,
            )

            if self._cancelled:
                # Clean up partial output
                if self._output_path.exists():
                    self._output_path.unlink()
                return

            if success:
                self.finished_signal.emit(str(self._output_path))
            else:
                self.error.emit("Encoding failed")

        except Exception as e:
            logger.exception("Encoding thread error")
            self.error.emit(str(e))

    def _on_progress(self, progress: EncodingProgress) -> None:
        """Forward progress to signal."""
        if not self._cancelled:
            self.progress.emit(progress)
