"""Thread-safe ring buffer for high-speed frame storage."""

import logging
import threading
from dataclasses import dataclass
from typing import Optional, Tuple, Iterator
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FrameMetadata:
    """Metadata for a stored frame."""

    timestamp: float
    frame_number: int
    valid: bool = True


class FrameRingBuffer:
    """
    Pre-allocated ring buffer for storing video frames.

    Designed for high-speed capture where allocation overhead must be avoided.
    Uses a single lock for thread safety with minimal contention.

    Attributes:
        capacity: Maximum number of frames the buffer can hold
        width: Frame width in pixels
        height: Frame height in pixels
    """

    def __init__(self, capacity: int, width: int, height: int) -> None:
        """
        Initialize the ring buffer.

        Args:
            capacity: Number of frames to allocate
            width: Frame width in pixels
            height: Frame height in pixels
        """
        self._capacity = capacity
        self._width = width
        self._height = height

        # Pre-allocate frame storage (grayscale uint8)
        self._frames = np.zeros((capacity, height, width), dtype=np.uint8)

        # Metadata for each frame
        self._metadata: list[FrameMetadata] = [
            FrameMetadata(timestamp=0.0, frame_number=0, valid=False)
            for _ in range(capacity)
        ]

        # Buffer state
        self._write_idx = 0
        self._count = 0  # Number of valid frames
        self._lock = threading.Lock()

        # Recording markers
        self._recording = False
        self._record_start_idx: Optional[int] = None
        self._record_start_frame: Optional[int] = None

        logger.info(
            f"FrameRingBuffer allocated: {capacity} frames, "
            f"{width}x{height}, {self.memory_usage_mb:.1f} MB"
        )

    @property
    def capacity(self) -> int:
        """Maximum number of frames."""
        return self._capacity

    @property
    def width(self) -> int:
        """Frame width."""
        return self._width

    @property
    def height(self) -> int:
        """Frame height."""
        return self._height

    @property
    def count(self) -> int:
        """Number of valid frames in buffer."""
        with self._lock:
            return self._count

    @property
    def memory_usage_mb(self) -> float:
        """Memory usage in megabytes."""
        return self._frames.nbytes / (1024 * 1024)

    @property
    def is_recording(self) -> bool:
        """Whether recording is active."""
        return self._recording

    def write(self, frame: np.ndarray, timestamp: float, frame_number: int) -> int:
        """
        Write a frame to the buffer.

        Args:
            frame: Grayscale frame data (will auto-resize buffer if dimensions change)
            timestamp: Frame timestamp
            frame_number: Sequential frame number

        Returns:
            Buffer index where frame was written
        """
        with self._lock:
            # Check if frame dimensions match buffer
            frame_h, frame_w = frame.shape[:2]
            if frame_h != self._height or frame_w != self._width:
                logger.warning(
                    f"Frame size changed: {frame_w}x{frame_h} vs buffer {self._width}x{self._height}. "
                    f"Resizing buffer..."
                )
                # Resize buffer to match actual frame dimensions
                self._width = frame_w
                self._height = frame_h
                self._frames = np.zeros((self._capacity, frame_h, frame_w), dtype=np.uint8)
                # Reset buffer state
                self._write_idx = 0
                self._count = 0
                for meta in self._metadata:
                    meta.valid = False
                logger.info(
                    f"Buffer resized to {self._capacity} frames, "
                    f"{frame_w}x{frame_h}, {self.memory_usage_mb:.1f} MB"
                )

            idx = self._write_idx

            # Copy frame data (single memcpy)
            np.copyto(self._frames[idx], frame)

            # Update metadata
            self._metadata[idx] = FrameMetadata(
                timestamp=timestamp,
                frame_number=frame_number,
                valid=True,
            )

            # Advance write pointer
            self._write_idx = (self._write_idx + 1) % self._capacity
            self._count = min(self._count + 1, self._capacity)

            return idx

    def read(self, idx: int) -> Optional[Tuple[np.ndarray, FrameMetadata]]:
        """
        Read a frame from the buffer.

        Args:
            idx: Buffer index to read

        Returns:
            Tuple of (frame_copy, metadata) or None if invalid
        """
        if idx < 0 or idx >= self._capacity:
            return None

        with self._lock:
            meta = self._metadata[idx]
            if not meta.valid:
                return None

            # Return a copy to avoid data races
            return self._frames[idx].copy(), meta

    def read_latest(self) -> Optional[Tuple[np.ndarray, FrameMetadata]]:
        """
        Read the most recently written frame.

        Returns:
            Tuple of (frame_copy, metadata) or None if buffer empty
        """
        with self._lock:
            if self._count == 0:
                return None

            # Latest frame is one before write pointer
            idx = (self._write_idx - 1) % self._capacity
            meta = self._metadata[idx]

            if not meta.valid:
                return None

            return self._frames[idx].copy(), meta

    def get_frame_view(self, idx: int) -> Optional[np.ndarray]:
        """
        Get a direct view of a frame (no copy).

        WARNING: Only use when you're sure buffer won't be modified.
        Caller must not hold the view across write operations.

        Args:
            idx: Buffer index

        Returns:
            View of frame array or None if invalid
        """
        if idx < 0 or idx >= self._capacity:
            return None

        with self._lock:
            if not self._metadata[idx].valid:
                return None
            return self._frames[idx]

    def start_recording(self, pre_frames: int = 0) -> int:
        """
        Mark the start of a recording.

        Args:
            pre_frames: Number of frames before current to include

        Returns:
            Starting frame number for the recording
        """
        with self._lock:
            self._recording = True

            # Calculate start index (accounting for pre-roll)
            frames_available = min(pre_frames, self._count - 1) if self._count > 0 else 0
            self._record_start_idx = (
                self._write_idx - 1 - frames_available
            ) % self._capacity

            # Get the frame number at start
            start_meta = self._metadata[self._record_start_idx]
            self._record_start_frame = start_meta.frame_number if start_meta.valid else 0

            logger.info(
                f"Recording started from idx {self._record_start_idx}, "
                f"frame {self._record_start_frame} (pre-roll: {frames_available})"
            )

            return self._record_start_frame

    def stop_recording(self) -> Tuple[int, int]:
        """
        Mark the end of a recording.

        Returns:
            Tuple of (start_frame_number, end_frame_number)
        """
        with self._lock:
            self._recording = False

            if self._record_start_frame is None:
                return (0, 0)

            # End frame is the latest frame
            end_idx = (self._write_idx - 1) % self._capacity
            end_meta = self._metadata[end_idx]
            end_frame = end_meta.frame_number if end_meta.valid else 0

            result = (self._record_start_frame, end_frame)

            logger.info(
                f"Recording stopped. Frames {result[0]} to {result[1]} "
                f"({result[1] - result[0] + 1} frames)"
            )

            self._record_start_idx = None
            self._record_start_frame = None

            return result

    def get_recording_frames(
        self, start_frame: int, end_frame: int
    ) -> Iterator[Tuple[np.ndarray, FrameMetadata]]:
        """
        Iterate over frames in a recording range.

        Args:
            start_frame: First frame number
            end_frame: Last frame number (inclusive)

        Yields:
            Tuples of (frame_copy, metadata)
        """
        with self._lock:
            # Find indices for frame range
            for idx in range(self._capacity):
                meta = self._metadata[idx]
                if meta.valid and start_frame <= meta.frame_number <= end_frame:
                    yield self._frames[idx].copy(), meta

    def clear(self) -> None:
        """Clear all frames from buffer."""
        with self._lock:
            self._write_idx = 0
            self._count = 0
            self._recording = False
            self._record_start_idx = None
            self._record_start_frame = None

            for meta in self._metadata:
                meta.valid = False

            logger.info("Buffer cleared")

    def resize(self, capacity: int, width: int, height: int) -> None:
        """
        Resize the buffer (clears existing data).

        Args:
            capacity: New frame capacity
            width: New frame width
            height: New frame height
        """
        with self._lock:
            self._capacity = capacity
            self._width = width
            self._height = height

            self._frames = np.zeros((capacity, height, width), dtype=np.uint8)
            self._metadata = [
                FrameMetadata(timestamp=0.0, frame_number=0, valid=False)
                for _ in range(capacity)
            ]

            self._write_idx = 0
            self._count = 0
            self._recording = False
            self._record_start_idx = None
            self._record_start_frame = None

            logger.info(
                f"Buffer resized: {capacity} frames, "
                f"{width}x{height}, {self.memory_usage_mb:.1f} MB"
            )


def create_buffer_for_duration(
    duration_seconds: float, fps: int, width: int, height: int
) -> FrameRingBuffer:
    """
    Create a buffer sized for a specific duration.

    Args:
        duration_seconds: How many seconds of video to buffer
        fps: Frame rate
        width: Frame width
        height: Frame height

    Returns:
        Appropriately sized FrameRingBuffer
    """
    capacity = int(duration_seconds * fps) + 1
    return FrameRingBuffer(capacity, width, height)
