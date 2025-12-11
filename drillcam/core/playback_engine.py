"""Playback engine for video review with slow-motion support."""

import logging
import time
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal, QObject, QTimer

logger = logging.getLogger(__name__)


class PlaybackState(Enum):
    """Playback state."""

    STOPPED = auto()
    PLAYING = auto()
    PAUSED = auto()


@dataclass
class VideoInfo:
    """Information about a loaded video."""

    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float

    @classmethod
    def from_capture(cls, cap: cv2.VideoCapture, path: Path) -> "VideoInfo":
        """Create VideoInfo from OpenCV capture."""
        return cls(
            path=path,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=cap.get(cv2.CAP_PROP_FPS),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            duration_seconds=cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
            if cap.get(cv2.CAP_PROP_FPS) > 0
            else 0,
        )


class PlaybackEngine(QObject):
    """
    Video playback engine with variable speed control.

    Supports:
    - Play/pause/stop
    - Variable playback speed (0.1x to 2x)
    - Frame-by-frame stepping
    - Seeking to specific frames
    - Loop playback
    """

    # Signals
    frame_ready = Signal(np.ndarray, int)  # frame, frame_number
    position_changed = Signal(int)  # frame_number
    state_changed = Signal(PlaybackState)
    video_loaded = Signal(VideoInfo)
    video_ended = Signal()
    error = Signal(str)

    # Speed presets
    SPEED_PRESETS = [0.1, 0.25, 0.5, 1.0, 2.0]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._capture: Optional[cv2.VideoCapture] = None
        self._video_info: Optional[VideoInfo] = None
        self._state = PlaybackState.STOPPED

        self._current_frame = 0
        self._playback_speed = 1.0
        self._loop = False

        # Playback timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer_tick)

    @property
    def state(self) -> PlaybackState:
        """Current playback state."""
        return self._state

    @property
    def video_info(self) -> Optional[VideoInfo]:
        """Information about loaded video."""
        return self._video_info

    @property
    def current_frame(self) -> int:
        """Current frame number (0-indexed)."""
        return self._current_frame

    @property
    def playback_speed(self) -> float:
        """Current playback speed multiplier."""
        return self._playback_speed

    @property
    def is_loaded(self) -> bool:
        """Check if a video is loaded."""
        return self._capture is not None and self._video_info is not None

    @property
    def loop(self) -> bool:
        """Whether loop playback is enabled."""
        return self._loop

    @loop.setter
    def loop(self, value: bool) -> None:
        """Set loop playback."""
        self._loop = value

    def load_video(self, path: Path) -> bool:
        """
        Load a video file.

        Args:
            path: Path to video file

        Returns:
            True if loaded successfully
        """
        self.unload()

        try:
            cap = cv2.VideoCapture(str(path))

            if not cap.isOpened():
                self.error.emit(f"Failed to open video: {path}")
                return False

            self._capture = cap
            self._video_info = VideoInfo.from_capture(cap, path)
            self._current_frame = 0

            logger.info(
                f"Video loaded: {path.name} "
                f"({self._video_info.width}x{self._video_info.height}, "
                f"{self._video_info.fps:.1f}fps, "
                f"{self._video_info.frame_count} frames)"
            )

            self.video_loaded.emit(self._video_info)

            # Read and emit first frame
            self._read_and_emit_frame()

            return True

        except Exception as e:
            logger.error(f"Error loading video: {e}")
            self.error.emit(str(e))
            return False

    def unload(self) -> None:
        """Unload the current video."""
        self.stop()

        if self._capture:
            self._capture.release()
            self._capture = None

        self._video_info = None
        self._current_frame = 0

    def play(self) -> None:
        """Start or resume playback."""
        if not self.is_loaded:
            return

        if self._state == PlaybackState.PLAYING:
            return

        self._set_state(PlaybackState.PLAYING)
        self._start_timer()
        logger.debug("Playback started")

    def pause(self) -> None:
        """Pause playback."""
        if self._state != PlaybackState.PLAYING:
            return

        self._timer.stop()
        self._set_state(PlaybackState.PAUSED)
        logger.debug("Playback paused")

    def stop(self) -> None:
        """Stop playback and reset to beginning."""
        self._timer.stop()
        self._set_state(PlaybackState.STOPPED)

        if self.is_loaded:
            self.seek_to_frame(0)

        logger.debug("Playback stopped")

    def toggle_play_pause(self) -> None:
        """Toggle between play and pause."""
        if self._state == PlaybackState.PLAYING:
            self.pause()
        else:
            self.play()

    def set_speed(self, speed: float) -> None:
        """
        Set playback speed.

        Args:
            speed: Speed multiplier (0.1 to 2.0)
        """
        self._playback_speed = max(0.1, min(2.0, speed))

        # Restart timer with new interval if playing
        if self._state == PlaybackState.PLAYING:
            self._start_timer()

        logger.debug(f"Playback speed: {self._playback_speed}x")

    def step_forward(self, frames: int = 1) -> None:
        """
        Step forward by N frames.

        Args:
            frames: Number of frames to advance
        """
        if not self.is_loaded:
            return

        self.pause()
        new_frame = min(
            self._current_frame + frames, self._video_info.frame_count - 1
        )
        self.seek_to_frame(new_frame)

    def step_backward(self, frames: int = 1) -> None:
        """
        Step backward by N frames.

        Args:
            frames: Number of frames to go back
        """
        if not self.is_loaded:
            return

        self.pause()
        new_frame = max(self._current_frame - frames, 0)
        self.seek_to_frame(new_frame)

    def seek_to_frame(self, frame_number: int) -> None:
        """
        Seek to a specific frame.

        Args:
            frame_number: Frame number (0-indexed)
        """
        if not self.is_loaded:
            return

        frame_number = max(0, min(frame_number, self._video_info.frame_count - 1))

        self._capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        self._current_frame = frame_number
        self._read_and_emit_frame()
        self.position_changed.emit(frame_number)

    def seek_to_time(self, seconds: float) -> None:
        """
        Seek to a specific time.

        Args:
            seconds: Time in seconds
        """
        if not self.is_loaded or self._video_info.fps <= 0:
            return

        frame_number = int(seconds * self._video_info.fps)
        self.seek_to_frame(frame_number)

    def seek_to_percent(self, percent: float) -> None:
        """
        Seek to a percentage of the video.

        Args:
            percent: Position as percentage (0-100)
        """
        if not self.is_loaded:
            return

        frame_number = int((percent / 100) * self._video_info.frame_count)
        self.seek_to_frame(frame_number)

    def get_frame_at(self, frame_number: int) -> Optional[np.ndarray]:
        """
        Get a specific frame without changing playback position.

        Args:
            frame_number: Frame number to retrieve

        Returns:
            Frame as numpy array, or None on error
        """
        if not self.is_loaded:
            return None

        # Save current position
        current_pos = self._capture.get(cv2.CAP_PROP_POS_FRAMES)

        # Seek and read
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self._capture.read()

        # Restore position
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, current_pos)

        if ret:
            # Convert to grayscale if color
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return frame

        return None

    def _set_state(self, state: PlaybackState) -> None:
        """Update state and emit signal."""
        self._state = state
        self.state_changed.emit(state)

    def _start_timer(self) -> None:
        """Start the playback timer with appropriate interval."""
        if not self._video_info or self._video_info.fps <= 0:
            return

        # Calculate interval based on speed
        base_interval = 1000 / self._video_info.fps  # ms per frame
        adjusted_interval = base_interval / self._playback_speed

        # Clamp to reasonable range
        interval = max(1, int(adjusted_interval))

        self._timer.start(interval)

    def _on_timer_tick(self) -> None:
        """Handle timer tick for playback."""
        if not self.is_loaded:
            self._timer.stop()
            return

        # Read next frame
        ret = self._read_and_emit_frame()

        if not ret:
            # End of video
            if self._loop:
                self.seek_to_frame(0)
                self.play()
            else:
                self.pause()
                self.video_ended.emit()

    def _read_and_emit_frame(self) -> bool:
        """Read current frame and emit signal."""
        if not self._capture:
            return False

        ret, frame = self._capture.read()

        if not ret:
            return False

        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        self._current_frame = int(self._capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        self.frame_ready.emit(frame, self._current_frame)
        self.position_changed.emit(self._current_frame)

        return True


def format_time(seconds: float) -> str:
    """Format seconds as MM:SS.mmm"""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:06.3f}"


def format_frame_time(frame: int, fps: float) -> str:
    """Format frame number as time."""
    if fps <= 0:
        return "00:00.000"
    return format_time(frame / fps)
