"""Playback widget with slow-motion controls."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QImage, QPixmap, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QComboBox,
    QFrame,
    QSizePolicy,
    QFileDialog,
    QStyle,
)

from ..core.playback_engine import (
    PlaybackEngine,
    PlaybackState,
    VideoInfo,
    format_time,
    format_frame_time,
)

logger = logging.getLogger(__name__)


class PlaybackWidget(QWidget):
    """
    Widget for video playback with slow-motion support.

    Features:
    - Play/pause/stop controls
    - Variable speed playback (0.1x to 2x)
    - Frame-by-frame stepping
    - Timeline scrubber
    - Frame/time display
    """

    # Signals
    video_loaded = Signal(VideoInfo)
    video_closed = Signal()
    frame_exported = Signal(str)  # filepath

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._engine = PlaybackEngine(self)
        self._current_frame: Optional[np.ndarray] = None

        self._setup_ui()
        self._connect_signals()
        self._setup_shortcuts()
        self._update_ui_state()

    def _setup_ui(self) -> None:
        """Set up the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Video display area
        self._display_label = QLabel()
        self._display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._display_label.setMinimumSize(320, 200)
        self._display_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._display_label.setStyleSheet("background-color: #1a1a1a;")
        self._display_label.setText("No video loaded")
        layout.addWidget(self._display_label)

        # Controls container
        controls_frame = QFrame()
        controls_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        controls_layout = QVBoxLayout(controls_frame)

        # Timeline slider
        timeline_layout = QHBoxLayout()

        self._time_label = QLabel("00:00.000")
        self._time_label.setMinimumWidth(80)
        timeline_layout.addWidget(self._time_label)

        self._timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self._timeline_slider.setRange(0, 100)
        self._timeline_slider.setValue(0)
        timeline_layout.addWidget(self._timeline_slider)

        self._duration_label = QLabel("00:00.000")
        self._duration_label.setMinimumWidth(80)
        timeline_layout.addWidget(self._duration_label)

        controls_layout.addLayout(timeline_layout)

        # Playback controls
        playback_layout = QHBoxLayout()

        # Previous frame
        self._prev_btn = QPushButton()
        self._prev_btn.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSkipBackward)
        )
        self._prev_btn.setToolTip("Previous frame (Left arrow)")
        playback_layout.addWidget(self._prev_btn)

        # Step back 10
        self._back10_btn = QPushButton("-10")
        self._back10_btn.setToolTip("Back 10 frames")
        self._back10_btn.setMaximumWidth(40)
        playback_layout.addWidget(self._back10_btn)

        # Play/Pause
        self._play_btn = QPushButton()
        self._play_btn.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        )
        self._play_btn.setToolTip("Play/Pause (Space)")
        playback_layout.addWidget(self._play_btn)

        # Stop
        self._stop_btn = QPushButton()
        self._stop_btn.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop)
        )
        self._stop_btn.setToolTip("Stop")
        playback_layout.addWidget(self._stop_btn)

        # Step forward 10
        self._fwd10_btn = QPushButton("+10")
        self._fwd10_btn.setToolTip("Forward 10 frames")
        self._fwd10_btn.setMaximumWidth(40)
        playback_layout.addWidget(self._fwd10_btn)

        # Next frame
        self._next_btn = QPushButton()
        self._next_btn.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSkipForward)
        )
        self._next_btn.setToolTip("Next frame (Right arrow)")
        playback_layout.addWidget(self._next_btn)

        playback_layout.addStretch()

        # Speed control
        playback_layout.addWidget(QLabel("Speed:"))
        self._speed_combo = QComboBox()
        self._speed_combo.addItems(["0.1x", "0.25x", "0.5x", "1x", "2x"])
        self._speed_combo.setCurrentIndex(2)  # Default 0.5x for slow-mo
        self._speed_combo.setToolTip("Playback speed")
        playback_layout.addWidget(self._speed_combo)

        # Loop toggle
        self._loop_btn = QPushButton("Loop")
        self._loop_btn.setCheckable(True)
        self._loop_btn.setToolTip("Loop playback")
        playback_layout.addWidget(self._loop_btn)

        controls_layout.addLayout(playback_layout)

        # Frame info and export
        info_layout = QHBoxLayout()

        self._frame_label = QLabel("Frame: 0 / 0")
        info_layout.addWidget(self._frame_label)

        info_layout.addStretch()

        self._export_btn = QPushButton("Export Frame")
        self._export_btn.setToolTip("Save current frame as PNG")
        info_layout.addWidget(self._export_btn)

        controls_layout.addLayout(info_layout)

        layout.addWidget(controls_frame)

    def _connect_signals(self) -> None:
        """Connect signals."""
        # Engine signals
        self._engine.frame_ready.connect(self._on_frame_ready)
        self._engine.position_changed.connect(self._on_position_changed)
        self._engine.state_changed.connect(self._on_state_changed)
        self._engine.video_loaded.connect(self._on_video_loaded)
        self._engine.video_ended.connect(self._on_video_ended)
        self._engine.error.connect(self._on_error)

        # Control buttons
        self._play_btn.clicked.connect(self._engine.toggle_play_pause)
        self._stop_btn.clicked.connect(self._engine.stop)
        self._prev_btn.clicked.connect(lambda: self._engine.step_backward(1))
        self._next_btn.clicked.connect(lambda: self._engine.step_forward(1))
        self._back10_btn.clicked.connect(lambda: self._engine.step_backward(10))
        self._fwd10_btn.clicked.connect(lambda: self._engine.step_forward(10))

        # Speed control
        self._speed_combo.currentTextChanged.connect(self._on_speed_changed)

        # Loop toggle
        self._loop_btn.toggled.connect(self._on_loop_toggled)

        # Timeline slider
        self._timeline_slider.sliderPressed.connect(self._on_slider_pressed)
        self._timeline_slider.sliderReleased.connect(self._on_slider_released)
        self._timeline_slider.valueChanged.connect(self._on_slider_value_changed)

        # Export
        self._export_btn.clicked.connect(self._on_export_frame)

    def _setup_shortcuts(self) -> None:
        """Set up keyboard shortcuts."""
        # Space for play/pause
        space_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        space_shortcut.activated.connect(self._engine.toggle_play_pause)

        # Left/Right for frame stepping
        left_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Left), self)
        left_shortcut.activated.connect(lambda: self._engine.step_backward(1))

        right_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Right), self)
        right_shortcut.activated.connect(lambda: self._engine.step_forward(1))

        # Home/End for start/end
        home_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Home), self)
        home_shortcut.activated.connect(lambda: self._engine.seek_to_frame(0))

        end_shortcut = QShortcut(QKeySequence(Qt.Key.Key_End), self)
        end_shortcut.activated.connect(
            lambda: self._engine.seek_to_frame(
                self._engine.video_info.frame_count - 1
                if self._engine.video_info
                else 0
            )
        )

    def _update_ui_state(self) -> None:
        """Update UI based on current state."""
        has_video = self._engine.is_loaded
        is_playing = self._engine.state == PlaybackState.PLAYING

        # Enable/disable controls
        self._play_btn.setEnabled(has_video)
        self._stop_btn.setEnabled(has_video)
        self._prev_btn.setEnabled(has_video)
        self._next_btn.setEnabled(has_video)
        self._back10_btn.setEnabled(has_video)
        self._fwd10_btn.setEnabled(has_video)
        self._timeline_slider.setEnabled(has_video)
        self._speed_combo.setEnabled(has_video)
        self._loop_btn.setEnabled(has_video)
        self._export_btn.setEnabled(has_video)

        # Update play button icon
        if is_playing:
            self._play_btn.setIcon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause)
            )
        else:
            self._play_btn.setIcon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
            )

    def load_video(self, path: Path) -> bool:
        """Load a video file."""
        return self._engine.load_video(path)

    def close_video(self) -> None:
        """Close the current video."""
        self._engine.unload()
        self._display_label.clear()
        self._display_label.setText("No video loaded")
        self._time_label.setText("00:00.000")
        self._duration_label.setText("00:00.000")
        self._frame_label.setText("Frame: 0 / 0")
        self._timeline_slider.setValue(0)
        self._update_ui_state()
        self.video_closed.emit()

    @Slot(np.ndarray, int)
    def _on_frame_ready(self, frame: np.ndarray, frame_number: int) -> None:
        """Handle new frame from engine."""
        self._current_frame = frame
        self._update_display(frame)

    def _update_display(self, frame: np.ndarray) -> None:
        """Update the display with a frame."""
        height, width = frame.shape[:2]

        # Convert to QImage
        if len(frame.shape) == 2:
            # Grayscale
            qimage = QImage(
                frame.data, width, height, width, QImage.Format.Format_Grayscale8
            )
        else:
            # Color (BGR to RGB)
            import cv2

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qimage = QImage(
                rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888
            )

        # Scale to fit display
        pixmap = QPixmap.fromImage(qimage).scaled(
            self._display_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        self._display_label.setPixmap(pixmap)

    @Slot(int)
    def _on_position_changed(self, frame_number: int) -> None:
        """Handle position change."""
        if not self._engine.video_info:
            return

        info = self._engine.video_info

        # Update time label
        time_str = format_frame_time(frame_number, info.fps)
        self._time_label.setText(time_str)

        # Update frame label
        self._frame_label.setText(f"Frame: {frame_number} / {info.frame_count - 1}")

        # Update slider (without triggering value changed)
        if not self._timeline_slider.isSliderDown():
            percent = (frame_number / max(1, info.frame_count - 1)) * 100
            self._timeline_slider.blockSignals(True)
            self._timeline_slider.setValue(int(percent))
            self._timeline_slider.blockSignals(False)

    @Slot(PlaybackState)
    def _on_state_changed(self, state: PlaybackState) -> None:
        """Handle state change."""
        self._update_ui_state()

    @Slot(VideoInfo)
    def _on_video_loaded(self, info: VideoInfo) -> None:
        """Handle video loaded."""
        self._duration_label.setText(format_time(info.duration_seconds))
        self._frame_label.setText(f"Frame: 0 / {info.frame_count - 1}")
        self._timeline_slider.setRange(0, 100)
        self._update_ui_state()
        self.video_loaded.emit(info)

    @Slot()
    def _on_video_ended(self) -> None:
        """Handle video end."""
        logger.debug("Video playback ended")

    @Slot(str)
    def _on_error(self, error_msg: str) -> None:
        """Handle playback error."""
        logger.error(f"Playback error: {error_msg}")

    def _on_speed_changed(self, text: str) -> None:
        """Handle speed combo change."""
        speed = float(text.replace("x", ""))
        self._engine.set_speed(speed)

    def _on_loop_toggled(self, checked: bool) -> None:
        """Handle loop toggle."""
        self._engine.loop = checked

    def _on_slider_pressed(self) -> None:
        """Handle slider press - pause playback."""
        if self._engine.state == PlaybackState.PLAYING:
            self._was_playing = True
            self._engine.pause()
        else:
            self._was_playing = False

    def _on_slider_released(self) -> None:
        """Handle slider release - resume if was playing."""
        if hasattr(self, "_was_playing") and self._was_playing:
            self._engine.play()

    def _on_slider_value_changed(self, value: int) -> None:
        """Handle slider value change."""
        if self._timeline_slider.isSliderDown():
            self._engine.seek_to_percent(value)

    def _on_export_frame(self) -> None:
        """Export current frame as PNG."""
        if self._current_frame is None:
            return

        # Get save path
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Frame",
            f"frame_{self._engine.current_frame:06d}.png",
            "PNG Images (*.png)",
        )

        if path:
            import cv2

            cv2.imwrite(path, self._current_frame)
            self.frame_exported.emit(path)
            logger.info(f"Frame exported: {path}")

    def resizeEvent(self, event) -> None:
        """Handle resize."""
        super().resizeEvent(event)
        if self._current_frame is not None:
            self._update_display(self._current_frame)
