"""Control panel widget for camera and recording controls."""

import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QPushButton,
    QComboBox,
    QLabel,
    QSpinBox,
    QCheckBox,
)

from ..config.camera_modes import CAMERA_MODES, DEFAULT_MODE

logger = logging.getLogger(__name__)


class ControlsPanel(QWidget):
    """
    Control panel for camera settings and recording.

    Signals:
        camera_start_requested: User clicked start camera
        camera_stop_requested: User clicked stop camera
        mode_changed(str): User selected different camera mode
        recording_start_requested: User clicked record
        recording_stop_requested: User clicked stop recording
        snapshot_requested: User clicked snapshot
    """

    camera_start_requested = Signal()
    camera_stop_requested = Signal()
    mode_changed = Signal(str)
    recording_start_requested = Signal()
    recording_stop_requested = Signal()
    snapshot_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._is_camera_running = False
        self._is_recording = False
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the control panel layout."""
        layout = QVBoxLayout(self)

        # Camera controls group
        camera_group = QGroupBox("Camera")
        camera_layout = QVBoxLayout(camera_group)

        # Mode selector
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self._mode_combo = QComboBox()
        for key, mode in CAMERA_MODES.items():
            self._mode_combo.addItem(
                f"{mode.name} ({mode.width}x{mode.height} @ {mode.fps}fps)",
                key,
            )
        # Set default
        default_idx = self._mode_combo.findData(DEFAULT_MODE)
        if default_idx >= 0:
            self._mode_combo.setCurrentIndex(default_idx)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self._mode_combo)
        camera_layout.addLayout(mode_layout)

        # Start/Stop button
        self._camera_btn = QPushButton("Start Camera")
        self._camera_btn.setCheckable(True)
        self._camera_btn.clicked.connect(self._on_camera_btn_clicked)
        camera_layout.addWidget(self._camera_btn)

        layout.addWidget(camera_group)

        # Recording controls group
        record_group = QGroupBox("Recording")
        record_layout = QVBoxLayout(record_group)

        # Pre-record setting
        preroll_layout = QHBoxLayout()
        preroll_layout.addWidget(QLabel("Pre-record (s):"))
        self._preroll_spin = QSpinBox()
        self._preroll_spin.setRange(0, 30)
        self._preroll_spin.setValue(2)
        preroll_layout.addWidget(self._preroll_spin)
        record_layout.addLayout(preroll_layout)

        # Record button
        self._record_btn = QPushButton("Record")
        self._record_btn.setCheckable(True)
        self._record_btn.setEnabled(False)
        self._record_btn.clicked.connect(self._on_record_btn_clicked)
        self._record_btn.setStyleSheet(
            "QPushButton:checked { background-color: #c0392b; color: white; }"
        )
        record_layout.addWidget(self._record_btn)

        # Snapshot button
        self._snapshot_btn = QPushButton("Snapshot")
        self._snapshot_btn.setEnabled(False)
        self._snapshot_btn.clicked.connect(self.snapshot_requested.emit)
        record_layout.addWidget(self._snapshot_btn)

        layout.addWidget(record_group)

        # Status group
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)

        self._fps_label = QLabel("FPS: --")
        status_layout.addWidget(self._fps_label)

        self._frames_label = QLabel("Frames: 0")
        status_layout.addWidget(self._frames_label)

        self._buffer_label = QLabel("Buffer: 0 / 0")
        status_layout.addWidget(self._buffer_label)

        self._dropped_label = QLabel("Dropped: 0")
        self._dropped_label.setStyleSheet("color: #666;")
        status_layout.addWidget(self._dropped_label)

        layout.addWidget(status_group)

        # Display options group
        display_group = QGroupBox("Display")
        display_layout = QVBoxLayout(display_group)

        self._crosshair_check = QCheckBox("Show crosshair")
        self._crosshair_check.setChecked(True)
        display_layout.addWidget(self._crosshair_check)

        self._info_check = QCheckBox("Show info overlay")
        self._info_check.setChecked(True)
        display_layout.addWidget(self._info_check)

        layout.addWidget(display_group)

        # Stretch to push everything to top
        layout.addStretch()

    def _on_mode_changed(self, index: int) -> None:
        """Handle mode combo box change."""
        mode_key = self._mode_combo.currentData()
        if mode_key:
            self.mode_changed.emit(mode_key)

    def _on_camera_btn_clicked(self, checked: bool) -> None:
        """Handle camera start/stop button."""
        if checked:
            self.camera_start_requested.emit()
        else:
            self.camera_stop_requested.emit()

    def _on_record_btn_clicked(self, checked: bool) -> None:
        """Handle record button."""
        if checked:
            self.recording_start_requested.emit()
        else:
            self.recording_stop_requested.emit()

    def set_camera_running(self, running: bool) -> None:
        """Update UI state for camera running/stopped."""
        self._is_camera_running = running
        self._camera_btn.setChecked(running)
        self._camera_btn.setText("Stop Camera" if running else "Start Camera")
        self._mode_combo.setEnabled(not running)
        self._record_btn.setEnabled(running)
        self._snapshot_btn.setEnabled(running)

        if not running:
            self.set_recording(False)

    def set_recording(self, recording: bool) -> None:
        """Update UI state for recording on/off."""
        self._is_recording = recording
        self._record_btn.setChecked(recording)
        self._record_btn.setText("Stop Recording" if recording else "Record")

    def update_fps(self, fps: float) -> None:
        """Update FPS display."""
        self._fps_label.setText(f"FPS: {fps:.1f}")

    def update_frame_count(self, count: int) -> None:
        """Update frame count display."""
        self._frames_label.setText(f"Frames: {count}")

    def update_buffer_status(self, current: int, capacity: int) -> None:
        """Update buffer status display."""
        self._buffer_label.setText(f"Buffer: {current} / {capacity}")

    def update_dropped_count(self, count: int) -> None:
        """Update dropped frame count."""
        self._dropped_label.setText(f"Dropped: {count}")
        if count > 0:
            self._dropped_label.setStyleSheet("color: #e74c3c;")

    def get_preroll_seconds(self) -> int:
        """Get the pre-record duration setting."""
        return self._preroll_spin.value()

    def get_selected_mode(self) -> str:
        """Get the currently selected camera mode key."""
        return self._mode_combo.currentData()

    @property
    def crosshair_checkbox(self) -> QCheckBox:
        """Access to crosshair checkbox for connecting signals."""
        return self._crosshair_check

    @property
    def info_checkbox(self) -> QCheckBox:
        """Access to info checkbox for connecting signals."""
        return self._info_check
