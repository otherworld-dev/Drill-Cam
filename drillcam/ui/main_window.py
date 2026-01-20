"""Main application window for DrillCam."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QFileDialog,
    QStackedWidget,
    QTabWidget,
)
from PySide6.QtGui import QAction, QKeySequence

from ..config.settings import Settings
from ..config.camera_modes import CAMERA_MODES
from ..core.camera_controller import CameraController, create_camera_controller
from ..core.frame_buffer import FrameRingBuffer, create_buffer_for_duration
from ..core.recording_engine import RecordingEngine, RecordingState, RecordingInfo
from ..workers.capture_worker import CaptureWorker, PreviewWorker
from ..utils.video_io import EncodingProgress
from .preview_widget import PreviewWidget
from .controls_panel import ControlsPanel
from .playback_widget import PlaybackWidget
from .analysis_widget import AnalysisWidget
from .media_browser import MediaBrowserWidget
from .dialogs.progress_dialog import EncodingProgressDialog
from ..core.playback_engine import VideoInfo

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        super().__init__()

        self._settings = settings or Settings()
        self._camera: Optional[CameraController] = None
        self._buffer: Optional[FrameRingBuffer] = None
        self._capture_worker: Optional[CaptureWorker] = None
        self._preview_worker: Optional[PreviewWorker] = None
        self._recording_engine: Optional[RecordingEngine] = None
        self._encoding_dialog: Optional[EncodingProgressDialog] = None

        self._dropped_frames = 0
        self._recording_start_frame = 0

        self._setup_ui()
        self._setup_menu()
        self._connect_signals()

        # Initialize camera
        self._init_camera()

    def _setup_ui(self) -> None:
        """Set up the main window UI."""
        self.setWindowTitle("DrillCam - High-Speed Camera Monitor")
        self.setMinimumSize(800, 600)

        # Central widget with tabs for Live/Playback modes
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # Tab widget for mode switching
        self._tab_widget = QTabWidget()
        main_layout.addWidget(self._tab_widget)

        # Live capture tab
        live_widget = QWidget()
        live_layout = QHBoxLayout(live_widget)
        live_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Preview area (left, larger)
        self._preview = PreviewWidget()
        splitter.addWidget(self._preview)

        # Controls panel (right, fixed width)
        self._controls = ControlsPanel()
        self._controls.setMaximumWidth(280)
        splitter.addWidget(self._controls)

        # Set initial splitter proportions
        splitter.setSizes([700, 280])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        live_layout.addWidget(splitter)
        self._tab_widget.addTab(live_widget, "Live Capture")

        # Playback/Analysis tab
        playback_widget = QWidget()
        playback_layout = QHBoxLayout(playback_widget)
        playback_layout.setContentsMargins(0, 0, 0, 0)

        playback_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Playback area (left, larger)
        self._playback = PlaybackWidget()
        playback_splitter.addWidget(self._playback)

        # Analysis controls (right)
        self._analysis = AnalysisWidget()
        self._analysis.setMaximumWidth(300)
        playback_splitter.addWidget(self._analysis)

        playback_splitter.setSizes([700, 300])
        playback_splitter.setStretchFactor(0, 1)
        playback_splitter.setStretchFactor(1, 0)

        playback_layout.addWidget(playback_splitter)
        self._tab_widget.addTab(playback_widget, "Playback & Analysis")

        # Media browser tab
        self._media_browser = MediaBrowserWidget(self._settings.output_dir)
        self._tab_widget.addTab(self._media_browser, "Media")

        # Connect media browser signals
        self._media_browser.open_video_requested.connect(self._on_open_video_from_browser)
        self._media_browser.open_snapshot_requested.connect(self._on_open_snapshot)

        # Connect tab change
        self._tab_widget.currentChanged.connect(self._on_tab_changed)

        # Connect analysis signals
        self._analysis.enhancement_changed.connect(self._on_enhancement_changed)
        self._analysis.measurement_mode_changed.connect(self._on_measurement_mode_changed)
        self._analysis.calibration_requested.connect(self._on_calibration_requested)

        # Connect playback to measurement system
        self._playback.set_measurement_system(self._analysis.measurement_system)
        self._playback.point_clicked.connect(self._on_playback_point_clicked)

        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready")

    def _setup_menu(self) -> None:
        """Set up the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        open_action = QAction("&Open Video...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._open_video)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        snapshot_action = QAction("&Snapshot", self)
        snapshot_action.triggered.connect(self._on_snapshot)
        file_menu.addAction(snapshot_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        crosshair_action = QAction("Show &Crosshair", self)
        crosshair_action.setCheckable(True)
        crosshair_action.setChecked(True)
        crosshair_action.triggered.connect(self._preview.set_show_crosshair)
        view_menu.addAction(crosshair_action)

        info_action = QAction("Show &Info Overlay", self)
        info_action.setCheckable(True)
        info_action.setChecked(True)
        info_action.triggered.connect(self._preview.set_show_info)
        view_menu.addAction(info_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _connect_signals(self) -> None:
        """Connect UI signals to handlers."""
        self._controls.camera_start_requested.connect(self._start_camera)
        self._controls.camera_stop_requested.connect(self._stop_camera)
        self._controls.mode_changed.connect(self._change_mode)
        self._controls.recording_start_requested.connect(self._start_recording)
        self._controls.recording_stop_requested.connect(self._stop_recording)
        self._controls.snapshot_requested.connect(self._on_snapshot)

        # Connect display option checkboxes
        self._controls.crosshair_checkbox.toggled.connect(
            self._preview.set_show_crosshair
        )
        self._controls.info_checkbox.toggled.connect(self._preview.set_show_info)

    def _init_camera(self) -> None:
        """Initialize camera controller."""
        # Detect if we're on Pi or need mock
        use_mock = sys.platform != "linux"

        self._camera = create_camera_controller(use_mock=use_mock)

        if not self._camera.initialize():
            self._status_bar.showMessage("Camera initialization failed")
            QMessageBox.warning(
                self,
                "Camera Error",
                "Failed to initialize camera. Check that the camera is connected "
                "and libcamera is properly configured.",
            )
            return

        # Configure with default mode
        mode_key = self._controls.get_selected_mode()
        if not self._camera.configure(mode_key):
            self._status_bar.showMessage("Camera configuration failed")
            return

        # Create frame buffer
        mode = CAMERA_MODES[mode_key]
        self._buffer = create_buffer_for_duration(
            self._settings.pre_record_seconds + 5,  # Extra buffer
            mode.fps,
            mode.width,
            mode.height,
        )

        self._controls.update_buffer_status(0, self._buffer.capacity)

        # Initialize recording engine
        self._recording_engine = RecordingEngine(
            buffer=self._buffer,
            raw_storage_path=self._settings.get_recording_path(),
            output_dir=self._settings.output_dir,
        )
        self._recording_engine.encoding_progress.connect(self._on_encoding_progress)
        self._recording_engine.recording_complete.connect(self._on_recording_complete)
        self._recording_engine.recording_error.connect(self._on_recording_error)

        self._status_bar.showMessage(
            f"Camera ready: {mode.name} ({mode.width}x{mode.height} @ {mode.fps}fps)"
        )

    @Slot()
    def _start_camera(self) -> None:
        """Start camera capture."""
        if not self._camera or not self._buffer:
            QMessageBox.warning(self, "Error", "Camera not initialized")
            self._controls.set_camera_running(False)
            return

        try:
            # Create and start capture worker
            self._capture_worker = CaptureWorker(self._camera, self._buffer)
            self._capture_worker.fps_updated.connect(self._on_fps_updated)
            self._capture_worker.frame_captured.connect(self._on_frame_captured)
            self._capture_worker.dropped_frames.connect(self._on_dropped_frames)
            self._capture_worker.capture_error.connect(self._on_capture_error)

            # Create and start preview worker
            self._preview_worker = PreviewWorker(
                self._buffer, target_fps=self._settings.preview_fps
            )
            self._preview_worker.frame_ready.connect(self._preview.update_frame)

            # Start workers
            self._capture_worker.start()
            self._preview_worker.start()

            self._dropped_frames = 0
            self._controls.set_camera_running(True)
            self._status_bar.showMessage("Camera running")
            logger.info("Camera started")

        except Exception as e:
            logger.error(f"Failed to start camera: {e}", exc_info=True)
            # Ensure UI is in correct state on error
            self._controls.set_camera_running(False)
            self._status_bar.showMessage(f"Failed to start camera: {e}")
            QMessageBox.warning(
                self,
                "Camera Start Failed",
                f"Failed to start camera capture:\n{e}\n\n"
                "Try switching to a different mode or restarting."
            )

    @Slot()
    def _stop_camera(self) -> None:
        """Stop camera capture."""
        if self._capture_worker:
            self._capture_worker.stop()
            self._capture_worker.wait(5000)  # Wait up to 5 seconds
            self._capture_worker = None

        if self._preview_worker:
            self._preview_worker.stop()
            self._preview_worker.wait(2000)
            self._preview_worker = None

        self._controls.set_camera_running(False)
        self._preview.clear()
        self._status_bar.showMessage("Camera stopped")
        logger.info("Camera stopped")

    @Slot(str)
    def _change_mode(self, mode_key: str) -> None:
        """Change camera mode."""
        was_running = self._capture_worker is not None and self._capture_worker.is_running

        if was_running:
            self._stop_camera()

        if self._camera and self._camera.configure(mode_key):
            mode = CAMERA_MODES[mode_key]

            # Resize buffer for new mode
            if self._buffer:
                self._buffer.resize(
                    int((self._settings.pre_record_seconds + 5) * mode.fps),
                    mode.width,
                    mode.height,
                )
                self._controls.update_buffer_status(0, self._buffer.capacity)

            self._status_bar.showMessage(
                f"Mode changed: {mode.name} ({mode.width}x{mode.height} @ {mode.fps}fps)"
            )

            if was_running:
                self._start_camera()
        else:
            # Configuration failed - ensure UI is in correct state
            logger.error(f"Failed to configure mode: {mode_key}")
            self._controls.set_camera_running(False)  # Ensure mode combo is enabled
            self._status_bar.showMessage(f"Failed to change mode to {mode_key}")
            QMessageBox.warning(
                self,
                "Mode Change Failed",
                f"Failed to configure camera mode: {mode_key}\n\n"
                "Check logs for details. Try restarting the application."
            )

    @Slot()
    def _start_recording(self) -> None:
        """Start recording."""
        if not self._buffer or not self._recording_engine or not self._camera:
            return

        mode = self._camera.current_mode
        if not mode:
            return

        preroll = self._controls.get_preroll_seconds()
        pre_frames = int(preroll * mode.fps)

        # Start recording via engine
        if self._recording_engine.start_capture(
            fps=mode.fps,
            width=mode.width,
            height=mode.height,
            pre_roll_frames=pre_frames,
        ):
            self._controls.set_recording(True)
            self._status_bar.showMessage("Recording...")
            logger.info(f"Recording started with {preroll}s pre-roll")
        else:
            QMessageBox.warning(self, "Recording Error", "Failed to start recording")

    @Slot()
    def _stop_recording(self) -> None:
        """Stop recording and begin encoding."""
        if not self._recording_engine:
            return

        recording_info = self._recording_engine.stop_capture()
        self._controls.set_recording(False)

        if recording_info:
            frame_count = recording_info.frame_count
            self._status_bar.showMessage(
                f"Recording stopped: {frame_count} frames captured. Encoding..."
            )
            logger.info(f"Recording stopped: {frame_count} frames")

            # Show encoding progress dialog
            self._encoding_dialog = EncodingProgressDialog(frame_count, self)
            self._encoding_dialog.show()
        else:
            self._status_bar.showMessage("Recording stopped (no frames captured)")

    @Slot()
    def _on_snapshot(self) -> None:
        """Capture a single frame snapshot."""
        if not self._buffer:
            return

        result = self._buffer.read_latest()
        if result:
            frame, meta = result

            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"snapshot_{timestamp}.png"
            filepath = self._settings.output_dir / filename

            # Save using OpenCV
            cv2.imwrite(str(filepath), frame)

            self._status_bar.showMessage(f"Snapshot saved: {filename}")
            logger.info(f"Snapshot saved: {filepath}")

    @Slot(EncodingProgress)
    def _on_encoding_progress(self, progress: EncodingProgress) -> None:
        """Handle encoding progress update."""
        if self._encoding_dialog:
            self._encoding_dialog.update_progress(progress)

    @Slot(RecordingInfo)
    def _on_recording_complete(self, info: RecordingInfo) -> None:
        """Handle recording completion."""
        if self._encoding_dialog and info.output_path:
            self._encoding_dialog.set_complete(str(info.output_path))

        self._status_bar.showMessage(
            f"Recording saved: {info.output_path.name if info.output_path else 'unknown'} "
            f"({info.frame_count} frames, {info.duration_seconds:.1f}s)"
        )
        logger.info(f"Recording complete: {info.output_path}")

    @Slot(str)
    def _on_recording_error(self, error_msg: str) -> None:
        """Handle recording error."""
        if self._encoding_dialog:
            self._encoding_dialog.set_error(error_msg)

        self._status_bar.showMessage(f"Recording error: {error_msg}")
        logger.error(f"Recording error: {error_msg}")

    @Slot(float)
    def _on_fps_updated(self, fps: float) -> None:
        """Handle FPS update from capture worker."""
        self._controls.update_fps(fps)
        self._preview.update_fps(fps)

    @Slot(int)
    def _on_frame_captured(self, frame_number: int) -> None:
        """Handle frame captured signal."""
        self._controls.update_frame_count(frame_number)
        if self._buffer:
            self._controls.update_buffer_status(self._buffer.count, self._buffer.capacity)

    @Slot(int)
    def _on_dropped_frames(self, count: int) -> None:
        """Handle dropped frames signal."""
        self._dropped_frames += count
        self._controls.update_dropped_count(self._dropped_frames)

    @Slot(str)
    def _on_capture_error(self, error: str) -> None:
        """Handle capture error."""
        logger.error(f"Capture error received: {error}")
        self._status_bar.showMessage(f"Capture error: {error}")

        # Stop camera and ensure UI state is reset
        self._stop_camera()

        # Make sure mode combo is enabled (redundant but safe)
        self._controls.set_camera_running(False)

        QMessageBox.warning(
            self,
            "Capture Error",
            f"Camera capture failed:\n{error}\n\n"
            "The camera has been stopped. Try switching modes or restarting."
        )

    def _show_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About DrillCam",
            "DrillCam v0.1.0\n\n"
            "High-speed camera application for monitoring drilling conditions.\n\n"
            "Designed for OV9281 global shutter camera on Raspberry Pi.\n\n"
            "Designed by Adam Morgan\nOtherWorld.Dev",
        )

    @Slot()
    def _open_video(self) -> None:
        """Open a video file for playback."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video",
            str(self._settings.output_dir),
            "Video Files (*.mkv *.mp4 *.avi);;All Files (*)",
        )

        if path:
            # Switch to playback tab
            self._tab_widget.setCurrentIndex(1)

            if self._playback.load_video(Path(path)):
                self._status_bar.showMessage(f"Loaded: {Path(path).name}")
            else:
                QMessageBox.warning(self, "Error", f"Failed to open video: {path}")

    @Slot(int)
    def _on_tab_changed(self, index: int) -> None:
        """Handle tab change."""
        if index == 0:
            # Live capture tab
            self._status_bar.showMessage("Live capture mode")
        elif index == 1:
            # Playback tab
            self._status_bar.showMessage("Playback & Analysis mode")
        elif index == 2:
            # Media browser tab
            self._media_browser.refresh()
            self._status_bar.showMessage("Media browser")

    @Slot()
    def _on_enhancement_changed(self) -> None:
        """Handle enhancement settings change - refresh current frame."""
        # The playback widget would need to re-apply enhancement
        # This is a simplified version - full integration would modify
        # the playback widget to apply enhancement before display
        pass

    @Slot(Path)
    def _on_open_video_from_browser(self, video_path: Path) -> None:
        """Handle request to open video from media browser."""
        # Switch to playback tab
        self._tab_widget.setCurrentIndex(1)

        if self._playback.load_video(video_path):
            self._status_bar.showMessage(f"Loaded: {video_path.name}")
        else:
            QMessageBox.warning(self, "Error", f"Failed to open video: {video_path}")

    @Slot(Path)
    def _on_open_snapshot(self, snapshot_path: Path) -> None:
        """Handle request to view snapshot from media browser."""
        # Open snapshot in a simple dialog for now
        from PySide6.QtWidgets import QDialog, QLabel
        from PySide6.QtGui import QPixmap

        dialog = QDialog(self)
        dialog.setWindowTitle(snapshot_path.name)
        dialog.setMinimumSize(640, 480)

        layout = QVBoxLayout(dialog)

        label = QLabel()
        pixmap = QPixmap(str(snapshot_path))
        if not pixmap.isNull():
            scaled = pixmap.scaled(
                800, 600,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            label.setPixmap(scaled)
        else:
            label.setText("Failed to load image")

        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)

        dialog.exec()

    @Slot(str)
    def _on_measurement_mode_changed(self, mode: str) -> None:
        """Handle measurement mode change from analysis widget."""
        self._playback.set_measurement_mode(mode)
        if mode:
            self._status_bar.showMessage(f"Measurement mode: {mode} - click on video to add points")
        else:
            self._status_bar.showMessage("Measurement mode disabled")

    @Slot()
    def _on_calibration_requested(self) -> None:
        """Handle calibration request - enter calibration mode."""
        self._playback.set_measurement_mode("calibrate")
        self._status_bar.showMessage("Calibration: Draw a line of known length on the video")

    @Slot(float, float)
    def _on_playback_point_clicked(self, x: float, y: float) -> None:
        """Handle point click on playback widget."""
        mode = self._playback._measurement_mode

        if mode == "calibrate":
            # Check if we have 2 points for calibration
            if len(self._playback._pending_points) >= 2:
                pixel_dist = self._playback.get_calibration_distance()
                if pixel_dist > 0:
                    # Set calibration using the pixel distance
                    self._analysis.set_calibration(pixel_dist)
                    self._status_bar.showMessage(f"Calibration set from {pixel_dist:.1f} pixel line")
                self._playback.clear_pending_points()
                self._playback.set_measurement_mode("")
        else:
            # Update measurements list in analysis widget
            self._analysis._update_measurements_label()

    def closeEvent(self, event) -> None:
        """Handle window close."""
        # Cancel any ongoing recording
        if self._recording_engine and self._recording_engine.is_recording:
            self._recording_engine.cancel()

        self._stop_camera()

        if self._camera:
            self._camera.close()

        event.accept()
