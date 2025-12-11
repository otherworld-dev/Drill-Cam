"""Progress dialog for encoding and other long operations."""

import logging
from typing import Optional

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QFrame,
)

from ...utils.video_io import EncodingProgress

logger = logging.getLogger(__name__)


class EncodingProgressDialog(QDialog):
    """Dialog showing encoding progress with cancel option."""

    def __init__(self, total_frames: int, parent=None) -> None:
        super().__init__(parent)

        self._total_frames = total_frames
        self._cancelled = False

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the dialog UI."""
        self.setWindowTitle("Encoding Video")
        self.setMinimumWidth(400)
        self.setModal(True)

        # Remove close button from title bar
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowType.WindowCloseButtonHint
        )

        layout = QVBoxLayout(self)

        # Status label
        self._status_label = QLabel("Encoding frames to FFV1...")
        layout.addWidget(self._status_label)

        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        layout.addWidget(self._progress_bar)

        # Stats frame
        stats_frame = QFrame()
        stats_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        stats_layout = QVBoxLayout(stats_frame)

        # Frames progress
        self._frames_label = QLabel(f"Frames: 0 / {self._total_frames}")
        stats_layout.addWidget(self._frames_label)

        # Encoding speed
        self._speed_label = QLabel("Speed: -- fps")
        stats_layout.addWidget(self._speed_label)

        # ETA
        self._eta_label = QLabel("ETA: --")
        stats_layout.addWidget(self._eta_label)

        layout.addWidget(stats_frame)

        # Cancel button
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._on_cancel)
        button_layout.addWidget(self._cancel_btn)

        layout.addLayout(button_layout)

    @property
    def was_cancelled(self) -> bool:
        """Check if encoding was cancelled."""
        return self._cancelled

    @Slot(EncodingProgress)
    def update_progress(self, progress: EncodingProgress) -> None:
        """Update progress display."""
        self._progress_bar.setValue(int(progress.percent))
        self._frames_label.setText(
            f"Frames: {progress.frames_encoded} / {progress.total_frames}"
        )
        self._speed_label.setText(f"Speed: {progress.fps:.1f} fps")

        # Format ETA
        eta_seconds = int(progress.eta_seconds)
        if eta_seconds < 60:
            eta_str = f"{eta_seconds}s"
        elif eta_seconds < 3600:
            minutes = eta_seconds // 60
            seconds = eta_seconds % 60
            eta_str = f"{minutes}m {seconds}s"
        else:
            hours = eta_seconds // 3600
            minutes = (eta_seconds % 3600) // 60
            eta_str = f"{hours}h {minutes}m"

        self._eta_label.setText(f"ETA: {eta_str}")

    def set_complete(self, output_path: str) -> None:
        """Mark encoding as complete."""
        self._status_label.setText("Encoding complete!")
        self._progress_bar.setValue(100)
        self._frames_label.setText(f"Frames: {self._total_frames} / {self._total_frames}")
        self._speed_label.setText("")
        self._eta_label.setText(f"Saved to: {output_path}")

        self._cancel_btn.setText("Close")
        self._cancel_btn.clicked.disconnect()
        self._cancel_btn.clicked.connect(self.accept)

    def set_error(self, error_msg: str) -> None:
        """Show encoding error."""
        self._status_label.setText(f"Encoding failed: {error_msg}")
        self._status_label.setStyleSheet("color: #e74c3c;")

        self._cancel_btn.setText("Close")
        self._cancel_btn.clicked.disconnect()
        self._cancel_btn.clicked.connect(self.reject)

    def _on_cancel(self) -> None:
        """Handle cancel button."""
        self._cancelled = True
        self._status_label.setText("Cancelling...")
        self._cancel_btn.setEnabled(False)


class CaptureProgressDialog(QDialog):
    """Simple dialog showing capture is in progress."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the dialog UI."""
        self.setWindowTitle("Recording")
        self.setMinimumWidth(300)
        self.setModal(False)  # Non-modal so user can still see preview

        layout = QVBoxLayout(self)

        # Recording indicator
        self._status_label = QLabel("Recording in progress...")
        self._status_label.setStyleSheet(
            "color: #c0392b; font-weight: bold; font-size: 14px;"
        )
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status_label)

        # Frame counter
        self._frames_label = QLabel("Frames: 0")
        self._frames_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._frames_label)

        # Duration
        self._duration_label = QLabel("Duration: 0.0s")
        self._duration_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._duration_label)

        # Stop button
        self._stop_btn = QPushButton("Stop Recording")
        self._stop_btn.setStyleSheet(
            "QPushButton { background-color: #c0392b; color: white; "
            "padding: 8px; font-weight: bold; }"
        )
        layout.addWidget(self._stop_btn)

    @property
    def stop_button(self) -> QPushButton:
        """Access to stop button for connecting signals."""
        return self._stop_btn

    def update_stats(self, frame_count: int, duration_seconds: float) -> None:
        """Update recording statistics."""
        self._frames_label.setText(f"Frames: {frame_count}")
        self._duration_label.setText(f"Duration: {duration_seconds:.1f}s")
