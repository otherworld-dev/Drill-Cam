"""Preview widget for displaying live camera feed."""

import logging
from typing import Optional

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QSizePolicy

logger = logging.getLogger(__name__)


class PreviewWidget(QWidget):
    """
    Widget for displaying live camera preview.

    Features:
    - Maintains aspect ratio while scaling
    - Supports overlay rendering (crosshairs, ROI, measurements)
    - Efficient QImage to QPixmap conversion
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._current_image: Optional[QImage] = None
        self._show_crosshair = True
        self._show_info = True
        self._frame_count = 0
        self._fps = 0.0

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the widget layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Image display label
        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setMinimumSize(320, 200)
        self._image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._image_label.setStyleSheet("background-color: #1a1a1a;")

        layout.addWidget(self._image_label)

        # Set placeholder text
        self._image_label.setText("No camera feed")
        self._image_label.setStyleSheet(
            "background-color: #1a1a1a; color: #666; font-size: 14px;"
        )

    @Slot(object)
    def update_frame(self, qimage: QImage) -> None:
        """
        Update the displayed frame.

        Args:
            qimage: New frame to display
        """
        self._current_image = qimage
        self._frame_count += 1
        self._update_display()

    def update_fps(self, fps: float) -> None:
        """Update the FPS display value."""
        self._fps = fps

    def _update_display(self) -> None:
        """Render current image to display."""
        if self._current_image is None:
            return

        # Get available display size
        label_size = self._image_label.size()

        # Scale image to fit while maintaining aspect ratio
        scaled_pixmap = QPixmap.fromImage(self._current_image).scaled(
            label_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        # Draw overlays
        if self._show_crosshair or self._show_info:
            scaled_pixmap = self._draw_overlays(scaled_pixmap)

        self._image_label.setPixmap(scaled_pixmap)

    def _draw_overlays(self, pixmap: QPixmap) -> QPixmap:
        """Draw overlay elements on the pixmap."""
        result = QPixmap(pixmap)
        painter = QPainter(result)

        try:
            width = result.width()
            height = result.height()

            # Draw crosshair
            if self._show_crosshair:
                pen = QPen(QColor(0, 255, 0, 128))
                pen.setWidth(1)
                painter.setPen(pen)

                # Vertical line
                painter.drawLine(width // 2, 0, width // 2, height)
                # Horizontal line
                painter.drawLine(0, height // 2, width, height // 2)

                # Center circle
                painter.drawEllipse(
                    width // 2 - 20, height // 2 - 20, 40, 40
                )

            # Draw info overlay
            if self._show_info:
                pen = QPen(QColor(255, 255, 255))
                painter.setPen(pen)

                # FPS display
                painter.drawText(10, 20, f"FPS: {self._fps:.1f}")
                painter.drawText(10, 40, f"Frame: {self._frame_count}")

        finally:
            painter.end()

        return result

    def set_show_crosshair(self, show: bool) -> None:
        """Toggle crosshair overlay visibility."""
        self._show_crosshair = show
        self._update_display()

    def set_show_info(self, show: bool) -> None:
        """Toggle info overlay visibility."""
        self._show_info = show
        self._update_display()

    def clear(self) -> None:
        """Clear the display."""
        self._current_image = None
        self._frame_count = 0
        self._fps = 0.0
        self._image_label.clear()
        self._image_label.setText("No camera feed")

    def resizeEvent(self, event) -> None:
        """Handle resize by updating display."""
        super().resizeEvent(event)
        self._update_display()
