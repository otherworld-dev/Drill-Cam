"""Media browser widget for viewing saved snapshots and videos."""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal, Slot, QTimer, QSize
from PySide6.QtGui import QPixmap, QIcon
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QLabel,
    QPushButton,
    QComboBox,
    QSplitter,
    QFrame,
    QMessageBox,
    QSizePolicy,
)

from ..utils.thumbnails import (
    generate_video_thumbnail,
    generate_image_thumbnail,
    get_video_info,
    remove_from_cache,
)

logger = logging.getLogger(__name__)


@dataclass
class MediaItem:
    """Information about a media file."""

    path: Path
    is_video: bool
    size_bytes: int
    modified_time: datetime
    video_info: Optional[dict] = None  # For videos: fps, duration, frame_count


class MediaBrowserWidget(QWidget):
    """Widget for browsing saved snapshots and videos."""

    # Signals
    open_video_requested = Signal(Path)  # Request to open video in playback tab
    open_snapshot_requested = Signal(Path)  # Request to view snapshot

    def __init__(self, output_dir: Path, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._output_dir = output_dir
        self._media_items: list[MediaItem] = []
        self._current_filter = "all"  # all, videos, snapshots

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Toolbar
        toolbar = QHBoxLayout()

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.setFixedWidth(80)
        self._refresh_btn.setToolTip("Rescan output folder for new files")
        toolbar.addWidget(self._refresh_btn)

        self._open_btn = QPushButton("Open")
        self._open_btn.setFixedWidth(80)
        self._open_btn.setEnabled(False)
        self._open_btn.setToolTip(
            "Open selected file.\n"
            "Videos open in Playback tab.\n"
            "Images open in preview window."
        )
        toolbar.addWidget(self._open_btn)

        self._delete_btn = QPushButton("Delete")
        self._delete_btn.setFixedWidth(80)
        self._delete_btn.setEnabled(False)
        self._delete_btn.setToolTip("Permanently delete selected file")
        toolbar.addWidget(self._delete_btn)

        toolbar.addStretch()

        filter_label = QLabel("Filter:")
        toolbar.addWidget(filter_label)

        self._filter_combo = QComboBox()
        self._filter_combo.addItem("All", "all")
        self._filter_combo.addItem("Videos", "videos")
        self._filter_combo.addItem("Snapshots", "snapshots")
        self._filter_combo.setFixedWidth(100)
        self._filter_combo.setToolTip("Filter by file type")
        toolbar.addWidget(self._filter_combo)

        layout.addLayout(toolbar)

        # Main content: list + preview
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # File list (left side)
        self._file_list = QListWidget()
        self._file_list.setMinimumWidth(200)
        self._file_list.setIconSize(QSize(64, 48))
        self._file_list.setToolTip("Double-click to open, sorted by date (newest first)")
        splitter.addWidget(self._file_list)

        # Preview panel (right side)
        preview_panel = QFrame()
        preview_panel.setFrameStyle(QFrame.Shape.StyledPanel)
        preview_layout = QVBoxLayout(preview_panel)

        # Preview image
        self._preview_label = QLabel()
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setMinimumSize(320, 240)
        self._preview_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._preview_label.setStyleSheet("background-color: #1a1a1a; color: #888;")
        self._preview_label.setText("Select a file to preview")
        preview_layout.addWidget(self._preview_label, stretch=1)

        # Info panel
        info_frame = QFrame()
        info_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        info_layout = QVBoxLayout(info_frame)
        info_layout.setContentsMargins(8, 8, 8, 8)

        self._info_filename = QLabel("Filename: -")
        self._info_type = QLabel("Type: -")
        self._info_size = QLabel("Size: -")
        self._info_date = QLabel("Date: -")
        self._info_duration = QLabel("Duration: -")

        for label in [
            self._info_filename,
            self._info_type,
            self._info_size,
            self._info_date,
            self._info_duration,
        ]:
            label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            info_layout.addWidget(label)

        preview_layout.addWidget(info_frame)

        splitter.addWidget(preview_panel)

        # Set splitter proportions
        splitter.setSizes([300, 500])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter, stretch=1)

    def _connect_signals(self) -> None:
        """Connect widget signals."""
        self._refresh_btn.clicked.connect(self.refresh)
        self._open_btn.clicked.connect(self._on_open_clicked)
        self._delete_btn.clicked.connect(self._on_delete_clicked)
        self._filter_combo.currentIndexChanged.connect(self._on_filter_changed)
        self._file_list.currentItemChanged.connect(self._on_selection_changed)
        self._file_list.itemDoubleClicked.connect(self._on_item_double_clicked)

    def set_output_dir(self, output_dir: Path) -> None:
        """Update the output directory and refresh."""
        self._output_dir = output_dir
        self.refresh()

    @Slot()
    def refresh(self) -> None:
        """Refresh the media list."""
        self._media_items.clear()
        self._file_list.clear()

        if not self._output_dir.exists():
            logger.warning(f"Output directory does not exist: {self._output_dir}")
            return

        # Scan for videos and snapshots
        video_extensions = {".mkv", ".mp4", ".avi"}
        image_extensions = {".png", ".jpg", ".jpeg"}

        for file_path in self._output_dir.iterdir():
            if not file_path.is_file():
                continue

            suffix = file_path.suffix.lower()
            is_video = suffix in video_extensions
            is_image = suffix in image_extensions

            if not is_video and not is_image:
                continue

            try:
                stat = file_path.stat()
                modified_time = datetime.fromtimestamp(stat.st_mtime)

                video_info = None
                if is_video:
                    video_info = get_video_info(file_path)

                item = MediaItem(
                    path=file_path,
                    is_video=is_video,
                    size_bytes=stat.st_size,
                    modified_time=modified_time,
                    video_info=video_info,
                )
                self._media_items.append(item)

            except Exception as e:
                logger.warning(f"Failed to read file info: {file_path}: {e}")

        # Sort by modified time (newest first)
        self._media_items.sort(key=lambda x: x.modified_time, reverse=True)

        # Apply filter and populate list
        self._populate_list()

    def _populate_list(self) -> None:
        """Populate the file list based on current filter."""
        self._file_list.clear()

        for media_item in self._media_items:
            # Apply filter
            if self._current_filter == "videos" and not media_item.is_video:
                continue
            if self._current_filter == "snapshots" and media_item.is_video:
                continue

            # Create list item
            list_item = QListWidgetItem()
            list_item.setText(media_item.path.name)
            list_item.setData(Qt.ItemDataRole.UserRole, media_item)

            # Generate thumbnail
            if media_item.is_video:
                thumbnail = generate_video_thumbnail(media_item.path, size=(64, 48))
            else:
                thumbnail = generate_image_thumbnail(media_item.path, size=(64, 48))

            if thumbnail:
                list_item.setIcon(QIcon(thumbnail))

            self._file_list.addItem(list_item)

    @Slot(int)
    def _on_filter_changed(self, index: int) -> None:
        """Handle filter combo change."""
        self._current_filter = self._filter_combo.currentData()
        self._populate_list()

    @Slot(QListWidgetItem, QListWidgetItem)
    def _on_selection_changed(
        self, current: Optional[QListWidgetItem], previous: Optional[QListWidgetItem]
    ) -> None:
        """Handle file selection change."""
        has_selection = current is not None
        self._open_btn.setEnabled(has_selection)
        self._delete_btn.setEnabled(has_selection)

        if not current:
            self._clear_preview()
            return

        media_item: MediaItem = current.data(Qt.ItemDataRole.UserRole)
        self._update_preview(media_item)

    def _update_preview(self, item: MediaItem) -> None:
        """Update the preview panel for selected item."""
        # Generate larger preview
        if item.is_video:
            preview = generate_video_thumbnail(item.path, size=(320, 240))
        else:
            preview = generate_image_thumbnail(item.path, size=(320, 240))

        if preview:
            # Scale to fit preview label while maintaining aspect ratio
            scaled = preview.scaled(
                self._preview_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._preview_label.setPixmap(scaled)
        else:
            self._preview_label.setText("Preview not available")

        # Update info labels
        self._info_filename.setText(f"Filename: {item.path.name}")
        self._info_type.setText(f"Type: {'Video (MKV)' if item.is_video else 'Image (PNG)'}")
        self._info_size.setText(f"Size: {self._format_size(item.size_bytes)}")
        self._info_date.setText(
            f"Date: {item.modified_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        if item.is_video and item.video_info:
            duration = item.video_info.get("duration", 0)
            fps = item.video_info.get("fps", 0)
            frame_count = item.video_info.get("frame_count", 0)
            self._info_duration.setText(
                f"Duration: {duration:.1f}s @ {fps:.0f}fps ({frame_count} frames)"
            )
            self._info_duration.setVisible(True)
        else:
            self._info_duration.setVisible(False)

    def _clear_preview(self) -> None:
        """Clear the preview panel."""
        self._preview_label.clear()
        self._preview_label.setText("Select a file to preview")
        self._info_filename.setText("Filename: -")
        self._info_type.setText("Type: -")
        self._info_size.setText("Size: -")
        self._info_date.setText("Date: -")
        self._info_duration.setVisible(False)

    def _format_size(self, size_bytes: int) -> str:
        """Format file size for display."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"

    @Slot()
    def _on_open_clicked(self) -> None:
        """Handle open button click."""
        current = self._file_list.currentItem()
        if not current:
            return

        media_item: MediaItem = current.data(Qt.ItemDataRole.UserRole)
        self._open_media(media_item)

    @Slot(QListWidgetItem)
    def _on_item_double_clicked(self, item: QListWidgetItem) -> None:
        """Handle double click on item."""
        media_item: MediaItem = item.data(Qt.ItemDataRole.UserRole)
        self._open_media(media_item)

    def _open_media(self, item: MediaItem) -> None:
        """Open a media item."""
        if item.is_video:
            self.open_video_requested.emit(item.path)
        else:
            self.open_snapshot_requested.emit(item.path)

    @Slot()
    def _on_delete_clicked(self) -> None:
        """Handle delete button click."""
        current = self._file_list.currentItem()
        if not current:
            return

        media_item: MediaItem = current.data(Qt.ItemDataRole.UserRole)

        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete:\n{media_item.path.name}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                media_item.path.unlink()
                remove_from_cache(media_item.path)
                self.refresh()
                logger.info(f"Deleted: {media_item.path}")
            except Exception as e:
                QMessageBox.warning(
                    self, "Delete Error", f"Failed to delete file:\n{e}"
                )
                logger.error(f"Failed to delete {media_item.path}: {e}")
