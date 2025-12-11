"""Thumbnail generation utilities for media browser."""

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PySide6.QtGui import QPixmap, QImage

logger = logging.getLogger(__name__)

# In-memory thumbnail cache
_thumbnail_cache: dict[str, QPixmap] = {}


def _numpy_to_qpixmap(array: np.ndarray) -> QPixmap:
    """Convert numpy array to QPixmap."""
    if len(array.shape) == 2:
        # Grayscale
        height, width = array.shape
        bytes_per_line = width
        image = QImage(
            array.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8
        )
    else:
        # Color (BGR from OpenCV)
        height, width, channels = array.shape
        if channels == 3:
            # Convert BGR to RGB
            array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
            bytes_per_line = 3 * width
            image = QImage(
                array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888
            )
        elif channels == 4:
            bytes_per_line = 4 * width
            image = QImage(
                array.data, width, height, bytes_per_line, QImage.Format.Format_RGBA8888
            )
        else:
            return QPixmap()

    return QPixmap.fromImage(image)


def generate_video_thumbnail(
    video_path: Path, size: Tuple[int, int] = (160, 120), use_cache: bool = True
) -> Optional[QPixmap]:
    """
    Generate a thumbnail from the first frame of a video.

    Args:
        video_path: Path to video file
        size: Thumbnail size (width, height)
        use_cache: Whether to use cached thumbnails

    Returns:
        QPixmap thumbnail or None on error
    """
    cache_key = f"video:{video_path}:{size[0]}x{size[1]}"

    if use_cache and cache_key in _thumbnail_cache:
        return _thumbnail_cache[cache_key]

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning(f"Could not open video: {video_path}")
            return None

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            logger.warning(f"Could not read frame from: {video_path}")
            return None

        # Resize to thumbnail size maintaining aspect ratio
        h, w = frame.shape[:2]
        target_w, target_h = size

        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Create padded thumbnail with black background
        thumbnail = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        thumbnail[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

        pixmap = _numpy_to_qpixmap(thumbnail)

        if use_cache:
            _thumbnail_cache[cache_key] = pixmap

        return pixmap

    except Exception as e:
        logger.error(f"Failed to generate video thumbnail: {e}")
        return None


def generate_image_thumbnail(
    image_path: Path, size: Tuple[int, int] = (160, 120), use_cache: bool = True
) -> Optional[QPixmap]:
    """
    Generate a thumbnail from an image file.

    Args:
        image_path: Path to image file
        size: Thumbnail size (width, height)
        use_cache: Whether to use cached thumbnails

    Returns:
        QPixmap thumbnail or None on error
    """
    cache_key = f"image:{image_path}:{size[0]}x{size[1]}"

    if use_cache and cache_key in _thumbnail_cache:
        return _thumbnail_cache[cache_key]

    try:
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            logger.warning(f"Could not read image: {image_path}")
            return None

        # Convert grayscale to BGR for consistent handling
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Resize to thumbnail size maintaining aspect ratio
        h, w = image.shape[:2]
        target_w, target_h = size

        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Create padded thumbnail
        thumbnail = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        thumbnail[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

        pixmap = _numpy_to_qpixmap(thumbnail)

        if use_cache:
            _thumbnail_cache[cache_key] = pixmap

        return pixmap

    except Exception as e:
        logger.error(f"Failed to generate image thumbnail: {e}")
        return None


def get_video_info(video_path: Path) -> Optional[dict]:
    """
    Get video metadata.

    Args:
        video_path: Path to video file

    Returns:
        Dict with width, height, fps, frame_count, duration or None on error
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }

        if info["fps"] > 0 and info["frame_count"] > 0:
            info["duration"] = info["frame_count"] / info["fps"]
        else:
            info["duration"] = 0

        cap.release()
        return info

    except Exception as e:
        logger.error(f"Failed to get video info: {e}")
        return None


def clear_thumbnail_cache() -> None:
    """Clear the thumbnail cache."""
    _thumbnail_cache.clear()


def remove_from_cache(file_path: Path) -> None:
    """Remove a specific file's thumbnails from cache."""
    keys_to_remove = [k for k in _thumbnail_cache if str(file_path) in k]
    for key in keys_to_remove:
        del _thumbnail_cache[key]
