"""Image enhancement tools for video analysis."""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cv2

logger = logging.getLogger(__name__)


@dataclass
class EnhancementSettings:
    """Settings for image enhancement."""

    # Brightness and contrast
    brightness: int = 0  # -100 to 100
    contrast: float = 1.0  # 0.5 to 2.0

    # Histogram equalization
    histogram_equalization: bool = False
    clahe_enabled: bool = False
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8

    # Sharpening
    sharpen_enabled: bool = False
    sharpen_amount: float = 1.0  # 0.5 to 3.0

    # Noise reduction
    denoise_enabled: bool = False
    denoise_strength: int = 10  # 1 to 30

    # Edge enhancement
    edge_enhance_enabled: bool = False

    # Zoom
    zoom_factor: float = 1.0  # 1.0 to 10.0
    zoom_center_x: float = 0.5  # 0.0 to 1.0 (relative)
    zoom_center_y: float = 0.5  # 0.0 to 1.0 (relative)

    # Gamma correction
    gamma: float = 1.0  # 0.5 to 2.0

    def reset(self) -> None:
        """Reset all settings to defaults."""
        self.brightness = 0
        self.contrast = 1.0
        self.histogram_equalization = False
        self.clahe_enabled = False
        self.clahe_clip_limit = 2.0
        self.clahe_grid_size = 8
        self.sharpen_enabled = False
        self.sharpen_amount = 1.0
        self.denoise_enabled = False
        self.denoise_strength = 10
        self.edge_enhance_enabled = False
        self.zoom_factor = 1.0
        self.zoom_center_x = 0.5
        self.zoom_center_y = 0.5
        self.gamma = 1.0


class ImageEnhancer:
    """
    Applies various image enhancements to video frames.

    Designed for improving visibility of drilling features.
    """

    def __init__(self) -> None:
        self._settings = EnhancementSettings()
        self._clahe: Optional[cv2.CLAHE] = None
        self._gamma_lut: Optional[np.ndarray] = None
        self._last_gamma: float = 1.0

    @property
    def settings(self) -> EnhancementSettings:
        """Get current enhancement settings."""
        return self._settings

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply all enabled enhancements to a frame.

        Args:
            frame: Input grayscale frame

        Returns:
            Enhanced frame
        """
        result = frame.copy()

        # Apply enhancements in order
        if self._settings.gamma != 1.0:
            result = self._apply_gamma(result)

        if self._settings.brightness != 0 or self._settings.contrast != 1.0:
            result = self._apply_brightness_contrast(result)

        if self._settings.histogram_equalization:
            result = cv2.equalizeHist(result)
        elif self._settings.clahe_enabled:
            result = self._apply_clahe(result)

        if self._settings.denoise_enabled:
            result = self._apply_denoise(result)

        if self._settings.sharpen_enabled:
            result = self._apply_sharpen(result)

        if self._settings.edge_enhance_enabled:
            result = self._apply_edge_enhance(result)

        if self._settings.zoom_factor > 1.0:
            result = self._apply_zoom(result)

        return result

    def _apply_brightness_contrast(self, frame: np.ndarray) -> np.ndarray:
        """Apply brightness and contrast adjustment."""
        alpha = self._settings.contrast
        beta = self._settings.brightness

        # Use cv2.convertScaleAbs for efficient computation
        return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    def _apply_gamma(self, frame: np.ndarray) -> np.ndarray:
        """Apply gamma correction."""
        gamma = self._settings.gamma

        # Build lookup table if gamma changed
        if self._gamma_lut is None or gamma != self._last_gamma:
            inv_gamma = 1.0 / gamma
            self._gamma_lut = np.array([
                ((i / 255.0) ** inv_gamma) * 255
                for i in range(256)
            ]).astype(np.uint8)
            self._last_gamma = gamma

        return cv2.LUT(frame, self._gamma_lut)

    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization."""
        # Create CLAHE object if needed or settings changed
        if (self._clahe is None or
            self._clahe.getClipLimit() != self._settings.clahe_clip_limit):
            self._clahe = cv2.createCLAHE(
                clipLimit=self._settings.clahe_clip_limit,
                tileGridSize=(
                    self._settings.clahe_grid_size,
                    self._settings.clahe_grid_size,
                ),
            )

        return self._clahe.apply(frame)

    def _apply_sharpen(self, frame: np.ndarray) -> np.ndarray:
        """Apply sharpening filter."""
        amount = self._settings.sharpen_amount

        # Unsharp masking
        blurred = cv2.GaussianBlur(frame, (0, 0), 3)
        sharpened = cv2.addWeighted(
            frame, 1.0 + amount,
            blurred, -amount,
            0
        )

        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def _apply_denoise(self, frame: np.ndarray) -> np.ndarray:
        """Apply noise reduction."""
        strength = self._settings.denoise_strength

        # FastNlMeansDenoising for grayscale
        return cv2.fastNlMeansDenoising(
            frame,
            None,
            h=strength,
            templateWindowSize=7,
            searchWindowSize=21,
        )

    def _apply_edge_enhance(self, frame: np.ndarray) -> np.ndarray:
        """Apply edge enhancement."""
        # Sobel edge detection
        sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=3)

        # Magnitude
        edges = np.sqrt(sobelx**2 + sobely**2)
        edges = np.clip(edges, 0, 255).astype(np.uint8)

        # Blend with original
        return cv2.addWeighted(frame, 0.7, edges, 0.3, 0)

    def _apply_zoom(self, frame: np.ndarray) -> np.ndarray:
        """Apply digital zoom."""
        zoom = self._settings.zoom_factor
        if zoom <= 1.0:
            return frame

        h, w = frame.shape[:2]

        # Calculate crop region
        crop_w = int(w / zoom)
        crop_h = int(h / zoom)

        # Center point
        cx = int(w * self._settings.zoom_center_x)
        cy = int(h * self._settings.zoom_center_y)

        # Crop bounds
        x1 = max(0, cx - crop_w // 2)
        y1 = max(0, cy - crop_h // 2)
        x2 = min(w, x1 + crop_w)
        y2 = min(h, y1 + crop_h)

        # Adjust if hitting edges
        if x2 - x1 < crop_w:
            x1 = max(0, x2 - crop_w)
        if y2 - y1 < crop_h:
            y1 = max(0, y2 - crop_h)

        # Crop and resize back to original dimensions
        cropped = frame[y1:y2, x1:x2]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    def set_zoom_center(self, x: float, y: float) -> None:
        """
        Set zoom center point.

        Args:
            x: Relative x position (0.0 to 1.0)
            y: Relative y position (0.0 to 1.0)
        """
        self._settings.zoom_center_x = max(0.0, min(1.0, x))
        self._settings.zoom_center_y = max(0.0, min(1.0, y))

    def reset(self) -> None:
        """Reset all enhancement settings."""
        self._settings.reset()
        self._clahe = None
        self._gamma_lut = None


def compute_histogram(frame: np.ndarray) -> np.ndarray:
    """
    Compute histogram of grayscale frame.

    Args:
        frame: Grayscale frame

    Returns:
        Histogram array (256 bins)
    """
    return cv2.calcHist([frame], [0], None, [256], [0, 256]).flatten()


def draw_histogram(
    histogram: np.ndarray,
    width: int = 256,
    height: int = 100,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """
    Draw histogram as an image.

    Args:
        histogram: Histogram data
        width: Output image width
        height: Output image height
        color: Line color (BGR)

    Returns:
        BGR image with histogram
    """
    # Normalize histogram
    hist_normalized = histogram / histogram.max() * height

    # Create image
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Draw histogram bars
    bin_width = width // 256
    for i in range(256):
        h = int(hist_normalized[i])
        x = i * bin_width
        cv2.line(img, (x, height), (x, height - h), color, bin_width)

    return img


def auto_brightness_contrast(
    frame: np.ndarray,
    clip_percent: float = 1.0,
) -> Tuple[np.ndarray, float, float]:
    """
    Automatically adjust brightness and contrast.

    Uses histogram clipping to find optimal values.

    Args:
        frame: Input grayscale frame
        clip_percent: Percentage of pixels to clip at each end

    Returns:
        Tuple of (adjusted_frame, alpha, beta)
    """
    # Calculate histogram
    hist = cv2.calcHist([frame], [0], None, [256], [0, 256]).flatten()
    hist_size = hist.sum()

    # Find clip points
    accumulator = 0
    min_gray = 0
    for i in range(256):
        accumulator += hist[i]
        if accumulator > hist_size * clip_percent / 100:
            min_gray = i
            break

    accumulator = 0
    max_gray = 255
    for i in range(255, -1, -1):
        accumulator += hist[i]
        if accumulator > hist_size * clip_percent / 100:
            max_gray = i
            break

    # Calculate alpha (contrast) and beta (brightness)
    if max_gray - min_gray == 0:
        alpha = 1.0
        beta = 0
    else:
        alpha = 255.0 / (max_gray - min_gray)
        beta = -min_gray * alpha

    # Apply adjustment
    adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    return adjusted, alpha, beta
