"""Motion tracking and vibration analysis for drilling monitoring."""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum, auto

import numpy as np
import cv2

logger = logging.getLogger(__name__)


class TrackingMethod(Enum):
    """Available tracking methods."""

    PHASE_CORRELATION = auto()  # Good for small periodic motions
    OPTICAL_FLOW = auto()  # Good for tracking specific features
    TEMPLATE_MATCHING = auto()  # Good for tracking a known pattern


@dataclass
class TrackingPoint:
    """A point being tracked."""

    x: float
    y: float
    frame_number: int
    timestamp: float


@dataclass
class MotionData:
    """Motion data for a single frame."""

    frame_number: int
    timestamp: float
    displacement_x: float  # pixels
    displacement_y: float  # pixels
    velocity_x: float  # pixels/frame
    velocity_y: float  # pixels/frame

    @property
    def displacement_magnitude(self) -> float:
        """Total displacement magnitude."""
        return np.sqrt(self.displacement_x**2 + self.displacement_y**2)

    @property
    def velocity_magnitude(self) -> float:
        """Total velocity magnitude."""
        return np.sqrt(self.velocity_x**2 + self.velocity_y**2)


@dataclass
class VibrationAnalysis:
    """Results of vibration frequency analysis."""

    dominant_frequency_hz: float
    frequencies: np.ndarray
    amplitudes: np.ndarray
    peak_frequencies: List[float]
    peak_amplitudes: List[float]


@dataclass
class ROI:
    """Region of interest for tracking."""

    x: int
    y: int
    width: int
    height: int

    def to_slice(self) -> Tuple[slice, slice]:
        """Convert to numpy array slices."""
        return (
            slice(self.y, self.y + self.height),
            slice(self.x, self.x + self.width),
        )

    def to_rect(self) -> Tuple[int, int, int, int]:
        """Convert to (x, y, w, h) tuple."""
        return (self.x, self.y, self.width, self.height)

    def center(self) -> Tuple[float, float]:
        """Get center point."""
        return (self.x + self.width / 2, self.y + self.height / 2)


class MotionTracker:
    """
    Tracks motion in video frames for vibration analysis.

    Supports multiple tracking methods:
    - Phase correlation: Best for small, periodic motions (vibration)
    - Optical flow: Best for tracking specific features
    - Template matching: Best for tracking a known pattern
    """

    def __init__(
        self,
        method: TrackingMethod = TrackingMethod.PHASE_CORRELATION,
        roi: Optional[ROI] = None,
    ) -> None:
        """
        Initialize motion tracker.

        Args:
            method: Tracking method to use
            roi: Region of interest to track (None = full frame)
        """
        self._method = method
        self._roi = roi

        # Reference frame for tracking
        self._reference_frame: Optional[np.ndarray] = None
        self._previous_frame: Optional[np.ndarray] = None

        # Tracking history
        self._motion_history: List[MotionData] = []

        # For optical flow
        self._feature_points: Optional[np.ndarray] = None
        self._lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        # For template matching
        self._template: Optional[np.ndarray] = None

    @property
    def method(self) -> TrackingMethod:
        """Current tracking method."""
        return self._method

    @method.setter
    def method(self, value: TrackingMethod) -> None:
        """Set tracking method and reset state."""
        self._method = value
        self.reset()

    @property
    def roi(self) -> Optional[ROI]:
        """Current region of interest."""
        return self._roi

    @roi.setter
    def roi(self, value: Optional[ROI]) -> None:
        """Set ROI and reset state."""
        self._roi = value
        self.reset()

    @property
    def motion_history(self) -> List[MotionData]:
        """Get motion tracking history."""
        return self._motion_history.copy()

    def reset(self) -> None:
        """Reset tracker state."""
        self._reference_frame = None
        self._previous_frame = None
        self._motion_history.clear()
        self._feature_points = None
        self._template = None

    def set_reference_frame(self, frame: np.ndarray) -> None:
        """
        Set the reference frame for displacement calculation.

        Args:
            frame: Grayscale reference frame
        """
        if self._roi:
            self._reference_frame = frame[self._roi.to_slice()].copy()
        else:
            self._reference_frame = frame.copy()

        self._previous_frame = self._reference_frame.copy()

        # Initialize feature points for optical flow
        if self._method == TrackingMethod.OPTICAL_FLOW:
            self._init_feature_points(self._reference_frame)

        # Set template for template matching
        if self._method == TrackingMethod.TEMPLATE_MATCHING:
            self._template = self._reference_frame.copy()

    def _init_feature_points(self, frame: np.ndarray) -> None:
        """Initialize feature points for optical flow tracking."""
        # Find good features to track
        self._feature_points = cv2.goodFeaturesToTrack(
            frame,
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7,
        )

    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp: float,
    ) -> Optional[MotionData]:
        """
        Process a frame and calculate motion.

        Args:
            frame: Grayscale frame
            frame_number: Frame number
            timestamp: Frame timestamp

        Returns:
            MotionData if tracking successful, None otherwise
        """
        # Extract ROI if set
        if self._roi:
            current = frame[self._roi.to_slice()]
        else:
            current = frame

        # Set reference if not set
        if self._reference_frame is None:
            self.set_reference_frame(frame)
            return None

        # Calculate motion based on method
        if self._method == TrackingMethod.PHASE_CORRELATION:
            dx, dy = self._phase_correlation(current)
        elif self._method == TrackingMethod.OPTICAL_FLOW:
            dx, dy = self._optical_flow(current)
        elif self._method == TrackingMethod.TEMPLATE_MATCHING:
            dx, dy = self._template_matching(frame)  # Use full frame for search
        else:
            return None

        # Calculate velocity (change from previous frame)
        if self._motion_history:
            prev = self._motion_history[-1]
            vx = dx - prev.displacement_x
            vy = dy - prev.displacement_y
        else:
            vx, vy = 0.0, 0.0

        motion = MotionData(
            frame_number=frame_number,
            timestamp=timestamp,
            displacement_x=dx,
            displacement_y=dy,
            velocity_x=vx,
            velocity_y=vy,
        )

        self._motion_history.append(motion)
        self._previous_frame = current.copy()

        return motion

    def _phase_correlation(self, current: np.ndarray) -> Tuple[float, float]:
        """
        Calculate displacement using phase correlation.

        This method is excellent for detecting small, sub-pixel motions
        like vibration.
        """
        if self._reference_frame is None:
            return 0.0, 0.0

        # Ensure same size
        ref = self._reference_frame
        if current.shape != ref.shape:
            current = cv2.resize(current, (ref.shape[1], ref.shape[0]))

        # Convert to float
        ref_float = ref.astype(np.float32)
        cur_float = current.astype(np.float32)

        # Apply window function to reduce edge effects
        rows, cols = ref.shape
        hann_row = np.hanning(rows)
        hann_col = np.hanning(cols)
        hann_2d = np.outer(hann_row, hann_col).astype(np.float32)

        ref_windowed = ref_float * hann_2d
        cur_windowed = cur_float * hann_2d

        # Phase correlation
        shift, response = cv2.phaseCorrelate(ref_windowed, cur_windowed)

        return shift[0], shift[1]

    def _optical_flow(self, current: np.ndarray) -> Tuple[float, float]:
        """
        Calculate displacement using Lucas-Kanade optical flow.
        """
        if self._previous_frame is None or self._feature_points is None:
            return 0.0, 0.0

        if len(self._feature_points) == 0:
            # Re-initialize if we lost all points
            self._init_feature_points(self._previous_frame)
            if self._feature_points is None or len(self._feature_points) == 0:
                return 0.0, 0.0

        # Calculate optical flow
        new_points, status, err = cv2.calcOpticalFlowPyrLK(
            self._previous_frame,
            current,
            self._feature_points,
            None,
            **self._lk_params,
        )

        if new_points is None:
            return 0.0, 0.0

        # Filter good points
        good_old = self._feature_points[status == 1]
        good_new = new_points[status == 1]

        if len(good_old) == 0:
            return 0.0, 0.0

        # Calculate average displacement
        displacements = good_new - good_old
        dx = np.mean(displacements[:, 0])
        dy = np.mean(displacements[:, 1])

        # Update feature points
        self._feature_points = good_new.reshape(-1, 1, 2)

        # Re-initialize if too few points
        if len(self._feature_points) < 10:
            self._init_feature_points(current)

        return float(dx), float(dy)

    def _template_matching(self, frame: np.ndarray) -> Tuple[float, float]:
        """
        Calculate displacement using template matching.
        """
        if self._template is None or self._roi is None:
            return 0.0, 0.0

        # Define search region (larger than template)
        margin = 50  # pixels to search around ROI
        search_x = max(0, self._roi.x - margin)
        search_y = max(0, self._roi.y - margin)
        search_w = min(frame.shape[1] - search_x, self._roi.width + 2 * margin)
        search_h = min(frame.shape[0] - search_y, self._roi.height + 2 * margin)

        search_region = frame[search_y:search_y + search_h, search_x:search_x + search_w]

        # Template matching
        result = cv2.matchTemplate(
            search_region,
            self._template,
            cv2.TM_CCOEFF_NORMED,
        )

        # Find best match
        _, _, _, max_loc = cv2.minMaxLoc(result)

        # Calculate displacement from original position
        original_x = self._roi.x - search_x
        original_y = self._roi.y - search_y

        dx = max_loc[0] - original_x
        dy = max_loc[1] - original_y

        return float(dx), float(dy)

    def analyze_vibration(self, fps: float) -> Optional[VibrationAnalysis]:
        """
        Perform frequency analysis on motion history.

        Args:
            fps: Video frame rate for frequency calculation

        Returns:
            VibrationAnalysis with frequency data
        """
        if len(self._motion_history) < 10:
            logger.warning("Not enough motion data for vibration analysis")
            return None

        # Extract displacement magnitude over time
        displacements = np.array([
            m.displacement_magnitude for m in self._motion_history
        ])

        # Remove DC component (mean)
        displacements = displacements - np.mean(displacements)

        # Apply window function
        window = np.hanning(len(displacements))
        windowed = displacements * window

        # FFT
        n = len(windowed)
        fft_result = np.fft.rfft(windowed)
        amplitudes = np.abs(fft_result) * 2 / n

        # Frequency axis
        frequencies = np.fft.rfftfreq(n, d=1/fps)

        # Find peaks
        peak_indices = self._find_peaks(amplitudes)
        peak_frequencies = frequencies[peak_indices].tolist()
        peak_amplitudes = amplitudes[peak_indices].tolist()

        # Dominant frequency (highest amplitude peak)
        if len(peak_indices) > 0:
            dominant_idx = peak_indices[np.argmax(amplitudes[peak_indices])]
            dominant_freq = frequencies[dominant_idx]
        else:
            dominant_freq = 0.0

        return VibrationAnalysis(
            dominant_frequency_hz=float(dominant_freq),
            frequencies=frequencies,
            amplitudes=amplitudes,
            peak_frequencies=peak_frequencies,
            peak_amplitudes=peak_amplitudes,
        )

    def _find_peaks(
        self,
        data: np.ndarray,
        threshold: float = 0.1,
    ) -> np.ndarray:
        """Find peaks in data above threshold."""
        # Simple peak detection
        peaks = []
        max_val = np.max(data)

        for i in range(1, len(data) - 1):
            if data[i] > data[i-1] and data[i] > data[i+1]:
                if data[i] > threshold * max_val:
                    peaks.append(i)

        return np.array(peaks)

    def get_motion_trail(
        self,
        frame_shape: Tuple[int, int],
        scale: float = 10.0,
    ) -> np.ndarray:
        """
        Generate a visualization of the motion trail.

        Args:
            frame_shape: (height, width) of output image
            scale: Scale factor for displacements

        Returns:
            BGR image with motion trail
        """
        height, width = frame_shape
        trail = np.zeros((height, width, 3), dtype=np.uint8)

        if len(self._motion_history) < 2:
            return trail

        # Get center point
        if self._roi:
            cx, cy = self._roi.center()
        else:
            cx, cy = width / 2, height / 2

        # Draw trail
        points = []
        for motion in self._motion_history:
            x = int(cx + motion.displacement_x * scale)
            y = int(cy + motion.displacement_y * scale)
            points.append((x, y))

        # Draw lines connecting points
        for i in range(1, len(points)):
            # Color gradient from blue (old) to red (new)
            t = i / len(points)
            color = (
                int(255 * (1 - t)),  # B
                0,  # G
                int(255 * t),  # R
            )
            cv2.line(trail, points[i-1], points[i], color, 2)

        # Draw current position
        if points:
            cv2.circle(trail, points[-1], 5, (0, 255, 0), -1)

        return trail

    def export_data(self) -> dict:
        """Export motion data as dictionary for saving."""
        return {
            "method": self._method.name,
            "roi": self._roi.to_rect() if self._roi else None,
            "motion_history": [
                {
                    "frame_number": m.frame_number,
                    "timestamp": m.timestamp,
                    "displacement_x": m.displacement_x,
                    "displacement_y": m.displacement_y,
                    "velocity_x": m.velocity_x,
                    "velocity_y": m.velocity_y,
                }
                for m in self._motion_history
            ],
        }

    def export_csv(self, filepath: str) -> None:
        """Export motion data to CSV file."""
        import csv

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "frame_number",
                "timestamp",
                "displacement_x",
                "displacement_y",
                "displacement_magnitude",
                "velocity_x",
                "velocity_y",
                "velocity_magnitude",
            ])

            for m in self._motion_history:
                writer.writerow([
                    m.frame_number,
                    m.timestamp,
                    m.displacement_x,
                    m.displacement_y,
                    m.displacement_magnitude,
                    m.velocity_x,
                    m.velocity_y,
                    m.velocity_magnitude,
                ])

        logger.info(f"Motion data exported to {filepath}")
