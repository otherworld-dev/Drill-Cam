"""Measurement and calibration tools for dimensional analysis."""

import logging
import json
import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from pathlib import Path
from enum import Enum, auto

import numpy as np
import cv2

logger = logging.getLogger(__name__)


class MeasurementType(Enum):
    """Types of measurements."""

    DISTANCE = auto()
    ANGLE = auto()
    RECTANGLE = auto()
    CIRCLE = auto()


@dataclass
class Point:
    """A 2D point."""

    x: float
    y: float

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def to_int_tuple(self) -> Tuple[int, int]:
        return (int(self.x), int(self.y))

    def distance_to(self, other: "Point") -> float:
        """Calculate distance to another point."""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


@dataclass
class Measurement:
    """A measurement on the image."""

    id: int
    type: MeasurementType
    points: List[Point]
    label: str = ""
    color: Tuple[int, int, int] = (0, 255, 0)

    def get_value_pixels(self) -> float:
        """Get measurement value in pixels."""
        if self.type == MeasurementType.DISTANCE:
            if len(self.points) >= 2:
                return self.points[0].distance_to(self.points[1])
        elif self.type == MeasurementType.ANGLE:
            if len(self.points) >= 3:
                return self._calculate_angle()
        elif self.type == MeasurementType.RECTANGLE:
            if len(self.points) >= 2:
                w = abs(self.points[1].x - self.points[0].x)
                h = abs(self.points[1].y - self.points[0].y)
                return w * h  # Area
        elif self.type == MeasurementType.CIRCLE:
            if len(self.points) >= 2:
                radius = self.points[0].distance_to(self.points[1])
                return math.pi * radius ** 2  # Area
        return 0.0

    def _calculate_angle(self) -> float:
        """Calculate angle between three points (angle at middle point)."""
        if len(self.points) < 3:
            return 0.0

        p1, p2, p3 = self.points[0], self.points[1], self.points[2]

        # Vectors from p2 to p1 and p2 to p3
        v1 = (p1.x - p2.x, p1.y - p2.y)
        v2 = (p3.x - p2.x, p3.y - p2.y)

        # Dot product and magnitudes
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

        if mag1 == 0 or mag2 == 0:
            return 0.0

        # Angle in degrees
        cos_angle = max(-1, min(1, dot / (mag1 * mag2)))
        return math.degrees(math.acos(cos_angle))


@dataclass
class Calibration:
    """Calibration data for pixel-to-real-world conversion."""

    pixels_per_unit: float
    unit: str = "mm"
    reference_distance_pixels: float = 0.0
    reference_distance_real: float = 0.0

    def pixels_to_real(self, pixels: float) -> float:
        """Convert pixels to real-world units."""
        if self.pixels_per_unit <= 0:
            return pixels
        return pixels / self.pixels_per_unit

    def real_to_pixels(self, real: float) -> float:
        """Convert real-world units to pixels."""
        return real * self.pixels_per_unit

    def to_dict(self) -> dict:
        """Convert to dictionary for saving."""
        return {
            "pixels_per_unit": self.pixels_per_unit,
            "unit": self.unit,
            "reference_distance_pixels": self.reference_distance_pixels,
            "reference_distance_real": self.reference_distance_real,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Calibration":
        """Create from dictionary."""
        return cls(
            pixels_per_unit=data.get("pixels_per_unit", 1.0),
            unit=data.get("unit", "mm"),
            reference_distance_pixels=data.get("reference_distance_pixels", 0.0),
            reference_distance_real=data.get("reference_distance_real", 0.0),
        )


class MeasurementSystem:
    """
    System for calibration and measurements on video frames.

    Features:
    - Pixel-to-real-world calibration
    - Distance measurement
    - Angle measurement
    - Area measurement (rectangle, circle)
    - Persistent calibration storage
    """

    def __init__(self) -> None:
        self._calibration: Optional[Calibration] = None
        self._measurements: List[Measurement] = []
        self._next_id = 1

    @property
    def calibration(self) -> Optional[Calibration]:
        """Current calibration."""
        return self._calibration

    @property
    def is_calibrated(self) -> bool:
        """Check if system is calibrated."""
        return self._calibration is not None and self._calibration.pixels_per_unit > 0

    @property
    def measurements(self) -> List[Measurement]:
        """List of current measurements."""
        return self._measurements.copy()

    def calibrate_from_reference(
        self,
        pixel_distance: float,
        real_distance: float,
        unit: str = "mm",
    ) -> Calibration:
        """
        Calibrate using a known reference distance.

        Args:
            pixel_distance: Distance in pixels
            real_distance: Corresponding real-world distance
            unit: Unit of measurement (mm, cm, inch, etc.)

        Returns:
            New calibration
        """
        if real_distance <= 0:
            raise ValueError("Real distance must be positive")

        pixels_per_unit = pixel_distance / real_distance

        self._calibration = Calibration(
            pixels_per_unit=pixels_per_unit,
            unit=unit,
            reference_distance_pixels=pixel_distance,
            reference_distance_real=real_distance,
        )

        logger.info(
            f"Calibration set: {pixels_per_unit:.2f} pixels/{unit} "
            f"(reference: {real_distance} {unit} = {pixel_distance:.1f} px)"
        )

        return self._calibration

    def calibrate_from_points(
        self,
        p1: Point,
        p2: Point,
        real_distance: float,
        unit: str = "mm",
    ) -> Calibration:
        """
        Calibrate using two points with known real-world distance.

        Args:
            p1: First point
            p2: Second point
            real_distance: Known distance between points
            unit: Unit of measurement

        Returns:
            New calibration
        """
        pixel_distance = p1.distance_to(p2)
        return self.calibrate_from_reference(pixel_distance, real_distance, unit)

    def add_distance_measurement(
        self,
        p1: Point,
        p2: Point,
        label: str = "",
    ) -> Measurement:
        """Add a distance measurement."""
        measurement = Measurement(
            id=self._next_id,
            type=MeasurementType.DISTANCE,
            points=[p1, p2],
            label=label,
        )
        self._measurements.append(measurement)
        self._next_id += 1
        return measurement

    def add_angle_measurement(
        self,
        p1: Point,
        vertex: Point,
        p2: Point,
        label: str = "",
    ) -> Measurement:
        """Add an angle measurement (angle at vertex)."""
        measurement = Measurement(
            id=self._next_id,
            type=MeasurementType.ANGLE,
            points=[p1, vertex, p2],
            label=label,
            color=(255, 165, 0),  # Orange
        )
        self._measurements.append(measurement)
        self._next_id += 1
        return measurement

    def add_rectangle_measurement(
        self,
        corner1: Point,
        corner2: Point,
        label: str = "",
    ) -> Measurement:
        """Add a rectangle area measurement."""
        measurement = Measurement(
            id=self._next_id,
            type=MeasurementType.RECTANGLE,
            points=[corner1, corner2],
            label=label,
            color=(255, 0, 255),  # Magenta
        )
        self._measurements.append(measurement)
        self._next_id += 1
        return measurement

    def add_circle_measurement(
        self,
        center: Point,
        edge: Point,
        label: str = "",
    ) -> Measurement:
        """Add a circle area measurement."""
        measurement = Measurement(
            id=self._next_id,
            type=MeasurementType.CIRCLE,
            points=[center, edge],
            label=label,
            color=(0, 255, 255),  # Cyan
        )
        self._measurements.append(measurement)
        self._next_id += 1
        return measurement

    def remove_measurement(self, measurement_id: int) -> bool:
        """Remove a measurement by ID."""
        for i, m in enumerate(self._measurements):
            if m.id == measurement_id:
                self._measurements.pop(i)
                return True
        return False

    def clear_measurements(self) -> None:
        """Clear all measurements."""
        self._measurements.clear()

    def get_measurement_value(
        self,
        measurement: Measurement,
        use_calibration: bool = True,
    ) -> Tuple[float, str]:
        """
        Get measurement value with unit.

        Args:
            measurement: The measurement
            use_calibration: Whether to convert to real-world units

        Returns:
            Tuple of (value, unit_string)
        """
        pixels = measurement.get_value_pixels()

        if measurement.type == MeasurementType.ANGLE:
            return pixels, "°"

        if use_calibration and self.is_calibrated:
            cal = self._calibration
            if measurement.type == MeasurementType.DISTANCE:
                return cal.pixels_to_real(pixels), cal.unit
            else:
                # Area: convert using squared ratio
                area_real = pixels / (cal.pixels_per_unit ** 2)
                return area_real, f"{cal.unit}²"

        # Return pixels if not calibrated
        if measurement.type == MeasurementType.DISTANCE:
            return pixels, "px"
        else:
            return pixels, "px²"

    def draw_measurements(
        self,
        frame: np.ndarray,
        show_values: bool = True,
    ) -> np.ndarray:
        """
        Draw measurements on a frame.

        Args:
            frame: Input frame (grayscale or BGR)
            show_values: Whether to show measurement values

        Returns:
            Frame with measurements drawn
        """
        # Convert grayscale to BGR if needed
        if len(frame.shape) == 2:
            output = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            output = frame.copy()

        for m in self._measurements:
            self._draw_measurement(output, m, show_values)

        return output

    def _draw_measurement(
        self,
        frame: np.ndarray,
        measurement: Measurement,
        show_values: bool,
    ) -> None:
        """Draw a single measurement."""
        color = measurement.color
        points = measurement.points

        if measurement.type == MeasurementType.DISTANCE:
            if len(points) >= 2:
                p1, p2 = points[0].to_int_tuple(), points[1].to_int_tuple()
                cv2.line(frame, p1, p2, color, 2)
                cv2.circle(frame, p1, 4, color, -1)
                cv2.circle(frame, p2, 4, color, -1)

                if show_values:
                    value, unit = self.get_measurement_value(measurement)
                    text = f"{value:.2f} {unit}"
                    mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2 - 10)
                    cv2.putText(
                        frame, text, mid,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                    )

        elif measurement.type == MeasurementType.ANGLE:
            if len(points) >= 3:
                p1 = points[0].to_int_tuple()
                vertex = points[1].to_int_tuple()
                p2 = points[2].to_int_tuple()

                cv2.line(frame, vertex, p1, color, 2)
                cv2.line(frame, vertex, p2, color, 2)
                cv2.circle(frame, vertex, 4, color, -1)

                if show_values:
                    value, unit = self.get_measurement_value(measurement)
                    text = f"{value:.1f}{unit}"
                    cv2.putText(
                        frame, text,
                        (vertex[0] + 10, vertex[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                    )

        elif measurement.type == MeasurementType.RECTANGLE:
            if len(points) >= 2:
                p1, p2 = points[0].to_int_tuple(), points[1].to_int_tuple()
                cv2.rectangle(frame, p1, p2, color, 2)

                if show_values:
                    value, unit = self.get_measurement_value(measurement)
                    text = f"{value:.2f} {unit}"
                    cv2.putText(
                        frame, text,
                        (min(p1[0], p2[0]), min(p1[1], p2[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                    )

        elif measurement.type == MeasurementType.CIRCLE:
            if len(points) >= 2:
                center = points[0].to_int_tuple()
                radius = int(points[0].distance_to(points[1]))
                cv2.circle(frame, center, radius, color, 2)
                cv2.circle(frame, center, 3, color, -1)

                if show_values:
                    value, unit = self.get_measurement_value(measurement)
                    text = f"{value:.2f} {unit}"
                    cv2.putText(
                        frame, text,
                        (center[0] + 10, center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                    )

    def save_calibration(self, filepath: Path) -> None:
        """Save calibration to file."""
        if not self._calibration:
            logger.warning("No calibration to save")
            return

        with open(filepath, "w") as f:
            json.dump(self._calibration.to_dict(), f, indent=2)

        logger.info(f"Calibration saved to {filepath}")

    def load_calibration(self, filepath: Path) -> bool:
        """Load calibration from file."""
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            self._calibration = Calibration.from_dict(data)
            logger.info(f"Calibration loaded from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            return False

    def export_measurements(self, filepath: Path) -> None:
        """Export measurements to JSON file."""
        data = {
            "calibration": self._calibration.to_dict() if self._calibration else None,
            "measurements": [
                {
                    "id": m.id,
                    "type": m.type.name,
                    "points": [(p.x, p.y) for p in m.points],
                    "label": m.label,
                    "value_pixels": m.get_value_pixels(),
                    "value_real": self.get_measurement_value(m)[0] if self.is_calibrated else None,
                    "unit": self.get_measurement_value(m)[1],
                }
                for m in self._measurements
            ],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Measurements exported to {filepath}")
