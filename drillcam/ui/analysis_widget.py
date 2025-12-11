"""Analysis controls widget for motion tracking, measurement, and enhancement."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QPushButton,
    QComboBox,
    QLabel,
    QSlider,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QLineEdit,
    QTabWidget,
    QFileDialog,
    QMessageBox,
)

from ..processing.motion_tracker import (
    MotionTracker,
    TrackingMethod,
    ROI,
    MotionData,
    VibrationAnalysis,
)
from ..processing.measurement import (
    MeasurementSystem,
    MeasurementType,
    Point,
    Measurement,
)
from ..processing.enhancement import (
    ImageEnhancer,
    EnhancementSettings,
)

logger = logging.getLogger(__name__)


class AnalysisWidget(QWidget):
    """
    Widget for analysis controls including:
    - Motion tracking and vibration analysis
    - Measurement tools
    - Image enhancement
    """

    # Signals
    tracking_started = Signal()
    tracking_stopped = Signal()
    roi_selection_requested = Signal()  # Request user to select ROI
    measurement_mode_changed = Signal(str)  # "distance", "angle", etc. or ""
    calibration_requested = Signal()
    enhancement_changed = Signal()
    export_requested = Signal(str)  # Export type

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._tracker = MotionTracker()
        self._measurement = MeasurementSystem()
        self._enhancer = ImageEnhancer()

        self._is_tracking = False
        self._measurement_mode: str = ""

        self._setup_ui()
        self._connect_signals()

    @property
    def tracker(self) -> MotionTracker:
        """Get the motion tracker."""
        return self._tracker

    @property
    def measurement_system(self) -> MeasurementSystem:
        """Get the measurement system."""
        return self._measurement

    @property
    def enhancer(self) -> ImageEnhancer:
        """Get the image enhancer."""
        return self._enhancer

    @property
    def is_tracking(self) -> bool:
        """Check if tracking is active."""
        return self._is_tracking

    @property
    def measurement_mode(self) -> str:
        """Current measurement mode."""
        return self._measurement_mode

    def _setup_ui(self) -> None:
        """Set up the widget UI."""
        layout = QVBoxLayout(self)

        # Tab widget for different analysis types
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # Motion tracking tab
        tracking_widget = self._create_tracking_tab()
        tabs.addTab(tracking_widget, "Motion")

        # Measurement tab
        measurement_widget = self._create_measurement_tab()
        tabs.addTab(measurement_widget, "Measure")

        # Enhancement tab
        enhancement_widget = self._create_enhancement_tab()
        tabs.addTab(enhancement_widget, "Enhance")

    def _create_tracking_tab(self) -> QWidget:
        """Create motion tracking controls."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Tracking method
        method_group = QGroupBox("Tracking Method")
        method_layout = QVBoxLayout(method_group)

        self._method_combo = QComboBox()
        self._method_combo.addItem("Phase Correlation", TrackingMethod.PHASE_CORRELATION)
        self._method_combo.addItem("Optical Flow", TrackingMethod.OPTICAL_FLOW)
        self._method_combo.addItem("Template Matching", TrackingMethod.TEMPLATE_MATCHING)
        self._method_combo.setToolTip(
            "Phase Correlation: Fast, good for small displacements\n"
            "Optical Flow: Tracks feature points, good for complex motion\n"
            "Template Matching: Tracks specific pattern, most accurate"
        )
        method_layout.addWidget(self._method_combo)

        layout.addWidget(method_group)

        # ROI selection
        roi_group = QGroupBox("Region of Interest")
        roi_layout = QVBoxLayout(roi_group)

        self._roi_btn = QPushButton("Select ROI")
        self._roi_btn.setToolTip(
            "Define a region to track.\n"
            "Smaller regions = faster tracking.\n"
            "Select area with consistent features."
        )
        roi_layout.addWidget(self._roi_btn)

        self._roi_label = QLabel("ROI: Not set")
        self._roi_label.setToolTip("Current tracking region coordinates")
        roi_layout.addWidget(self._roi_label)

        self._clear_roi_btn = QPushButton("Clear ROI")
        self._clear_roi_btn.setToolTip("Remove ROI and track entire frame")
        roi_layout.addWidget(self._clear_roi_btn)

        layout.addWidget(roi_group)

        # Tracking controls
        control_group = QGroupBox("Tracking")
        control_layout = QVBoxLayout(control_group)

        self._start_tracking_btn = QPushButton("Start Tracking")
        self._start_tracking_btn.setCheckable(True)
        self._start_tracking_btn.setToolTip(
            "Begin motion tracking during playback.\n"
            "Measures displacement from reference frame."
        )
        control_layout.addWidget(self._start_tracking_btn)

        self._set_reference_btn = QPushButton("Set Reference Frame")
        self._set_reference_btn.setToolTip(
            "Use current frame as the reference point.\n"
            "All motion is measured relative to this frame."
        )
        control_layout.addWidget(self._set_reference_btn)

        layout.addWidget(control_group)

        # Results
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)

        self._displacement_label = QLabel("Displacement: --")
        self._displacement_label.setToolTip("Distance moved from reference position (pixels)")
        results_layout.addWidget(self._displacement_label)

        self._velocity_label = QLabel("Velocity: --")
        self._velocity_label.setToolTip("Speed of motion (pixels per frame)")
        results_layout.addWidget(self._velocity_label)

        self._vibration_btn = QPushButton("Analyze Vibration")
        self._vibration_btn.setToolTip(
            "Perform FFT analysis on tracked motion.\n"
            "Identifies dominant vibration frequencies.\n"
            "Requires sufficient tracking data first."
        )
        results_layout.addWidget(self._vibration_btn)

        self._vibration_label = QLabel("Dominant freq: --")
        self._vibration_label.setToolTip("Primary vibration frequency detected (Hz)")
        results_layout.addWidget(self._vibration_label)

        layout.addWidget(results_group)

        # Export
        export_layout = QHBoxLayout()
        self._export_motion_btn = QPushButton("Export CSV")
        self._export_motion_btn.setToolTip(
            "Export tracking data to CSV file.\n"
            "Includes frame numbers, displacements, and velocities."
        )
        export_layout.addWidget(self._export_motion_btn)
        layout.addLayout(export_layout)

        layout.addStretch()

        return widget

    def _create_measurement_tab(self) -> QWidget:
        """Create measurement controls."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Calibration
        cal_group = QGroupBox("Calibration")
        cal_layout = QVBoxLayout(cal_group)

        cal_btn_layout = QHBoxLayout()
        self._calibrate_btn = QPushButton("Calibrate")
        self._calibrate_btn.setToolTip(
            "Click to enter calibration mode.\n"
            "1. Click two points on something of known size\n"
            "2. Enter the real-world distance below\n"
            "3. All measurements will use this scale"
        )
        cal_btn_layout.addWidget(self._calibrate_btn)
        cal_layout.addLayout(cal_btn_layout)

        ref_layout = QHBoxLayout()
        ref_layout.addWidget(QLabel("Reference:"))
        self._ref_distance_spin = QDoubleSpinBox()
        self._ref_distance_spin.setRange(0.1, 10000)
        self._ref_distance_spin.setValue(10.0)
        self._ref_distance_spin.setDecimals(2)
        self._ref_distance_spin.setToolTip(
            "Enter the real-world length of your calibration line.\n"
            "Set this BEFORE clicking Calibrate."
        )
        ref_layout.addWidget(self._ref_distance_spin)

        self._unit_combo = QComboBox()
        self._unit_combo.addItems(["mm", "cm", "inch", "px"])
        self._unit_combo.setToolTip("Unit of measurement for calibration and results")
        ref_layout.addWidget(self._unit_combo)
        cal_layout.addLayout(ref_layout)

        self._cal_status_label = QLabel("Not calibrated")
        self._cal_status_label.setToolTip("Current calibration status (pixels per unit)")
        cal_layout.addWidget(self._cal_status_label)

        layout.addWidget(cal_group)

        # Measurement tools
        tools_group = QGroupBox("Measurement Tools")
        tools_layout = QVBoxLayout(tools_group)

        self._distance_btn = QPushButton("Distance")
        self._distance_btn.setCheckable(True)
        self._distance_btn.setToolTip(
            "Measure distance between two points.\n"
            "Click start point, then end point on the video."
        )
        tools_layout.addWidget(self._distance_btn)

        self._angle_btn = QPushButton("Angle")
        self._angle_btn.setCheckable(True)
        self._angle_btn.setToolTip(
            "Measure angle between three points.\n"
            "Click: first arm end → vertex → second arm end.\n"
            "Angle is measured at the middle (vertex) point."
        )
        tools_layout.addWidget(self._angle_btn)

        self._rectangle_btn = QPushButton("Rectangle")
        self._rectangle_btn.setCheckable(True)
        self._rectangle_btn.setToolTip(
            "Measure rectangular area.\n"
            "Click two opposite corners of the rectangle."
        )
        tools_layout.addWidget(self._rectangle_btn)

        self._circle_btn = QPushButton("Circle")
        self._circle_btn.setCheckable(True)
        self._circle_btn.setToolTip(
            "Measure circular area.\n"
            "Click center point, then any point on the edge."
        )
        tools_layout.addWidget(self._circle_btn)

        self._clear_measurements_btn = QPushButton("Clear All")
        self._clear_measurements_btn.setToolTip("Remove all measurements from the video")
        tools_layout.addWidget(self._clear_measurements_btn)

        layout.addWidget(tools_group)

        # Measurements list
        list_group = QGroupBox("Measurements")
        list_layout = QVBoxLayout(list_group)

        self._measurements_label = QLabel("No measurements")
        self._measurements_label.setToolTip("List of all measurements taken")
        list_layout.addWidget(self._measurements_label)

        self._export_measurements_btn = QPushButton("Export")
        self._export_measurements_btn.setToolTip("Save all measurements to JSON file")
        list_layout.addWidget(self._export_measurements_btn)

        layout.addWidget(list_group)

        layout.addStretch()

        return widget

    def _create_enhancement_tab(self) -> QWidget:
        """Create image enhancement controls."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Brightness/Contrast
        bc_group = QGroupBox("Brightness / Contrast")
        bc_layout = QVBoxLayout(bc_group)

        # Brightness
        bright_layout = QHBoxLayout()
        bright_layout.addWidget(QLabel("Brightness:"))
        self._brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self._brightness_slider.setRange(-100, 100)
        self._brightness_slider.setValue(0)
        self._brightness_slider.setToolTip("Adjust image brightness (-100 to +100)")
        bright_layout.addWidget(self._brightness_slider)
        self._brightness_value = QLabel("0")
        self._brightness_value.setMinimumWidth(30)
        bright_layout.addWidget(self._brightness_value)
        bc_layout.addLayout(bright_layout)

        # Contrast
        contrast_layout = QHBoxLayout()
        contrast_layout.addWidget(QLabel("Contrast:"))
        self._contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self._contrast_slider.setRange(50, 200)
        self._contrast_slider.setValue(100)
        self._contrast_slider.setToolTip(
            "Adjust image contrast (0.5x to 2.0x).\n"
            "Higher = more difference between light and dark."
        )
        contrast_layout.addWidget(self._contrast_slider)
        self._contrast_value = QLabel("1.0")
        self._contrast_value.setMinimumWidth(30)
        contrast_layout.addWidget(self._contrast_value)
        bc_layout.addLayout(contrast_layout)

        # Gamma
        gamma_layout = QHBoxLayout()
        gamma_layout.addWidget(QLabel("Gamma:"))
        self._gamma_slider = QSlider(Qt.Orientation.Horizontal)
        self._gamma_slider.setRange(50, 200)
        self._gamma_slider.setValue(100)
        self._gamma_slider.setToolTip(
            "Adjust gamma curve (0.5 to 2.0).\n"
            "< 1.0 = brighter midtones\n"
            "> 1.0 = darker midtones"
        )
        gamma_layout.addWidget(self._gamma_slider)
        self._gamma_value = QLabel("1.0")
        self._gamma_value.setMinimumWidth(30)
        gamma_layout.addWidget(self._gamma_value)
        bc_layout.addLayout(gamma_layout)

        self._auto_bc_btn = QPushButton("Auto")
        self._auto_bc_btn.setToolTip("Automatically optimize brightness and contrast")
        bc_layout.addWidget(self._auto_bc_btn)

        layout.addWidget(bc_group)

        # Histogram
        hist_group = QGroupBox("Histogram")
        hist_layout = QVBoxLayout(hist_group)

        self._hist_eq_check = QCheckBox("Histogram Equalization")
        self._hist_eq_check.setToolTip(
            "Spread pixel values across full range.\n"
            "Improves contrast in low-contrast images."
        )
        hist_layout.addWidget(self._hist_eq_check)

        self._clahe_check = QCheckBox("CLAHE (Adaptive)")
        self._clahe_check.setToolTip(
            "Contrast Limited Adaptive Histogram Equalization.\n"
            "Better than standard equalization for local contrast.\n"
            "Prevents over-amplification of noise."
        )
        hist_layout.addWidget(self._clahe_check)

        layout.addWidget(hist_group)

        # Filters
        filter_group = QGroupBox("Filters")
        filter_layout = QVBoxLayout(filter_group)

        self._sharpen_check = QCheckBox("Sharpen")
        self._sharpen_check.setToolTip("Enhance edges and fine details")
        filter_layout.addWidget(self._sharpen_check)

        self._denoise_check = QCheckBox("Denoise")
        self._denoise_check.setToolTip(
            "Reduce image noise (slower processing).\n"
            "Useful for low-light or high-ISO footage."
        )
        filter_layout.addWidget(self._denoise_check)

        self._edge_check = QCheckBox("Edge Enhance")
        self._edge_check.setToolTip("Highlight edges in the image")
        filter_layout.addWidget(self._edge_check)

        layout.addWidget(filter_group)

        # Zoom
        zoom_group = QGroupBox("Digital Zoom")
        zoom_layout = QVBoxLayout(zoom_group)

        zoom_slider_layout = QHBoxLayout()
        zoom_slider_layout.addWidget(QLabel("Zoom:"))
        self._zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self._zoom_slider.setRange(10, 100)
        self._zoom_slider.setValue(10)
        self._zoom_slider.setToolTip("Digital zoom level (1x to 10x)")
        zoom_slider_layout.addWidget(self._zoom_slider)
        self._zoom_value = QLabel("1.0x")
        self._zoom_value.setMinimumWidth(40)
        zoom_slider_layout.addWidget(self._zoom_value)
        zoom_layout.addLayout(zoom_slider_layout)

        self._zoom_info = QLabel("Click on video to set zoom center")
        self._zoom_info.setStyleSheet("color: gray; font-size: 10px;")
        zoom_layout.addWidget(self._zoom_info)

        layout.addWidget(zoom_group)

        # Reset
        self._reset_enhance_btn = QPushButton("Reset All")
        self._reset_enhance_btn.setToolTip("Reset all enhancement settings to defaults")
        layout.addWidget(self._reset_enhance_btn)

        layout.addStretch()

        return widget

    def _connect_signals(self) -> None:
        """Connect internal signals."""
        # Tracking
        self._method_combo.currentIndexChanged.connect(self._on_method_changed)
        self._roi_btn.clicked.connect(self.roi_selection_requested.emit)
        self._clear_roi_btn.clicked.connect(self._clear_roi)
        self._start_tracking_btn.toggled.connect(self._on_tracking_toggled)
        self._set_reference_btn.clicked.connect(self._on_set_reference)
        self._vibration_btn.clicked.connect(self._on_analyze_vibration)
        self._export_motion_btn.clicked.connect(self._on_export_motion)

        # Measurement
        self._calibrate_btn.clicked.connect(self._on_calibrate)
        self._distance_btn.toggled.connect(lambda c: self._on_measurement_tool("distance", c))
        self._angle_btn.toggled.connect(lambda c: self._on_measurement_tool("angle", c))
        self._rectangle_btn.toggled.connect(lambda c: self._on_measurement_tool("rectangle", c))
        self._circle_btn.toggled.connect(lambda c: self._on_measurement_tool("circle", c))
        self._clear_measurements_btn.clicked.connect(self._clear_measurements)
        self._export_measurements_btn.clicked.connect(self._on_export_measurements)

        # Enhancement
        self._brightness_slider.valueChanged.connect(self._on_brightness_changed)
        self._contrast_slider.valueChanged.connect(self._on_contrast_changed)
        self._gamma_slider.valueChanged.connect(self._on_gamma_changed)
        self._auto_bc_btn.clicked.connect(self._on_auto_bc)
        self._hist_eq_check.toggled.connect(self._on_hist_eq_changed)
        self._clahe_check.toggled.connect(self._on_clahe_changed)
        self._sharpen_check.toggled.connect(self._on_sharpen_changed)
        self._denoise_check.toggled.connect(self._on_denoise_changed)
        self._edge_check.toggled.connect(self._on_edge_changed)
        self._zoom_slider.valueChanged.connect(self._on_zoom_changed)
        self._reset_enhance_btn.clicked.connect(self._reset_enhancement)

    # Tracking methods
    def _on_method_changed(self, index: int) -> None:
        method = self._method_combo.currentData()
        self._tracker.method = method

    def set_roi(self, x: int, y: int, w: int, h: int) -> None:
        """Set the tracking ROI."""
        roi = ROI(x, y, w, h)
        self._tracker.roi = roi
        self._roi_label.setText(f"ROI: ({x}, {y}) {w}x{h}")

    def _clear_roi(self) -> None:
        self._tracker.roi = None
        self._roi_label.setText("ROI: Not set")

    def _on_tracking_toggled(self, checked: bool) -> None:
        self._is_tracking = checked
        if checked:
            self._start_tracking_btn.setText("Stop Tracking")
            self.tracking_started.emit()
        else:
            self._start_tracking_btn.setText("Start Tracking")
            self.tracking_stopped.emit()

    def _on_set_reference(self) -> None:
        """Request setting reference frame (handled by parent)."""
        pass  # Parent will call set_reference_frame

    def set_reference_frame(self, frame: np.ndarray) -> None:
        """Set the reference frame for tracking."""
        self._tracker.set_reference_frame(frame)

    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp: float,
    ) -> Optional[MotionData]:
        """Process a frame for tracking."""
        if not self._is_tracking:
            return None

        motion = self._tracker.process_frame(frame, frame_number, timestamp)

        if motion:
            self._displacement_label.setText(
                f"Displacement: {motion.displacement_magnitude:.2f} px"
            )
            self._velocity_label.setText(
                f"Velocity: {motion.velocity_magnitude:.2f} px/frame"
            )

        return motion

    def _on_analyze_vibration(self) -> None:
        """Analyze vibration from tracking data."""
        # Get FPS from parent (would need to be passed in)
        fps = 120.0  # Default, should be set properly

        analysis = self._tracker.analyze_vibration(fps)
        if analysis:
            self._vibration_label.setText(
                f"Dominant freq: {analysis.dominant_frequency_hz:.1f} Hz"
            )
            # Could show more detailed analysis in a dialog
        else:
            QMessageBox.warning(
                self, "Analysis Error",
                "Not enough tracking data for vibration analysis"
            )

    def _on_export_motion(self) -> None:
        """Export motion data to CSV."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Motion Data",
            "motion_data.csv",
            "CSV Files (*.csv)",
        )
        if path:
            self._tracker.export_csv(path)

    # Measurement methods
    def _on_calibrate(self) -> None:
        self.calibration_requested.emit()

    def set_calibration(self, pixel_distance: float) -> None:
        """Set calibration from measured pixel distance."""
        real_distance = self._ref_distance_spin.value()
        unit = self._unit_combo.currentText()

        self._measurement.calibrate_from_reference(pixel_distance, real_distance, unit)
        self._cal_status_label.setText(
            f"Calibrated: {self._measurement.calibration.pixels_per_unit:.2f} px/{unit}"
        )

    def _on_measurement_tool(self, tool: str, checked: bool) -> None:
        """Handle measurement tool button toggle."""
        if checked:
            # Uncheck other tools
            for btn, name in [
                (self._distance_btn, "distance"),
                (self._angle_btn, "angle"),
                (self._rectangle_btn, "rectangle"),
                (self._circle_btn, "circle"),
            ]:
                if name != tool:
                    btn.setChecked(False)

            self._measurement_mode = tool
        else:
            self._measurement_mode = ""

        self.measurement_mode_changed.emit(self._measurement_mode)

    def add_measurement_point(self, x: float, y: float) -> Optional[Measurement]:
        """Add a point to current measurement (called by parent on click)."""
        # This would be handled by the parent widget coordinating clicks
        pass

    def _clear_measurements(self) -> None:
        self._measurement.clear_measurements()
        self._update_measurements_label()

    def _update_measurements_label(self) -> None:
        measurements = self._measurement.measurements
        if measurements:
            lines = []
            for m in measurements:
                value, unit = self._measurement.get_measurement_value(m)
                lines.append(f"{m.type.name}: {value:.2f} {unit}")
            self._measurements_label.setText("\n".join(lines))
        else:
            self._measurements_label.setText("No measurements")

    def _on_export_measurements(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Measurements",
            "measurements.json",
            "JSON Files (*.json)",
        )
        if path:
            self._measurement.export_measurements(Path(path))

    # Enhancement methods
    def _on_brightness_changed(self, value: int) -> None:
        self._brightness_value.setText(str(value))
        self._enhancer.settings.brightness = value
        self.enhancement_changed.emit()

    def _on_contrast_changed(self, value: int) -> None:
        contrast = value / 100.0
        self._contrast_value.setText(f"{contrast:.1f}")
        self._enhancer.settings.contrast = contrast
        self.enhancement_changed.emit()

    def _on_gamma_changed(self, value: int) -> None:
        gamma = value / 100.0
        self._gamma_value.setText(f"{gamma:.1f}")
        self._enhancer.settings.gamma = gamma
        self.enhancement_changed.emit()

    def _on_auto_bc(self) -> None:
        """Auto brightness/contrast would need current frame."""
        pass  # Parent would provide frame and call auto_brightness_contrast

    def _on_hist_eq_changed(self, checked: bool) -> None:
        self._enhancer.settings.histogram_equalization = checked
        if checked:
            self._clahe_check.setChecked(False)
        self.enhancement_changed.emit()

    def _on_clahe_changed(self, checked: bool) -> None:
        self._enhancer.settings.clahe_enabled = checked
        if checked:
            self._hist_eq_check.setChecked(False)
        self.enhancement_changed.emit()

    def _on_sharpen_changed(self, checked: bool) -> None:
        self._enhancer.settings.sharpen_enabled = checked
        self.enhancement_changed.emit()

    def _on_denoise_changed(self, checked: bool) -> None:
        self._enhancer.settings.denoise_enabled = checked
        self.enhancement_changed.emit()

    def _on_edge_changed(self, checked: bool) -> None:
        self._enhancer.settings.edge_enhance_enabled = checked
        self.enhancement_changed.emit()

    def _on_zoom_changed(self, value: int) -> None:
        zoom = value / 10.0
        self._zoom_value.setText(f"{zoom:.1f}x")
        self._enhancer.settings.zoom_factor = zoom
        self.enhancement_changed.emit()

    def set_zoom_center(self, x: float, y: float) -> None:
        """Set zoom center from click."""
        self._enhancer.set_zoom_center(x, y)
        self.enhancement_changed.emit()

    def _reset_enhancement(self) -> None:
        self._enhancer.reset()

        # Reset UI
        self._brightness_slider.setValue(0)
        self._contrast_slider.setValue(100)
        self._gamma_slider.setValue(100)
        self._hist_eq_check.setChecked(False)
        self._clahe_check.setChecked(False)
        self._sharpen_check.setChecked(False)
        self._denoise_check.setChecked(False)
        self._edge_check.setChecked(False)
        self._zoom_slider.setValue(10)

        self.enhancement_changed.emit()

    def apply_enhancement(self, frame: np.ndarray) -> np.ndarray:
        """Apply current enhancement settings to frame."""
        return self._enhancer.apply(frame)
