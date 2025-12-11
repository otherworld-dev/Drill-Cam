"""DrillCam application entry point."""

import logging
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from .config.settings import Settings
from .ui.main_window import MainWindow


def setup_logging(debug: bool = False) -> None:
    """Configure application logging."""
    level = logging.DEBUG if debug else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Reduce noise from Qt and other libraries
    logging.getLogger("PySide6").setLevel(logging.WARNING)


def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="DrillCam - High-speed camera for drilling analysis"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock camera (for testing without hardware)",
    )
    parser.add_argument(
        "--reset-settings",
        action="store_true",
        help="Reset settings to defaults",
    )

    args = parser.parse_args()

    setup_logging(debug=args.debug)
    logger = logging.getLogger(__name__)

    logger.info("Starting DrillCam")

    # Load settings (or use defaults if reset requested)
    if args.reset_settings:
        settings = Settings()
        logger.info("Using default settings")
    else:
        settings = Settings.load()

    settings.ensure_directories()

    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("DrillCam")
    app.setOrganizationName("DrillCam")

    # Set application style
    app.setStyle("Fusion")

    # Create and show main window
    window = MainWindow(settings)

    # Restore window geometry
    window.setGeometry(
        settings.window_x,
        settings.window_y,
        settings.window_width,
        settings.window_height,
    )
    window.show()

    logger.info("Application started")

    # Run application
    result = app.exec()

    # Save settings on exit
    try:
        # Save window geometry
        geometry = window.geometry()
        settings.window_x = geometry.x()
        settings.window_y = geometry.y()
        settings.window_width = geometry.width()
        settings.window_height = geometry.height()

        settings.save()
        logger.info("Settings saved")
    except Exception as e:
        logger.error(f"Failed to save settings: {e}")

    return result


if __name__ == "__main__":
    sys.exit(main())
