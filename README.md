# DrillCam

High-speed camera application for monitoring drilling conditions using an OV9281 global shutter camera on Raspberry Pi.

## Features

### Live Capture
- Live camera preview at up to 309fps capture / 30fps display
- Two camera modes:
  - **High Speed**: 640x400 @ 309fps for vibration analysis
  - **High Resolution**: 1280x800 @ 120fps for detail inspection
- Pre-record buffer (captures N seconds before you hit record)
- Crosshair and info overlays
- Snapshot capture (PNG)

### Recording
- Lossless FFV1 video recording
- Two-phase recording: raw capture → background encoding
- Progress bar during encoding
- Automatic file naming with timestamps

### Playback & Analysis
- Load and play recorded videos
- **Slow-motion playback**: 0.01x to 2x speeds (extreme slow-mo for high-fps footage)
- Frame-by-frame stepping (±1, ±10 frames)
- Timeline scrubbing
- Loop playback
- Frame export to PNG

### Media Browser
- Browse saved recordings and snapshots
- Thumbnail previews
- Filter by type (videos/snapshots)
- File info display (size, date, duration)
- Quick open and delete

### Motion Tracking
- **Phase correlation**: Sub-pixel motion detection for vibration analysis
- **Optical flow**: Feature tracking for larger motions
- **Template matching**: Track specific patterns
- ROI selection for focused tracking
- **Vibration frequency analysis**: FFT to find dominant frequencies
- Export motion data to CSV

### Measurement Tools
- **Calibration**: Set pixels-per-mm using known reference
- **Distance measurement**: Point-to-point
- **Angle measurement**: Three-point angle
- **Area measurement**: Rectangle and circle
- Export measurements to JSON

### Image Enhancement
- Brightness / Contrast / Gamma adjustment
- Histogram equalization (standard and CLAHE)
- Sharpening filter
- Noise reduction
- Edge enhancement
- **Digital zoom**: 1x to 10x with pan

### Settings
- Persistent settings (saved on exit)
- Calibration data persistence
- Window geometry remembered

## Hardware Requirements

- Raspberry Pi 5 (8GB recommended)
- Innomaker OV9281 Global Shutter Camera (CSI)
- Active cooling (fan/heatsink required for sustained capture)
- NVMe SSD via USB 3.0 (for recordings > 5 seconds)

## Installation

### Quick Setup (Recommended)

**On Raspberry Pi / Linux:**
```bash
git clone https://github.com/otherworld-dev/Drill-Cam.git
cd Drill-Cam
chmod +x setup.sh
./setup.sh
```

**On Windows (for development):**
```cmd
git clone https://github.com/otherworld-dev/Drill-Cam.git
cd Drill-Cam
setup_windows.bat
```

The setup scripts will:
- Create a Python virtual environment
- Install all dependencies
- Create a launcher script

### Manual Installation

#### On Raspberry Pi

1. Enable the camera in `/boot/firmware/config.txt`:
   ```
   dtoverlay=ov9281
   ```
   Then reboot.

2. Install system dependencies:
   ```bash
   sudo apt update
   sudo apt install -y python3-pip python3-venv python3-picamera2 \
       python3-libcamera python3-opencv ffmpeg libgl1-mesa-glx
   ```

3. Create virtual environment and install:
   ```bash
   python3 -m venv ~/.drillcam-venv --system-site-packages
   source ~/.drillcam-venv/bin/activate
   pip install -e .
   ```

4. Test camera:
   ```bash
   libcamera-hello
   ```

#### On Windows (Development/Testing)

1. Install Python 3.9+ from [python.org](https://www.python.org/downloads/)

2. Install FFmpeg:
   - Download from https://www.gyan.dev/ffmpeg/builds/ (ffmpeg-release-essentials.zip)
   - Extract to `C:\ffmpeg`
   - Add `C:\ffmpeg\bin` to your PATH environment variable

3. Install dependencies:
   ```cmd
   python -m venv %USERPROFILE%\.drillcam-venv
   %USERPROFILE%\.drillcam-venv\Scripts\activate
   pip install -e .
   ```

4. Run with mock camera:
   ```cmd
   python -m drillcam.main --mock
   ```

#### On Ubuntu/Debian (Development)

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv ffmpeg libgl1-mesa-glx
python3 -m venv ~/.drillcam-venv
source ~/.drillcam-venv/bin/activate
pip install -e .
drillcam --mock
```

## Usage

```bash
# Run the application
drillcam

# With debug logging
drillcam --debug

# Force mock camera (for testing)
drillcam --mock

# Reset settings to defaults
drillcam --reset-settings
```

## Keyboard Shortcuts

### Live Capture
- `Space` - Take snapshot
- `Ctrl+O` - Open video file
- `Ctrl+Q` - Quit

### Playback
- `Space` - Play/Pause
- `Left` - Previous frame
- `Right` - Next frame
- `Home` - Go to start
- `End` - Go to end

## Project Structure

```
DrillCam/
├── run.py                  # Launcher script
├── setup.sh                # Linux/Pi setup script
├── setup_windows.bat       # Windows setup script
├── pyproject.toml          # Package configuration
├── requirements.txt        # Python dependencies
└── drillcam/
    ├── main.py                 # Entry point
    ├── config/
    │   ├── settings.py         # Settings with persistence
    │   └── camera_modes.py     # OV9281 resolution/fps presets
    ├── core/
    │   ├── camera_controller.py    # picamera2 wrapper + mock
    │   ├── frame_buffer.py         # Ring buffer for frames
    │   ├── recording_engine.py     # Two-phase FFV1 recording
    │   └── playback_engine.py      # Video playback
    ├── processing/
    │   ├── motion_tracker.py       # Motion/vibration analysis
    │   ├── measurement.py          # Calibration and measurement
    │   └── enhancement.py          # Image enhancement
    ├── workers/
    │   └── capture_worker.py       # Capture thread
    ├── ui/
    │   ├── main_window.py          # Main application window
    │   ├── preview_widget.py       # Live preview
    │   ├── playback_widget.py      # Playback controls
    │   ├── controls_panel.py       # Camera controls
    │   ├── analysis_widget.py      # Analysis tools UI
    │   ├── media_browser.py        # Saved files browser
    │   └── dialogs/
    │       └── progress_dialog.py  # Encoding progress
    └── utils/
        ├── video_io.py             # FFmpeg wrappers
        └── thumbnails.py           # Thumbnail generation
```

## Configuration

Settings are stored in:
- Linux: `~/.config/drillcam/settings.json`
- Windows: `~/.drillcam/settings.json`

Environment variables (override settings):
- `DRILLCAM_OUTPUT_DIR` - Recording output directory
- `DRILLCAM_CAMERA_MODE` - Default camera mode
- `DRILLCAM_PRE_RECORD_SECONDS` - Pre-record buffer duration

## Pi 5 Notes

The OV9281 may require extra setup on Pi 5:

1. libcamera support was merged to the "next" branch - may need to build from source
2. Create tuning file if not detected: `/usr/share/libcamera/ipa/rpi/pisp/uncalibrated.json`
3. No hardware H.264 encoder on Pi 5 - uses FFmpeg software encoding

## Dependencies

```
picamera2          # Camera access (Pi only)
opencv-python      # Image processing
PySide6            # GUI framework
numpy              # Frame buffers
pydantic           # Settings validation
pydantic-settings  # Environment config
ffmpeg             # System package for encoding
```

## License

MIT

## Author

Designed by Adam Morgan
[OtherWorld.Dev](https://otherworld.dev)
