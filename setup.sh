#!/bin/bash
# DrillCam Setup Script for Raspberry Pi OS / Ubuntu

set -e

echo "=========================================="
echo "DrillCam Setup Script"
echo "=========================================="

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
else
    OS=$(uname -s)
fi

echo "Detected OS: $OS"

# Check if running on Raspberry Pi
IS_PI=false
if [ -f /proc/device-tree/model ]; then
    if grep -q "Raspberry Pi" /proc/device-tree/model; then
        IS_PI=true
        PI_MODEL=$(cat /proc/device-tree/model)
        echo "Detected: $PI_MODEL"
    fi
fi

echo ""
echo "Step 1: Installing system dependencies..."
echo "------------------------------------------"

# Install system packages
sudo apt update
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    ffmpeg \
    libgl1-mesa-glx \
    libxcb-xinerama0 \
    libxkbcommon-x11-0 \
    libxcb-cursor0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0

# Install Pi-specific packages
if [ "$IS_PI" = true ]; then
    echo ""
    echo "Installing Raspberry Pi camera packages..."
    sudo apt install -y \
        python3-picamera2 \
        python3-libcamera \
        python3-opencv

    # Check if camera overlay is enabled
    if ! grep -q "dtoverlay=ov9281" /boot/firmware/config.txt 2>/dev/null; then
        echo ""
        echo "WARNING: OV9281 camera overlay not found in /boot/firmware/config.txt"
        echo "You may need to add: dtoverlay=ov9281"
        echo ""
    fi
fi

echo ""
echo "Step 2: Creating Python virtual environment..."
echo "-----------------------------------------------"

# Create virtual environment
VENV_DIR="$HOME/.drillcam-venv"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR" --system-site-packages
    echo "Created virtual environment at $VENV_DIR"
else
    echo "Virtual environment already exists at $VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

echo ""
echo "Step 3: Installing Python dependencies..."
echo "------------------------------------------"

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install PySide6>=6.5.0
pip install opencv-python>=4.8.0
pip install numpy>=1.24.0
pip install pydantic>=2.0.0
pip install pydantic-settings>=2.0.0

# Install DrillCam in development mode
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pip install -e "$SCRIPT_DIR"

echo ""
echo "Step 4: Creating launcher script..."
echo "------------------------------------"

# Create launcher script
LAUNCHER="$HOME/.local/bin/drillcam"
mkdir -p "$HOME/.local/bin"

cat > "$LAUNCHER" << 'LAUNCHER_EOF'
#!/bin/bash
source "$HOME/.drillcam-venv/bin/activate"
python -m drillcam.main "$@"
LAUNCHER_EOF

chmod +x "$LAUNCHER"

echo "Created launcher at $LAUNCHER"

# Add to PATH if needed
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo ""
    echo "NOTE: Add ~/.local/bin to your PATH by adding this to ~/.bashrc:"
    echo '  export PATH="$HOME/.local/bin:$PATH"'
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To run DrillCam:"
echo "  1. Activate the virtual environment:"
echo "     source $VENV_DIR/bin/activate"
echo ""
echo "  2. Run the application:"
echo "     drillcam"
echo ""
echo "  Or run directly:"
echo "     $LAUNCHER"
echo ""

if [ "$IS_PI" = true ]; then
    echo "Raspberry Pi detected - using real camera"
    echo ""
    echo "Make sure your OV9281 camera is connected and enabled:"
    echo "  1. Add 'dtoverlay=ov9281' to /boot/firmware/config.txt"
    echo "  2. Reboot"
    echo "  3. Test with: libcamera-hello"
else
    echo "Non-Pi system detected - use --mock flag for testing:"
    echo "  drillcam --mock"
fi

echo ""
