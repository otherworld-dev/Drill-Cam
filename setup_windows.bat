@echo off
REM DrillCam Setup Script for Windows (Development/Testing)

echo ==========================================
echo DrillCam Setup Script for Windows
echo ==========================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.9+ from python.org
    pause
    exit /b 1
)

echo Step 1: Creating Python virtual environment...
echo ----------------------------------------------

set VENV_DIR=%USERPROFILE%\.drillcam-venv

if not exist "%VENV_DIR%" (
    python -m venv "%VENV_DIR%"
    echo Created virtual environment at %VENV_DIR%
) else (
    echo Virtual environment already exists at %VENV_DIR%
)

REM Activate virtual environment
call "%VENV_DIR%\Scripts\activate.bat"

echo.
echo Step 2: Installing Python dependencies...
echo ------------------------------------------

REM Upgrade pip
python -m pip install --upgrade pip

REM Install dependencies
pip install PySide6>=6.5.0
pip install opencv-python>=4.8.0
pip install numpy>=1.24.0
pip install pydantic>=2.0.0
pip install pydantic-settings>=2.0.0

REM Install DrillCam in development mode
pip install -e "%~dp0"

echo.
echo Step 3: Creating launcher script...
echo ------------------------------------

REM Create launcher batch file
set LAUNCHER=%USERPROFILE%\.local\bin\drillcam.bat
if not exist "%USERPROFILE%\.local\bin" mkdir "%USERPROFILE%\.local\bin"

(
echo @echo off
echo call "%VENV_DIR%\Scripts\activate.bat"
echo python -m drillcam.main %%*
) > "%LAUNCHER%"

echo Created launcher at %LAUNCHER%

echo.
echo ==========================================
echo Setup Complete!
echo ==========================================
echo.
echo To run DrillCam:
echo   1. Activate the virtual environment:
echo      %VENV_DIR%\Scripts\activate.bat
echo.
echo   2. Run the application (mock camera for Windows):
echo      drillcam --mock
echo.
echo   Or run directly:
echo      %LAUNCHER% --mock
echo.
echo NOTE: Windows uses mock camera mode for development.
echo       Deploy to Raspberry Pi for real camera support.
echo.

pause
