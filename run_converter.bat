@echo off
REM ============================================================
REM   Portable 2D->3D Converter – 2025 “Depth-Anything” Edition
REM   ----------------------------------------------------------
REM   • Creates / re-uses a local venv
REM   • Installs CUDA PyTorch + deps
REM   • Downloads FFmpeg (static build)               (~  70 MB)
REM   • Downloads Depth-Anything ViT-L-14 checkpoint  (~ 450 MB)
REM   • Launches the Tk-GUI converter
REM   • Note: Internet is required on first run for MiDaS clone
REM ============================================================

setlocal EnableDelayedExpansion
set "APP_DIR=%~dp0"
set "VENV_DIR=%APP_DIR%env"
set "MODEL_DIR=%APP_DIR%models"
set "MODEL_FILE=%MODEL_DIR%\depth_anything_vitl14.pth"
set "LOG_FILE=%APP_DIR%run_converter.log"

echo [%date% %time%] ---- START ---->> "%LOG_FILE%"

:: ---------- create venv once ----------
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [SETUP] Creating virtual environment...>> "%LOG_FILE%"
    python -m venv "%VENV_DIR%"
)

:: ---------- make sure FFmpeg exists ----------
if not exist "%APP_DIR%ffmpeg.exe" (
    echo [SETUP] Downloading FFmpeg...>> "%LOG_FILE%"
    curl -L -o "%APP_DIR%ffmpeg.zip" ^
      https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip
    tar -xf "%APP_DIR%ffmpeg.zip" -C "%APP_DIR%" --strip-components=3 */ffmpeg.exe
    del "%APP_DIR%ffmpeg.zip"
)

:: ---------- make sure Depth-Anything weights exist ----------
if not exist "%MODEL_FILE%" (
    echo [SETUP] Downloading Depth-Anything weights...>> "%LOG_FILE%"
    mkdir "%MODEL_DIR%" 2>nul
    curl -L -o "%MODEL_FILE%" ^
      https://huggingface.co/isl-org/DepthAnything/resolve/main/depth_anything_vitl14.pth
)

:: ---------- upgrade pip & install wheels ----------
echo [SETUP] Upgrading pip...>> "%LOG_FILE%"
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip >> "%LOG_FILE%" 2>&1

echo [SETUP] Installing / updating core packages...>> "%LOG_FILE%"
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade ^
  torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 ^
  opencv-python numpy tqdm timm >> "%LOG_FILE%" 2>&1

:: ---------- launch the GUI ----------
echo [RUN] Launching converter...>> "%LOG_FILE%"
"%VENV_DIR%\Scripts\python.exe" "%APP_DIR%converter.py" >> "%LOG_FILE%" 2>&1

echo.
echo [DONE] Logs saved to run_converter.log
pause
endlocal
