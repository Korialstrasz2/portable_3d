──────────────────────────────────────────────────────────────
Portable 2-D ➜ 3-D Converter (2025 Depth-Anything Edition)
──────────────────────────────────────────────────────────────

■ OVERVIEW
   This tool converts any 2-D video into a half-side-by-side (HSBS)
   stereoscopic 3-D video, suitable for VR/AR playback. It uses:
     • Depth-Anything ViT-L-14 (state-of-the-art, metric depth, high detail)
     • CUDA-accelerated PyTorch for depth inference
     • OpenCV + FFmpeg (NVENC if available) for video I/O
     • Tkinter for a minimal GUI

   The entire app—including model weights, FFmpeg, and Python packages—
   lives inside its folder. After the first run, you can move it anywhere.

───────────────────────────────────────────────────────────────────────
■ SYSTEM REQUIREMENTS
   1.  Windows 10 or 11 (64-bit)
   2.  NVIDIA GPU with CUDA 11.8+ (recent driver installed)
   3.  Python 3.9 – 3.12 on PATH (only for first-time run)
       • used to create the virtual environment
       • after that, the included env is used
   4.  ~3 GB free disk space in this folder (for venv, model, ffmpeg, temp)
   5.  Internet connection on **first run** (to allow MiDaS to be cloned)
       • MiDaS is pulled via torch.hub at runtime; afterward it’s cached.

───────────────────────────────────────────────────────────────────────
■ FIRST-TIME SET-UP (one click)
   1. Place these files side by side in a new folder:
        • run_converter.bat
        • converter.py
        • README.txt
   2. Double-click **run_converter.bat**.
      The script will:
        ▸ Create `env\` (virtual environment)
        ▸ pip-install:
            • torch, torchvision, torchaudio (CUDA 11.8)
            • opencv-python, numpy, tqdm, timm
        ▸ Download FFmpeg static build (~70 MB) → `ffmpeg.exe`
        ▸ Download Depth-Anything ViT-L-14 weights (~450 MB) → `models\depth_anything_vitl14.pth`
        ▸ Launch `converter.py` using the new `env\`
      All of that may take **3–6 minutes** (depends on your Internet speed and GPU driver checks).

───────────────────────────────────────────────────────────────────────
■ HOW TO USE
   1. After setup, a small console window and Tkinter dialog will appear.
   2. Choose your source 2-D video (any common format: .mp4, .mkv, .mov, .avi).
   3. Select an output folder (where the 3-D video will be saved).
   4. Enter “Max pixel shift” (1–100). This controls perceived depth:
      • ~15–40 px is usually comfortable on most VR headsets.
   5. The console shows a progress bar (“Converting (NVENC)” or “Converting (CPU)”).
      • GPU (NVENC) encoding happens if your system supports it; otherwise falls back to CPU.
   6. When done, you’ll see a popup with:
        Saved stereo video:
          <output_folder>\YourVideo_3D_HSBS.mp4
        Elapsed: X.Y min
      This final file:
        • Contains side-by-side (left|right) 3-D frames
        • Has the original audio (losslessly copied via FFmpeg)

───────────────────────────────────────────────────────────────────────
■ FILES & FOLDERS AFTER SETUP
   • converter.py              ← main Python script
   • run_converter.bat         ← launcher/installer
   • README.txt                ← this guide
   • env\                      ← Python virtual environment (auto-created)
       (inside env, you’ll see standard venv folders: Scripts\, Lib\, etc.)
   • models\                   ← folder created by BAT
       • depth_anything_vitl14.pth
   • ffmpeg.exe                ← downloaded static FFmpeg build
   • run_converter.log         ← log file (captures setup + runtime messages)

───────────────────────────────────────────────────────────────────────
■ TROUBLESHOOTING

  1. “CUDA not available” or “Torch cannot find a GPU”  
     • Ensure you have an NVIDIA GPU and have installed the latest Game/Studio driver.  
     • If you only have CPU, the script will still work (just slower).

  2. “Could not open VideoWriter with any codec”  
     • Means neither “H264 (NVENC)” nor “mp4v (CPU)” was available.  
     • Solution: install the *K-Lite Codec Pack* (free) or check your OpenCV build.

  3. “No module named 'cv2'” or similar after BAT finishes  
     • Make sure the BAT did not error in the “pip install” step.  
     • Re-run `run_converter.bat` in a console so you can see any pip errors.

  4. “Error: Model weights not found”  
     • Ensure “models\depth_anything_vitl14.pth” exists.  
     • If the BAT’s download failed, fetch manually from:  
       https://huggingface.co/isl-org/DepthAnything/resolve/main/depth_anything_vitl14.pth  
     • Place the file at exactly `models\depth_anything_vitl14.pth`.

  5. “MiDaS clone failed (no internet?)”  
     • On first launch, converter.py will call `torch.hub.load("intel-isl/MiDaS", …)`.  
       This needs internet to clone MiDaS. After that, it’s cached.  
     • If you absolutely cannot connect to the internet, you’d need to pre-clone MiDaS:
         a) Open a separate PowerShell / CMD.  
         b) Run:  
            git clone https://github.com/isl-org/MiDaS.git %USERPROFILE%\.cache\torch\hub\intel-isl_MiDaS  
         c) Then re-run the converter.  
       (Torch will detect the local MiDaS cache and skip downloading.)

───────────────────────────────────────────────────────────────────────
■ PERFORMANCE & TIPS

  • Depth-Anything is a DPT_Large backbone → each 720p frame may take ~1–2 seconds to infer on a
    modern NVIDIA GPU (e.g., RTX 20xx, 30xx, or 40xx). Higher resolutions take longer.

  • If you want faster “preview” (lower quality), reduce the video resolution or use a smaller
    “Max pixel shift.” Also, you could swap the depth model in `converter.py` to MiDaS directly
    by replacing `load_depth_anything()` with plain MiDaS (still via torch.hub).

  • “Max pixel shift” controls 3-D strength. Too large → uncomfortable or “exaggerated” depth.
    Too small → 3-D effect looks flat. Start at 20–30 px, then adjust.

───────────────────────────────────────────────────────────────────────
Enjoy crisp, artifact-free 3-D conversions! 🎉
──────────────────────────────────────────────────────────────
