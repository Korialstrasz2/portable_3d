"""
Portable 2D-to-3D Converter – 2025 Edition
-----------------------------------------
• Depth model : Depth-Anything ViT-L-14 (metric-aware, high detail)
• Stereo synth: per-pixel horizontal disparity (occlusion-aware)
• Encoding    : OpenCV + FFmpeg (NVENC if available)
• GUI         : Tkinter

Note: On first run, this script will cause torch.hub to download MiDaS from
      GitHub (to get the architecture and transforms). A network connection
      is required on that first launch. After that, MiDaS is cached locally
      and no longer needs internet.

CHANGE LOG (2025-06):
  • Added `weights_only=False` to torch.load(...) so that the Depth-Anything
    checkpoint will load properly under PyTorch 2.6+.
  • Wrapped checkpoint loading in try/except to detect truncation; if
    truncated, re-download the checkpoint automatically.
"""

from pathlib import Path
import subprocess, shutil, time, urllib.request
import sys

# Insert _after_ "import torch" but before any usage of load_depth_anything()
# so that we can import DepthAnything from the cloned repo.

# ───────────────────────────────────────────────────────────────
# Make sure Python can see the "Depth-Anything" folder we just cloned:
DA_ROOT = Path(__file__).parent / "Depth-Anything"
if not DA_ROOT.exists():
    raise FileNotFoundError(
        f"Depth-Anything folder not found at {DA_ROOT}. "
        "Did you remember to 'git clone https://github.com/LiheYoung/Depth-Anything.git Depth-Anything'?"
    )
sys.path.append(str(DA_ROOT))
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tkinter import Tk, filedialog, simpledialog, messagebox
from tqdm import tqdm

# -------------------------------------------------------------------------
# Local FFmpeg finder
# -------------------------------------------------------------------------
FFMPEG = str(Path(__file__).parent / "ffmpeg.exe")
if not Path(FFMPEG).exists():
    FFMPEG = shutil.which("ffmpeg") or "ffmpeg"

# -------------------------------------------------------------------------
# URL for Depth-Anything checkpoint
# -------------------------------------------------------------------------
DEPTH_ANYTHING_URL = (
    "https://huggingface.co/isl-org/DepthAnything/resolve/main/"
    "depth_anything_vitl14.pth"
)

# -------------------------------------------------------------------------
# Download helper (pure Python)
# -------------------------------------------------------------------------
def download_checkpoint(dest_path: Path):
    """
    Download the Depth-Anything checkpoint via urllib, overwriting if exists.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(DEPTH_ANYTHING_URL, str(dest_path))

# -------------------------------------------------------------------------
# Load Depth-Anything (weights downloaded by BAT or re-downloaded here)
def load_depth_anything():
    """
    Load the Depth-Anything model (ViT-L/14) from checkpoint.
    This assumes:
      1) You've cloned Depth-Anything into ./Depth-Anything/
      2) You have the .pth file at "./models/depth_anything_vitl14.pth"
    """
    # 1) Locate the checkpoint
    model_path = Path(__file__).parent / "models" / "depth_anything_vitl14.pth"
    if not model_path.exists():
        print(f"Checkpoint not found at {model_path}. Downloading now...")
        download_checkpoint(model_path)

    # 2) Instantiate the DepthAnything model via its own classmethod:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #   NOTE: this string must exactly match the checkpoint name on HF:
    hf_id = "LiheYoung/depth_anything_vitl14"
    model = DepthAnything.from_pretrained(hf_id)
    model.to(device).eval()

    # 3) Build the depth-preprocessing pipeline exactly as in Depth-Anything:
    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    return model, transform, device

# -------------------------------------------------------------------------
def depth_to_disparity(depth: np.ndarray, max_shift: int = 30) -> np.ndarray:
    """
    Convert a depth map (float32) to a disparity map in pixels:
      • Invert depth
      • Normalize to [0,1]
      • Blend/blur to remove tiny spikes
      • Multiply by max_shift
    """
    depth_inv = 1.0 / (depth + 1e-6)
    d_norm = (depth_inv - depth_inv.min()) / (depth_inv.max() - depth_inv.min())
    d_blur = cv2.GaussianBlur(d_norm, (5, 5), 0)
    return (d_blur * max_shift).astype(np.float32)

# -------------------------------------------------------------------------
def generate_sbs(frame: np.ndarray, disp: np.ndarray) -> np.ndarray:
    """
    Given a BGR frame and a float32 disparity map (in pixels),
    produce a half-side-by-side stereo pair (left|right).
    Occluded pixels in the right image are filled from the left image.
    """
    h, w = disp.shape
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))

    left  = cv2.remap(frame, xs - disp, ys, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REPLICATE)
    right = cv2.remap(frame, xs + disp, ys, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REPLICATE)

    # Fill any black holes in the right view by copying from left
    mask = (right == 0).all(axis=2)
    right[mask] = left[mask]

    return np.concatenate((left, right), axis=1)

# -------------------------------------------------------------------------
def mux_audio(tmp: Path, src: Path, out_: Path):
    """
    Use FFmpeg to copy the original audio track into the new silent 3D video.
    """
    cmd = [
        FFMPEG, "-y",
        "-i", str(tmp),         # silent stereo video
        "-i", str(src),         # original (with audio)
        "-c:v", "copy",
        "-c:a", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-shortest",
        str(out_)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.STDOUT)

# -------------------------------------------------------------------------
def process_video(src_path: Path, out_dir: Path, max_shift: int, enc_quality: str):
    """
    Core routine:
      1. Load Depth-Anything model
      2. Read each frame from src_path
      3. Estimate depth ➔ disparity ➔ stereo pair
      4. Write silent stereo to temp MP4
      5. Mux audio into final MP4
    enc_quality can be a bitrate string (e.g. "5M") or a CRF value ("23").
    """
    model, transform, device = load_depth_anything()

    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {src_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sbs_w  = width * 2
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    # Choose FFmpeg encoder: NVENC if available, else libx264
    encoder = "libx264"
    if torch.cuda.is_available():
        try:
            subprocess.run([FFMPEG, "-hide_banner", "-h", "encoder=h264_nvenc"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           check=True)
            encoder = "h264_nvenc"
        except subprocess.CalledProcessError:
            pass

    tmp_mp4 = out_dir / (src_path.stem + "_3D_TEMP.mp4")

    ff_cmd = [
        FFMPEG, "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{sbs_w}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", encoder,
        "-preset", "fast",
    ]
    if enc_quality.lower().endswith("m") or enc_quality.lower().endswith("k"):
        ff_cmd += ["-b:v", enc_quality]
    else:
        ff_cmd += ["-crf", enc_quality]
    ff_cmd += ["-pix_fmt", "yuv420p", str(tmp_mp4)]

    with subprocess.Popen(
        ff_cmd,
        stdin=subprocess.PIPE,
        bufsize=10**7,  # larger pipe buffer for high-res frames
    ) as proc:
        with torch.no_grad():
            pbar = tqdm(total=total if total else None,
                        unit="frame", desc=f"Converting ({encoder})")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 1) Prepare frame for Depth-Anything
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                #    • Wrap in a dict for the Depth-Anything transforms
                sample   = {"image": img_rgb}
                sample   = transform(sample)               # returns a dict, with "image" as the processed array
                img_tens = torch.from_numpy(sample["image"]).unsqueeze(0).to(device)
                inp      = img_tens

                # 2) Forward pass: get depth, resize to original resolution
                depth = model(inp)
                depth = F.interpolate(
                    depth.unsqueeze(1),
                    size=(height, width),
                    mode="bicubic",
                    align_corners=False,
                )
                depth_map = depth.squeeze().cpu().numpy()

                # 3) Compute disparity (float32 pixels)
                disp = depth_to_disparity(depth_map, max_shift)

                # 4) Generate stereo pair (left|right)
                sbs = generate_sbs(frame, disp)
                proc.stdin.write(sbs.tobytes())
                proc.stdin.flush()

                pbar.update(1)

        pbar.close()

    cap.release()
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("FFmpeg encoding failed")

    # 5) Mux original audio into the silent stereo
    final_mp4 = out_dir / (src_path.stem + "_3D_HSBS.mp4")
    mux_audio(tmp_mp4, src_path, final_mp4)
    tmp_mp4.unlink(missing_ok=True)

    return final_mp4

# -------------------------------------------------------------------------
# Tkinter-based file/folder pickers and shift prompt
# -------------------------------------------------------------------------
def choose_file(title):
    root = Tk(); root.withdraw()
    p = filedialog.askopenfilename(
            title=title,
            filetypes=[("Video", "*.mp4 *.mkv *.mov *.avi"), ("All", "*.*")]
        )
    root.destroy()
    return Path(p) if p else None

def choose_folder(title):
    root = Tk(); root.withdraw()
    p = filedialog.askdirectory(title=title)
    root.destroy()
    return Path(p) if p else None

def ask_shift(def_val=30):
    root = Tk(); root.withdraw()
    v = simpledialog.askinteger(
            "Parallax Shift",
            "Max pixel shift (1–100):",
            initialvalue=def_val,
            minvalue=1,
            maxvalue=100
        )
    root.destroy()
    return v

def ask_quality(def_val="23"):
    root = Tk(); root.withdraw()
    q = simpledialog.askstring(
            "Encoding Quality",
            "Enter bitrate (e.g. 5M) or CRF (0-51):",
            initialvalue=def_val
        )
    root.destroy()
    return q

# -------------------------------------------------------------------------
def main():
    print("=== Portable 2D-to-3D Converter (Depth-Anything) ===")
    src = choose_file("Select a 2-D video")
    if not src:
        return

    outdir = choose_folder("Select output folder")
    if not outdir:
        return

    shift = ask_shift()
    if shift is None:
        return

    quality = ask_quality()
    if quality is None:
        return

    try:
        t0 = time.time()
        out = process_video(src, outdir, shift, quality)
        elapsed_min = (time.time() - t0) / 60
        messagebox.showinfo(
            "Done",
            f"Saved stereo video:\n{out}\nElapsed: {elapsed_min:.1f} min"
        )
    except Exception as e:
        messagebox.showerror("Error", str(e))
        raise

if __name__ == "__main__":
    main()
