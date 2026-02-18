# 🚫📱 Doomscrolling Detector v2

Real-time computer vision app that catches you doomscrolling and plays a meme audio **on loop** until you stop. Uses **head pose + eye gaze + phone detection** — all three signals fused into one verdict.

> *"¡¡ TRABAJA !!"*

---

## ✨ What's New in v2

- **Eye gaze tracking** — iris position detection (not just head tilt)
- **Combined detection** — triggers on head down **OR** eyes looking down at phone
- **Looping audio** — meme plays continuously while doomscrolling, stops when you look away
- **No cooldown** — triggers again immediately if you resume scrolling
- **Image overlay** — custom image appears in bottom-right while triggered
- **"!! TRABAJA !!" banner** — large red text at the bottom of the screen while active
- **Multi-class phone detection** — detects cell phones, laptops, and remotes (COCO 67, 63, 62)
- **Faster trigger** — 2.5s sustained threshold (was 7s)
- **Custom audio & image** — easily swap your own meme files (see guide below)

---

## How It Works

```
Camera Frame (1280×720 @ 30fps)
    │
    ├──► Pose Estimator (MediaPipe FaceLandmarker)
    │       → Head pitch angle via solvePnP
    │       → Eye gaze ratio via iris landmarks (468, 473)
    │       → Combined: head_down OR eyes_down
    │
    ├──► Phone Detector (YOLOv8n)
    │       → COCO classes: cell phone, laptop, remote
    │       → Bounding box in torso/hands zone?
    │
    └──► Doom Logic (fusion)
            → looking_down AND phone_detected for 2.5s?
            → YES → 🔊 Loop audio + 🖼️ Show overlay + !! TRABAJA !!
            → Conditions lost → ⏹ Stop everything
```

### When It Triggers

| Looking Down | Phone Detected | Duration | Result |
|-------------|---------------|----------|--------|
| ✅ | ✅ | ≥ 2.5s | 🔊 **LOOP AUDIO + OVERLAY** |
| ✅ | ❌ | any | Silent |
| ❌ | ✅ | any | Silent |
| ❌ | ❌ | any | Silent |

### False Positive Prevention

- Phone bbox must be in the **torso zone** (lower 75%, central 80%)
- Large objects (monitors, TVs) rejected by max area ratio (30%)
- Tiny noise rejected by min area ratio (0.3%)
- **Grace period** (400ms) prevents brief interruptions from resetting
- **Eye gaze** uses iris landmark positions, not just head tilt

---

## Project Structure

```
doomscrolling/
├── vision/
│   ├── pose_estimator.py      # Head pitch + eye gaze (iris tracking)
│   ├── phone_detector.py      # YOLOv8 multi-class detection
│   ├── inference_engine.py    # Parallel inference orchestrator
│   └── face_landmarker.task   # MediaPipe model (auto-downloaded)
├── logic/
│   └── doom_logic.py          # Fusion state machine + looping audio
├── audio/
│   └── tienes_que_tlabajal.mp3  # Default TTS audio (or use custom)
├── main.py                    # Camera loop + debug overlay + image overlay
├── config.json                # All tunable parameters
├── export_onnx.py             # Export YOLOv8 → ONNX for NPU
├── generate_audio.py          # Generate meme TTS audio
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Download the face model (first time only)

```bash
curl -o vision/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

### 3. Run

```bash
python main.py
```

Press **`q`** in the preview window to quit.

### CLI options

```bash
python main.py --config custom.json   # Use a different config
python main.py --no-preview           # Headless mode (no window)
```

---

## 🎨 Customising Your Meme (Audio & Image)

You can easily swap the audio and image to any meme you want!

### Changing the Audio

1. Place your audio file anywhere on your computer (`.mp3` or `.wav`)
2. Open `config.json` and change the `audio.file_path` to your file:

```json
{
  "audio": {
    "file_path": "C:\\Users\\YourName\\Music\\your_meme.mp3",
    "volume": 0.9
  }
}
```

> **Tip:** The audio loops continuously while you're doomscrolling and stops when you look away. Short clips (2–5 seconds) work best for maximum annoyance!

### Changing the Overlay Image

1. Place your image anywhere (`.jpg`, `.png`, or `.png` with transparency)
2. Open `config.json` and update the `overlay` section:

```json
{
  "overlay": {
    "image_path": "C:\\Users\\YourName\\Pictures\\your_meme.jpg",
    "size": 200,
    "margin": 15
  }
}
```

| Setting | Description |
|---------|-------------|
| `image_path` | Full path to your image file |
| `size` | Max width/height in pixels (keeps aspect ratio) |
| `margin` | Distance from the bottom-right corner in pixels |

> **Tip:** PNG images with transparency will be alpha-blended over the camera feed!

### Using the Default TTS Audio

If you don't have a custom audio file, generate one with:

```bash
python generate_audio.py
```

Then set `audio.file_path` to `"audio/tienes_que_tlabajal.mp3"` in config.

---

## Configuration — `config.json`

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `camera` | `width` / `height` | `1280×720` | Camera resolution |
| `camera` | `fps` | `30` | Target frame rate |
| `pose` | `pitch_threshold_deg` | `15.0` | Head pitch angle (°) for "looking down" |
| `pose` | `eye_gaze_threshold` | `0.07` | Eye gaze sensitivity (lower = more sensitive) |
| `phone_detection` | `confidence_threshold` | `0.30` | YOLO confidence threshold |
| `phone_detection` | `target_classes` | `[67,63,62]` | COCO class IDs to detect |
| `doom_logic` | `sustained_seconds` | `2.5` | How long conditions must hold |
| `doom_logic` | `cooldown_seconds` | `0` | Cooldown after trigger (0 = instant re-trigger) |
| `doom_logic` | `grace_period_ms` | `400` | Brief interruptions allowed without reset |
| `audio` | `file_path` | (custom) | Path to your meme audio file |
| `audio` | `volume` | `0.9` | Audio volume (0.0 – 1.0) |
| `overlay` | `image_path` | (custom) | Path to your meme overlay image |
| `overlay` | `size` | `200` | Max image dimension in pixels |
| `debug` | `show_preview` | `true` | Show OpenCV debug window |
| `debug` | `show_eye_gaze` | `true` | Show eye gaze info in overlay |

---

## NPU / GPU Acceleration

### Option 1: DirectML (Windows NPU/GPU)

```bash
pip install onnxruntime-directml
python export_onnx.py
```

Then in `config.json`:
```json
{
  "phone_detection": { "use_onnx": true },
  "npu": {
    "enabled": true,
    "execution_provider": "DmlExecutionProvider"
  }
}
```

### Option 2: NVIDIA TensorRT

```bash
pip install onnxruntime-gpu
python export_onnx.py
```

```json
{
  "npu": {
    "execution_provider": "TensorrtExecutionProvider"
  }
}
```

### Parallel Inference

Both models run simultaneously via `ThreadPoolExecutor(max_workers=2)`:

- **Thread 1**: Pose estimation (MediaPipe, CPU)
- **Thread 2**: Phone detection (YOLO, NPU/GPU via ONNX)

---

## Tuning Thresholds

| Problem | Adjustment |
|---------|-----------|
| Triggers too easily | Increase `pitch_threshold_deg` (try 25–30°) |
| Doesn't trigger when scrolling | Decrease `pitch_threshold_deg` (try 10°) or `eye_gaze_threshold` (try 0.04) |
| Detects TV/monitor as phone | Increase `confidence_threshold` (try 0.45) |
| Misses phone in hands | Lower `confidence_threshold` (try 0.20) |
| Timer resets on small movements | Increase `grace_period_ms` (try 600–800) |
| Want slower trigger | Increase `sustained_seconds` (try 5–7) |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Head Pose | MediaPipe FaceLandmarker + OpenCV solvePnP |
| Eye Gaze | Iris landmark tracking (landmarks 468, 473) |
| Phone Detection | Ultralytics YOLOv8n / ONNX Runtime |
| Audio | pygame.mixer (looping playback) |
| Camera | OpenCV VideoCapture |
| Parallelism | concurrent.futures.ThreadPoolExecutor |
| NPU | onnxruntime-directml / onnxruntime-gpu |

## License

MIT
