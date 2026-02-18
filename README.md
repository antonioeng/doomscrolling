# 🚫📱 Doomscrolling Detector

Real-time computer vision app that catches you doomscrolling and plays a meme audio to snap you out of it.

Uses **two simultaneous conditions** — head tilted down **AND** a cell phone in your hands — sustained for a configurable duration before triggering. No overlays, no floating windows; just an audio blast.

> *"TIENES QUE TLABAJAL"*

---

## How It Works

```
Camera Frame
    │
    ├──► Pose Estimator (MediaPipe Face Mesh)
    │       → Head pitch angle via solvePnP
    │       → Is head down? (pitch > 25°)
    │
    ├──► Phone Detector (YOLOv8n)
    │       → COCO class "cell phone"
    │       → Bounding box in torso/hands zone?
    │
    └──► Doom Logic (fusion)
            → Both conditions TRUE for 7+ seconds?
            → YES → 🔊 Play audio
            → 60s cooldown (anti-spam)
```

### When It Triggers

| Head Down | Phone Detected | Duration | Result |
|-----------|---------------|----------|--------|
| ✅ | ✅ | ≥ 7s | 🔊 **TRIGGER** |
| ✅ | ❌ | any | Silent |
| ❌ | ✅ | any | Silent |
| ❌ | ❌ | any | Silent |

### False Positive Prevention

- Phone bbox must be in the **torso/hands zone** (lower 70% of frame, central 70%)
- Large objects (monitors, TVs) rejected by area ratio filter
- **Grace period** (500ms) prevents brief interruptions from resetting the timer
- **60s cooldown** after each trigger prevents audio spam

---

## Project Structure

```
doomscrolling/
├── vision/
│   ├── pose_estimator.py      # MediaPipe Face Mesh → head pitch
│   ├── phone_detector.py      # YOLOv8 / ONNX → phone detection
│   └── inference_engine.py    # Parallel inference orchestrator
├── logic/
│   └── doom_logic.py          # Fusion state machine + audio
├── audio/
│   └── tienes_que_tlabajal.mp3
├── main.py                    # Camera loop + debug preview
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

### 2. Generate the meme audio

```bash
python generate_audio.py
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

## Configuration — `config.json`

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `pose` | `pitch_threshold_deg` | `25.0` | Head pitch angle (°) that counts as "looking down" |
| `pose` | `min_detection_confidence` | `0.7` | MediaPipe face detection confidence |
| `phone_detection` | `confidence_threshold` | `0.45` | YOLO confidence threshold for "cell phone" |
| `phone_detection` | `proximity_margin` | `0.15` | Margin (%) defining the torso zone |
| `phone_detection` | `use_onnx` | `false` | Use ONNX Runtime instead of Ultralytics |
| `doom_logic` | `sustained_seconds` | `7` | How long both conditions must hold |
| `doom_logic` | `cooldown_seconds` | `60` | Cooldown after a trigger |
| `doom_logic` | `grace_period_ms` | `500` | Brief interruptions allowed without reset |
| `npu` | `enabled` | `false` | Enable NPU/GPU via DirectML |
| `debug` | `show_preview` | `true` | Show OpenCV debug window |

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

This maximises throughput, especially when the phone model is offloaded to NPU/GPU.

---

## Tuning Thresholds

| Problem | Adjustment |
|---------|-----------|
| Triggers too easily | Increase `pitch_threshold_deg` (try 30–35°) |
| Doesn't trigger when obviously scrolling | Decrease `pitch_threshold_deg` (try 18–20°) |
| Detects TV/monitor as phone | Decrease `proximity_margin` (try 0.20) |
| Misses phone in hands | Lower `confidence_threshold` (try 0.35) |
| Audio fires too often | Increase `cooldown_seconds` |
| Timer resets on small movements | Increase `grace_period_ms` (try 800–1000) |

---

## Packaging as Executable

```bash
pip install pyinstaller
pyinstaller --onefile --add-data "config.json;." --add-data "audio;audio" main.py
```

The executable will be in `dist/main.exe`.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Head Pose | MediaPipe Face Mesh + OpenCV solvePnP |
| Phone Detection | Ultralytics YOLOv8n / ONNX Runtime |
| Audio | pygame.mixer |
| Camera | OpenCV VideoCapture |
| Parallelism | concurrent.futures.ThreadPoolExecutor |
| NPU | onnxruntime-directml / onnxruntime-gpu |

## License

MIT
