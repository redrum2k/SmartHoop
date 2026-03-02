# SmartHoop

An innovative solution for boredom! SmartHoop is a mini basketball hoop with artificial intelligence that tracks hits and misses and communicates with you after your shots.

This project implements a **real-time mini basketball shot classification system** using OpenCV and Ultralytics YOLOv8. It detects and tracks a red mini basketball and classifies each shot attempt as:

- **hit** – ball goes into the hoop (successful make)
- **backboard** – ball hits the backboard but does not go in
- **miss** – ball does neither (no backboard contact and no make)

---

## Quick Start

### Prerequisites

- Python 3.10+
- Webcam or video file
- Mini basketball hoop mounted on a wall, with camera above (ceiling view)

### Installation

```bash
cd SmartHoop
pip install -r requirements.txt
```

### First Run

1. **Calibrate** the hoop and backboard zones (required once, or when camera position changes):

   ```bash
   python main.py --calibrate --source 0
   ```

   Follow the on-screen prompts to click points defining:
   - **Hoop zone**: Polygon around the rim opening
   - **Backboard zone**: Polygon around the backboard
   - **Attempt ROI** (optional): Region where shots occur (press `s` to skip)

   Zones are saved to `zones.json`.

2. **Run** the classifier:

   ```bash
   python main.py --source 0
   ```

---

## Running

| Command | Description |
|--------|-------------|
| `python main.py --source 0` | Use webcam (default camera) |
| `python main.py --source path/to/video.mp4` | Replay mode with video file |
| `python main.py --calibrate --source 0` | Calibration mode |
| `python main.py --debug` | Show debug overlay (default) |
| `python main.py --no-debug` | Hide debug overlay |
| `python main.py --hsv` | Use HSV red segmentation instead of YOLO (debugging) |
| `python main.py --config my_config.yaml` | Use custom config file |
| `python main.py --zones my_zones.json` | Use custom zones file |

Press `q` in the window to quit.

---

## Calibration

Calibration defines the regions used for shot classification:

1. Start calibration: `python main.py --calibrate --source 0` (or a video path).
2. **Hoop zone**: Click points around the rim opening as seen from the camera. At least 3 points. Press **Enter** when done.
3. **Backboard zone**: Click points around the backboard. At least 3 points. Press **Enter** when done.
4. **Attempt ROI** (optional): Click points around the region where shot attempts occur. Press **Enter** when done, or **s** to skip.
5. Zones are saved to `zones.json`.

Re-run calibration if you move the camera or hoop.

---

## Configuration

Key tunables in `config.yaml`:

| Section | Key | Description |
|--------|-----|-------------|
| Detection | `model` | YOLO model (`yolov8n.pt`, `yolov8s.pt`) |
| Detection | `custom_model_path` | Path to custom-trained model (if any) |
| Detection | `confidence_threshold` | Min detection confidence (0.35) |
| Detection | `inference_size` | Input size for YOLO (416 for speed, 640 for accuracy) |
| Detection | `infer_every_n_frames` | Run YOLO every N frames (1 = every frame) |
| Detection | `use_hsv_fallback` | Use HSV segmentation instead of YOLO |
| Tracking | `max_occlusion_frames` | Frames to keep track during ball loss |
| Tracking | `history_sec` | Trajectory history length in seconds |
| Classification | `K_hit_frames` | Consecutive frames to confirm hit |
| Classification | `K_backboard_frames` | Consecutive frames to confirm backboard |
| Classification | `T_end_sec` | Timeout for attempt (→ miss) |
| Classification | `cooldown_sec` | Debounce between classifications |
| Performance | `attempt_roi_crop` | `[x, y, w, h]` to crop frame before inference |

---

## Custom Training (YOLO)

If the pretrained COCO "sports ball" class (class 32) does not perform well on your red mini basketball, you can train a custom YOLOv8 model.

### 1. Dataset

Create a dataset in Ultralytics YOLO format:

```
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

- Annotate images with bounding boxes (e.g., LabelImg, RectLabel, or Roboflow).
- Save labels as `.txt` per image: `class x_center y_center width height` (normalized 0–1).
- Use a single class (e.g., `0` for "ball").

### 2. Data YAML

Create `ball_dataset.yaml`:

```yaml
path: /path/to/dataset
train: images/train
val: images/val
nc: 1
names: ['ball']
```

### 3. Train

```bash
yolo train model=yolov8n.pt data=ball_dataset.yaml epochs=100 imgsz=640
```

Model is saved to `runs/detect/train/weights/best.pt`.

### 4. Use Custom Model

Set in `config.yaml`:

```yaml
custom_model_path: runs/detect/train/weights/best.pt
```

Or keep `model: yolov8n.pt` and set `custom_model_path` to override.

---

## Replay / Testing with Video

Use a recorded video to test classification without a live camera:

```bash
python main.py --source test_shot.mp4
```

Ensure `zones.json` is calibrated for the same camera view. The counters (hit/backboard/miss) should match expected results for the video.

---

## Raspberry Pi

For Raspberry Pi, use lighter settings in `config.yaml`:

```yaml
inference_size: 320
model: yolov8n.pt
infer_every_n_frames: 2
```

Optional: export YOLOv8 to ONNX for faster inference:

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.export(format="onnx", imgsz=320)
```

Then point `custom_model_path` to the exported `.onnx` file.

---

## Project Structure

```
SmartHoop/
├── main.py              # Entry point, CLI, main loop, calibration
├── config.yaml          # Tunables
├── zones.json           # Calibrated zones (generated)
├── requirements.txt
├── vision/
│   ├── ball_detector.py # YOLO + optional HSV fallback
│   ├── tracker.py       # Nearest-neighbor + optional Kalman
│   └── zones.py         # Zone geometry and helpers
└── logic/
    └── event_classifier.py  # Shot attempt state machine
```

---

## Output

When a shot is classified, a JSON log line is printed:

```json
{"event": "hit", "timestamp": "2025-03-02T12:34:56", "attempt_id": 1, "confidence": 0.92, "notes": "rim-crossing"}
```

The debug overlay shows:
- Ball bounding box and center
- Trajectory polyline
- Hoop, backboard, and attempt ROI zones
- State (IDLE / ATTEMPT / CLASSIFIED)
- Counters (hit / backboard / miss)
- FPS

---

## License

See project license file.
