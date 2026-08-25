# DroneInterceptor — Drone (vs Bird) Detection

> **Two-repo workflow — validate on a laptop, then deploy to the Pi:**
> - 💻 **Laptop validation:** run `scripts/webcam_test.py` with your webcam to check
>   drone/bird accuracy *before* shipping.
> - 🍓 **Raspberry Pi 5 deployment:** [sam-black007/drone-bird-rpi5](https://github.com/sam-black007/drone-bird-rpi5)
>   — INT8 TFLite + `rpi_infer.py` + flight-control `aim_controller.py`. Build/deploy there
>   **after** laptop testing passes.

YOLO11n **2-class** object detector for drone tracking, optimized for the
**Raspberry Pi 5 (8 GB)**. The primary tracked object is the **drone**; the
**bird** class is a confuser/negative class added to cut false positives
(birds mistaken for drones).

| Property | Value |
|---|---|
| Model | YOLO11n |
| Task | Object Detection |
| Classes | 2 — `0: drone` (tracked), `1: bird` (ignored) |
| Input resolution | 640 × 640 |
| Input format | RGB, normalized [0,1] |
| Deployed format | INT8 TensorFlow Lite (CPU, XNNPACK) |
| FP32 ONNX | reference |
| Original model size | ~20 MB (.pt) |
| INT8 TFLite size | ~5 MB |

The model is **INT8-quantized** for fast inference on the Pi 5's ARM CPU
(~4× smaller, much faster, minimal accuracy loss vs FP32).

---

## Repository layout

```
data/
  dataset.yaml            # 2-class YOLO config
  raw/                    # downloaded datasets (gitignored)
  processed/              # merged 2-class YOLO set (gitignored)
scripts/
  download_datasets.py    # Kaggle + Roboflow downloader (DATA_SOURCES)
  prepare_data.py         # merge/remap -> 2-class YOLO + clean/dedup
  validate_dataset.py     # audit boxes, splits, leakage, imbalance
  make_subset.py          # build data/dataset_small.yaml smoke-test set
  train.py                # fine-tune YOLO11n (GPU)
  export.py               # .pt -> ONNX + INT8 Tflite
deploy/
  rpi_infer.py            # Pi 5 INT8 inference + drone tracking
  aim_controller.py       # flight-control loop (pitch/yaw/roll)
  requirements-pi.txt     # Pi runtime deps
docs/                     # design spec / architecture notes
best_new.pt / .onnx / .tflite   # retrained 2-class models
test.py / testOnnx.py     # desktop inference (2-class labels)
```

---

## 1. Training pipeline (on a GPU machine)

### 1.1 Install
```
pip install -r requirements.txt
```

### 1.2 Secrets
Copy `.env.example` to `.env` and fill in:
```
KAGGLE_API_TOKEN=KGAT_xxx        # https://www.kaggle.com/settings -> Create New Token
ROBOFLOW_API_KEY=xxx             # https://app.roboflow.com/settings/api
```
`.env` is gitignored — never commit it.

### 1.3 Download data
```
python scripts/download_datasets.py
```
Pulls:
- `troykueh/multi-class-drone-detection-dataset-yolov8-ready` (Kaggle) → drone
- `gpiosenka/birdies` (Kaggle) → bird
- `muki2003/yolo-drone-detection-dataset` (Kaggle) → drone (added as
  `drone_yolo_extra` in `DATA_SOURCES`; smallest usable extra drone set)
- Optional: uncomment `chunggr/fbd-sv-2024` (12.5 GB flying-bird set) in
  `download_datasets.py` for more bird data.

### 1.4 Prepare (merge → 2-class YOLO)
```
python scripts/prepare_data.py
```
Remaps all drone subtypes → class `0`, birds → class `1`, then splits
90/10 train/val into `data/processed/` and writes `data/dataset.yaml`.

`prepare_data.py` also:
- drops degenerate/tiny boxes (`MIN_BOX_AREA = 1e-4`),
- gives every image a unique name (kills cross-source collisions),
- MD5-dedups so identical images can't land in both train and val
  (prevents train/val leakage).

### 1.4b Validate / clean (recommended)
```
python scripts/validate_dataset.py
```
Audits every label for zero/negative-area boxes, out-of-range coords,
class-ID errors, duplicate images, train/val leakage, and reports the
class balance. The merged set currently trains on **11,914** images
(drone 10,096 / bird 1,818) and validates on **1,293** (drone 1,106 /
bird 187) — 0 annotation errors, 0 leakage. Bird is the minority
confuser class (~11% of instances).

For a quick smoke test use `python scripts/make_subset.py` →
`data/dataset_small.yaml` (1,200 train / 120 val, balanced).

### 1.5 Train
```
python scripts/train.py --data data/dataset.yaml --device 0 --epochs 100 --imgsz 640 --name drone_bird_rtx
python scripts/train.py --resume        # continue if interrupted
```
All flags: `--data`, `--device`, `--epochs`, `--imgsz`, `--batch`
(default 16), `--workers` (default 4), `--cache` (`ram`/`disk`/`False`),
`--plots` (bool), `--name`, `--resume`.

> **RTX 4060 (8 GB VRAM) note:** batch 16 is the stable max at 640 px
> (~5 GB VRAM). batch 32 OOMs the GPU. Use `--workers 4` (8 workers hit a
> Windows shared-memory commit limit, `error 1455`). Expect ~3.5–4 h for
> 100 epochs.

Checkpoints are saved to `runs/detect/<name>/weights/`. If training is
interrupted (power loss, crash), re-run with `--resume` to continue from
the last epoch instead of restarting.

### 1.6 Export
```
python scripts/export.py
```
Produces `best_new.onnx` (FP32) and `best_new.tflite` (INT8, calibrated from
the training set). Both are copied to the repo root.

---

## 2. Raspberry Pi 5 deployment

### 2.1 Install (on the Pi, Raspberry Pi OS 64-bit)
```
sudo apt update && sudo apt install -y python3-opencv python3-numpy
pip install -r deploy/requirements-pi.txt   # tflite-runtime, picamera2
```

### 2.2 Run
Copy `best_new.tflite` and `deploy/rpi_infer.py` to the Pi, then:
```
# Pi camera
python deploy/rpi_infer.py --source picamera

# USB/webcam
python deploy/rpi_infer.py --source 0

# video file
python deploy/rpi_infer.py --source video.mp4
```

What it does:
- Loads the INT8 TFLite model with XNNPACK for fast ARM inference.
- **Tracks drones** (green boxes, ID, center coordinates printed for
  interception/aiming). A simple centroid tracker keeps drone IDs stable
  across frames.
- Draws **birds** in orange and labels them `bird (ignored)` — they are
  detected but never tracked, reducing false drone alarms.
- Prints FPS/latency in the terminal.

Tune with `--conf` (default 0.25) and `--iou` (default 0.45).

---

## 3. Flight-control aim (interceptor drone)

`deploy/aim_controller.py` turns the tracker into a **flight-control loop**: it
detects/tracks the target drone, computes its angle from the camera center, and
sends **pitch / yaw / roll** commands to a flight controller so the interceptor
drone turns to face the target. Assumes the Pi + camera are **onboard** the
interceptor.

How the angle is computed (from lens FOV):

```
yaw_angle   = (cx - W/2) / (W/2) * (FOV_h / 2)     # +right
pitch_angle = -(cy - H/2) / (H/2) * (FOV_v / 2)    # +up
cmd = clamp(kp * degrees(angle), -1, 1)            # P-controller
```

`roll` is kept at 0 — the FC auto-levels. Yaw aims left/right, pitch aims
up/down.

Run (after training + exporting the 2-class model, and `pip install tflite-runtime`):

```powershell
# verify the logic, no hardware
python deploy/aim_controller.py --link print --source 0

# send to a flight controller once hardware is chosen
python deploy/aim_controller.py --link mavlink --port /dev/ttyAMA0 --baud 57600
python deploy/aim_controller.py --link pwm --pins 17,18,19
python deploy/aim_controller.py --link serial --port COM3 --baud 115200
```

Tuning: `--fov-h/--fov-v` (your lens), `--kp` (gain), `--conf/--iou`.
Flight-controller links: `print`, `serial` (pyserial), `mavlink`
(pymavlink, Pixhawk/ArduPilot/PX4), `pwm` (pigpio). Add an RC override /
kill-switch before any autonomous flight.

---

## 4. Notes & next steps
- **Class imbalance:** the merged set is ~89% drone / ~11% bird by
  instance count (drone 10,096 vs bird 1,818 train). Bird is the minority
  confuser class. For higher bird precision, enable the larger
  `fbd-sv-2024` source or add your own bird images under `data/raw/`.
  Audit anytime with `python scripts/validate_dataset.py`.
- **Accuracy vs speed:** INT8 is recommended for the Pi. For maximum accuracy
  (slower) use `best_new.onnx` with ONNX Runtime instead.
- **Hardware accel:** a Hailo AI HAT+ can be added later; that requires
  exporting to `.hef` (not covered here).

---

## Contributors

- **[Carol-here](https://github.com/Carol-here)** — original
  `DroneInterceptor`: single-class (drone) YOLO detection model, base
  codebase, desktop inference (`test.py` / `testOnnx.py`), and the
  starting trained weights.
- **[sam-black007](https://github.com/sam-black007)** (you) — extended
  it to a **2-class (drone + bird)** detector: added bird + extra drone
  datasets and the download / prepare / validate / make-subset pipeline,
  retrained on an RTX 4060, added INT8 TFLite export for the Raspberry
  Pi 5, the Pi inference + drone-tracking script, and the flight-control
  aim loop (`deploy/aim_controller.py`), plus this documentation.
