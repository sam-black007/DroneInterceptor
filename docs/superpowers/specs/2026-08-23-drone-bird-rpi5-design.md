# Drone vs Bird Detection — Raspberry Pi 5 Deployment

**Date:** 2026-08-23
**Status:** Approved design, pending implementation plan

---

## 1. Goal

Retrain the existing single-class (drone) YOLO11n detector into a **2-class model
(drone, bird)** so the system discriminates birds from drones and reduces false
positives, then deploy it efficiently on a **Raspberry Pi 5 (8 GB)** using
**CPU-only, INT8-quantized TensorFlow Lite** inference.

The current repo contains only trained weights (`.pt`, `.onnx`, `.tflite`) and
inference scripts — there is no training code and no dataset. This work adds the
full training + export + deployment pipeline.

---

## 2. Architecture / File Layout

```
DroneInterceptor/
├── data/
│   ├── dataset.yaml            # 2-class YOLO config (0=drone, 1=bird)
│   ├── raw/                    # downloaded datasets (gitignored)
│   │   ├── kaggle/
│   │   └── roboflow/
│   └── processed/              # merged 2-class YOLO set (gitignored)
├── scripts/
│   ├── download_datasets.py    # Kaggle + Roboflow downloader
│   ├── prepare_data.py         # normalize/merge/remap -> 2-class YOLO
│   ├── train.py                # fine-tune YOLO11n (RTX 4060)
│   └── export.py               # .pt -> ONNX + INT8 TFLite
├── deploy/
│   ├── rpi_infer.py            # clean INT8 TFLite inference for Pi 5
│   └── requirements-pi.txt     # Pi runtime deps
├── .env.example                # documents required API keys (not committed)
├── requirements.txt            # training/dev deps
├── best_new.pt / .onnx / .tflite   # retrained 2-class models (replaced)
├── test.py / testOnnx.py / testTf.py   # updated for 2 classes (labels)
└── README.md                   # updated for 2-class + Pi 5 INT8
```

Pipeline: **download → prepare → train (this PC) → export INT8 → deploy on Pi.**

---

## 3. Data

### 3.1 Sources
Both require free API keys supplied by the user (never committed):
- **Kaggle**: `KAGGLE_USERNAME` / `KAGGLE_KEY` env vars (or `kaggle.json`).
- **Roboflow**: `ROBOFLOW_API_KEY` env var.

`download_datasets.py` supports both and is driven by a config list of sources.

#### Drone (class 0)
- `troykueh/multi-class-drone-detection-dataset-yolov8-ready` (Kaggle) — 10,000
  images, YOLO labels, 5 drone subtypes. All 5 subtypes merge → `drone`.
- Optional: `muki2003/yolo-drone-detection-dataset` (Kaggle) for variety.
- Optional Roboflow drone dataset(s) supplied by user via API key.

#### Bird (class 1)
- `chunggr/fbd-sv-2024` (Kaggle) — Flying Bird Dataset, ~28,694 frames, **VOC/XML**
  labels, single class `bird`. Chosen because it is *flying* birds — the exact
  confuser of drones in the sky.
- Optional: `gpiosenka/birdies` (Kaggle, YOLO, 2,010 pairs) as supplement.
- Optional Roboflow bird dataset(s) supplied by user.

### 3.2 Normalization (`prepare_data.py`)
- Auto-detect label format per source: YOLO `.txt` vs VOC `.xml`.
- Convert VOC (`xmin,ymin,xmax,ymax`) → YOLO (`x_center,y_center,w,h` normalized).
- Remap: all drone subtypes → class `0`; birds → class `1`.
- Merge, dedupe by image hash, and split **train/val = 90/10** (stratified by class).
- Output to `data/processed/{images,labels}/{train,val}` plus `data/dataset.yaml`.
- Print final per-class image/instance counts so balance can be verified before training.

### 3.3 Class balance / accuracy
The 2-class design directly improves discrimination (bird vs drone) vs the prior
"object vs background" single class. Report mAP@0.5 and mAP@0.5:0.95 per class
after training. If one class is heavily underrepresented, note augmentation or
additional sources as a follow-up (out of scope unless requested).

---

## 4. Training (`train.py`)

- Base weights: `yolo11n.pt` (Ultralytics pretrained). Optionally seed from the
  existing `best_new.pt` for drone transfer; head is reinitialized for `nc=2` by
  Ultralytics when class count differs.
- `imgsz=640`, `epochs=100`, `batch=16–32` (tune to RTX 4060 8 GB), `optimizer=auto`,
  `patience=20` (early stop), `workers` auto.
- Device: CUDA (RTX 4060). Pin/seed for reproducibility.
- Output: `runs/detect/train/weights/best.pt` → copied to `best_new.pt`.

---

## 5. Export (`export.py`)

- **ONNX** (`best_new.onnx`): FP32 reference, `imgsz=640`, dynamic batch optional.
- **INT8 TFLite** (`best_new.tflite`):
  - Calibration: random subset (~100–300 images) from the processed train set.
  - INT8 quantization via Ultralytics `model.export(format="tflite", int8=True,
    nms=True, data=dataset.yaml, imgsz=640)`.
  - Result ≈ 4× smaller and much faster on Pi CPU, minimal accuracy loss.
- Existing `.pt`/`.onnx`/`.tflite` are replaced with the new 2-class versions.

---

## 6. Raspberry Pi 5 Deployment (`deploy/rpi_infer.py`)

CPU-only target with INT8 TFLite.

- Runtime: `tflite-runtime` with XNNPACK delegate for ARM.
- **INT8 handling**: read input/output `quantization_params` (scale, zero_point)
  and dequantize outputs before postprocessing.
- Postprocess: decode YOLO head → per-class score → confidence threshold
  (`0.25`) → per-class NMS (`0.45`) → draw box + label `Drone` / `Bird` + center.
  (Replaces the brittle `testTf.py` which assumed a single detection per frame.)
- Input: `Picamera2` (Pi 5 native) with `cv2.VideoCapture` fallback; also accepts
  a video file / webcam index for testing.
- Prints FPS / latency on the Pi. Optional annotated output save.
- `deploy/requirements-pi.txt`: `tflite-runtime`, `opencv-python-headless`,
  `numpy`, `picamera2` (Pi only).

---

## 7. Environment & Secrets

- `requirements.txt` (training/dev): `ultralytics`, `opencv-python`, `numpy`,
  `roboflow`, `kaggle`, `psutil`, `python-dotenv`.
- `.env.example` documents `KAGGLE_USERNAME`, `KAGGLE_KEY`, `ROBOFLOW_API_KEY`.
  Real `.env` is gitignored. Scripts load via `python-dotenv`.
- Add `data/raw/`, `data/processed/`, `runs/`, `.env` to `.gitignore`.

---

## 8. Validation

- Training: Ultralytics prints mAP@0.5 / mAP@0.5:0.95 per class. Target: drone
  mAP comparable to prior single-class model; bird mAP reported; confusion between
  classes low.
- Export: re-run inference on a few validation frames; confirm INT8 outputs match
  FP32 within tolerance.
- Pi: `rpi_infer.py` runs live; record FPS (target: real-time-ish on Pi 5 CPU;
  exact number depends on INT8 + XNNPACK, expected low double-digit FPS at 640).

---

## 9. Risks / Constraints

- Dataset label quality varies; VOC→YOLO conversion and merges must be verified
  by printed counts and a spot-check of a few label files.
- INT8 may drop a few mAP points; acceptable trade for Pi speed.
- Class imbalance may bias the model; addressed by reporting and optional follow-up.
- Large downloads (tens of GB) — ensure disk space; raw/processed are gitignored so
  the repo stays small (only model weights are committed).

---

## 10. Implementation Steps (summary)

1. Add `requirements.txt`, `.env.example`; update `.gitignore`.
2. `scripts/download_datasets.py` — Kaggle + Roboflow download.
3. `scripts/prepare_data.py` — normalize/merge/remap → 2-class YOLO + `dataset.yaml`.
4. `scripts/train.py` — fine-tune YOLO11n on RTX 4060.
5. `scripts/export.py` — ONNX + INT8 TFLite.
6. `deploy/rpi_infer.py` + `requirements-pi.txt` — Pi 5 INT8 inference.
7. Update `test.py`/`testOnnx.py`/`testTf.py` labels; update `README.md`.
8. Commit training pipeline + new 2-class weights to the fork; push.
