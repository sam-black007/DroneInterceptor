# Drone Detection Model

YOLO11n-based single-class drone detection model.

This repository contains three versions of the same trained model:

- `best_new.pt` — PyTorch / Ultralytics
- `best_new.onnx` — ONNX Runtime
- `best_new.tflite` — TensorFlow Lite / LiteRT

The intended deployment target is a Raspberry Pi with a camera.

---

## 1. Model Information

| Property | Value |
|---|---|
| Model | YOLO11n |
| Task | Object Detection |
| Classes | 1 |
| Class 0 | Drone |
| Input resolution | 640 × 640 |
| Input format | RGB |
| Output | Bounding boxes + confidence |
| Original model size | ~20.3 MB |
| ONNX model size | ~10.1 MB |
| TFLite model size | ~10.1 MB |

The model is currently **FP32**, not INT8 quantized.

---

## 2. Model Files

Place the three model files in the same directory as `test_camera.py`:

```text
drone-detection/
│
├── test.py
├── testOnnx.py
├── testTf.py
├── README.md
│
└── models/
    ├── best_new.pt
    ├── best_new.onnx
    └── best_new.tflite