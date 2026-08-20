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

```text
drone-detection/
│
├── test.py     #code to run pt model 
├── testOnnx.py     #code to run onnx model 
├── testTf.py   #code to run tflite model 
├── README.md
├── best_new.pt       #model formatted in pt
├── best_new.onnx       #model formatted in onnx
└── best_new.tflite    #model formatted in tflite 