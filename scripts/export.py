"""
export.py
=========
Export the trained 2-class YOLO11n weights to:

  best_new.onnx   - FP32 reference (ONNX Runtime)
  best_new.tflite - INT8 quantized (TensorFlow Lite, for Raspberry Pi 5 CPU)

INT8 uses a calibration subset from the training data for minimal accuracy
loss while being ~4x smaller and much faster on the Pi's ARM CPU.

Usage:
    python scripts/export.py
"""

import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "dataset.yaml"
WEIGHTS = ROOT / "runs" / "detect" / "drone_bird" / "weights" / "best.pt"

IMG_SIZE = 640


def main() -> None:
    if not WEIGHTS.exists():
        raise SystemExit(f"Missing {WEIGHTS}. Run scripts/train.py first.")

    model = YOLO(str(WEIGHTS))

    # ---- ONNX (FP32) ----
    print("[EXPORT] ONNX (FP32) ...")
    onnx_src = model.export(
        format="onnx",
        imgsz=IMG_SIZE,
        dynamic=False,
        opset=13,
        simplify=True,
    )
    onnx_dst = ROOT / "best_new.onnx"
    shutil.move(str(onnx_src), str(onnx_dst))
    print(f"  -> {onnx_dst}")

    # ---- TFLite INT8 ----
    print("[EXPORT] TFLite (INT8, calibrated) ...")
    tflite_src = model.export(
        format="tflite",
        imgsz=IMG_SIZE,
        int8=True,
        nms=False,           # we run NMS in deploy/rpi_infer.py
        batch=1,
        data=str(DATA),      # used for INT8 calibration
    )
    tflite_dst = ROOT / "best_new.tflite"
    shutil.move(str(tflite_src), str(tflite_dst))
    print(f"  -> {tflite_dst}")

    # also keep the .pt updated
    pt_dst = ROOT / "best_new.pt"
    shutil.copy(str(WEIGHTS), str(pt_dst))
    print(f"  -> {pt_dst}")

    print("\n[EXPORT] complete. Deploy best_new.tflite on the Raspberry Pi 5.")


if __name__ == "__main__":
    main()
