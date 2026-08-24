"""
rpi_infer.py  (Raspberry Pi 5)
==============================
INT8 TensorFlow-Lite inference for the 2-class drone/bird model.

  class 0 = drone  -> TRACKED (green box, ID, center coords printed)
  class 1 = bird   -> IGNORED (orange box, no track) - confuser rejection

Optimized for the Pi 5 CPU (tflite-runtime + XNNPACK). Handles INT8
dequantization, YOLO11 head decode, class-aware NMS, and simple
centroid tracking of drones across frames.

Usage:
  # Pi camera (Picamera2)
  python rpi_infer.py --source picamera

  # USB/webcam index
  python rpi_infer.py --source 0

  # video file
  python rpi_infer.py --source path/to/video.mp4
"""

import argparse
import time
import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:
    try:
        from tensorflow.lite import Interpreter, load_delegate
    except ImportError:
        Interpreter = None
        load_delegate = None

import cv2

MODEL = "best_new.tflite"
IMG_SIZE = 640
CLASS_NAMES = ["drone", "bird"]
DRONE_ID = 0
CONF_DEFAULT = 0.25
IOU_DEFAULT = 0.45


def load_interpreter(model_path: str):
    delegates = []
    if load_delegate is not None:
        try:
            delegates.append(load_delegate("libedgetpu.so.1"))  # not used; safe no-op
        except Exception:
            pass
        try:
            delegates.append(load_delegate("libxnnpack.so"))
        except Exception:
            pass
    if delegates:
        return Interpreter(model_path=model_path, experimental_delegates=delegates)
    return Interpreter(model_path=model_path)


def open_source(source):
    """Return (getter, stop) where getter() -> frame or None."""
    if isinstance(source, str) and source.lower() == "picamera":
        try:
            from picamera2 import Picamera2
            cam = Picamera2()
            cam.preview_configuration.main.size = (640, 480)
            cam.preview_configuration.main.format = "RGB888"
            cam.configure("preview")
            cam.start()
            def getter():
                arr = cam.capture_array()
                return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            return getter, (lambda: cam.stop())
        except Exception as e:
            print(f"[WARN] Picamera2 unavailable ({e}); falling back to webcam 0")
            source = 0
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {source}")
    def getter():
        ok, f = cap.read()
        return f if ok else None
    return getter, (lambda: cap.release())


def preprocess(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, 0)


def quantize_input(tensor, inp):
    scale, zp = inp["quantization"]
    if scale == 0:
        return tensor.astype(inp["dtype"])
    qmin = np.iinfo(inp["dtype"]).min
    qmax = np.iinfo(inp["dtype"]).max
    q = (tensor / scale + zp).clip(qmin, qmax).astype(inp["dtype"])
    return q


def decode(pred, conf_thresh):
    """pred: (8400, 4+nc) in normalized [0,1] box coords + raw class scores."""
    nc = pred.shape[1] - 4
    boxes = pred[:, :4]
    scores = 1.0 / (1.0 + np.exp(-pred[:, 4:]))  # sigmoid
    cls = np.argmax(scores, axis=1)
    best = scores[np.arange(len(cls)), cls]
    mask = best > conf_thresh
    boxes, cls, best = boxes[mask], cls[mask], best[mask]
    # to pixel coords in 640 space
    x1 = (boxes[:, 0] - boxes[:, 2] / 2) * IMG_SIZE
    y1 = (boxes[:, 1] - boxes[:, 3] / 2) * IMG_SIZE
    x2 = (boxes[:, 0] + boxes[:, 2] / 2) * IMG_SIZE
    y2 = (boxes[:, 1] + boxes[:, 3] / 2) * IMG_SIZE
    return np.stack([x1, y1, x2, y2], axis=1), cls, best


def nms(boxes, scores, iou_thresh):
    idx = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.0, iou_thresh)
    if isinstance(idx, tuple):
        idx = idx[0]
    return np.array(idx, dtype=int) if len(idx) else np.array([], dtype=int)


class DroneTracker:
    """Minimal centroid tracker for drones (class 0)."""
    def __init__(self, max_dist=80, max_lost=30):
        self.next_id = 1
        self.tracks = {}      # id -> {"cx":, "cy":, "lost":}
        self.max_dist = max_dist
        self.max_lost = max_lost

    def update(self, dets):
        """dets: list of (cx, cy). Returns list of (id, cx, cy)."""
        result = []
        for (cx, cy) in dets:
            best_id, best_d = None, self.max_dist
            for tid, t in self.tracks.items():
                d = ((t["cx"] - cx) ** 2 + (t["cy"] - cy) ** 2) ** 0.5
                if d < best_d:
                    best_id, best_d = tid, d
            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
            self.tracks[best_id] = {"cx": cx, "cy": cy, "lost": 0}
            result.append((best_id, cx, cy))
        for tid in list(self.tracks.keys()):
            if tid not in [r[0] for r in result]:
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > self.max_lost:
                    del self.tracks[tid]
        return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="picamera")
    ap.add_argument("--conf", type=float, default=CONF_DEFAULT)
    ap.add_argument("--iou", type=float, default=IOU_DEFAULT)
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    interp = load_interpreter(args.model)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    outp = interp.get_output_details()[0]

    getter, stop = open_source(args.source)
    tracker = DroneTracker()
    print(f"[INFO] model={args.model}  conf={args.conf}  iou={args.iou}")
    print("[INFO] drones are tracked (green); birds shown orange (ignored)")

    frames = 0
    t0 = time.time()
    try:
        while True:
            frame = getter()
            if frame is None:
                break
            H, W = frame.shape[:2]
            x = quantize_input(preprocess(frame), inp)
            interp.set_tensor(inp["index"], x)
            interp.invoke()
            raw = interp.get_tensor(outp["index"])
            oscale, ozp = outp["quantization"]
            pred = (raw.astype(np.float32) - ozp) * oscale
            pred = np.reshape(pred, (-1, pred.shape[-1]))
            if pred.shape[0] != 8400 and pred.shape[1] == 8400:
                pred = pred.T

            boxes, cls, scores = decode(pred, args.conf)
            result = []
            for c in range(len(CLASS_NAMES)):
                ci = np.where(cls == c)[0]
                if len(ci) == 0:
                    continue
                keep = nms(boxes[ci], scores[ci], args.iou)
                for k in keep:
                    i = ci[k]
                    x1, y1, x2, y2 = boxes[i]
                    # scale 640 -> frame
                    sx, sy = W / IMG_SIZE, H / IMG_SIZE
                    x1, x2 = x1 * sx, x2 * sx
                    y1, y2 = y1 * sy, y2 * sy
                    result.append((c, int(x1), int(y1), int(x2), int(y2), float(scores[i])))

            # track drones
            drone_dets = [( (x1+x2)//2, (y1+y2)//2 ) for (c, x1,y1,x2,y2,_) in result if c == DRONE_ID]
            tracked = tracker.update(drone_dets) if drone_dets else []

            # draw
            for (c, x1, y1, x2, y2, sc) in result:
                if c == DRONE_ID:
                    col = (0, 255, 0)
                    label = "drone"
                else:
                    col = (0, 165, 255)  # orange = ignored
                    label = "bird (ignored)"
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                cv2.putText(frame, f"{label} {sc:.2f}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

            for (tid, cx, cy) in tracked:
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, f"DRONE#{tid} ({cx},{cy})", (cx + 6, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                print(f"[TRACK] drone#{tid} center=({cx},{cy})")

            fps = frames / (time.time() - t0 + 1e-6)
            cv2.putText(frame, f"FPS {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Drone Tracker (Pi 5)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frames += 1
    finally:
        stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
