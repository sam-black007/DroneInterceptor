import cv2
import numpy as np
import tensorflow as tf
import psutil
import time

# ===========================
# CONFIG
# ===========================

MODEL = "best_new.tflite"
VIDEO = "v1.mp4"

IMG_SIZE = 640

CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45
SKIP_WARMUP = 10

# ===========================
# LOAD MODEL
# ===========================

interpreter = tf.lite.Interpreter(model_path=MODEL)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("=" * 60)
print("MODEL INFORMATION")
print("=" * 60)
print("Input Shape :", input_details[0]["shape"])
print("Input Type  :", input_details[0]["dtype"])
print("Output Shape:", output_details[0]["shape"])
print("Output Type :", output_details[0]["dtype"])
print("=" * 60)

# ===========================
# VIDEO
# ===========================

cap = cv2.VideoCapture(VIDEO)

process = psutil.Process()

frames = 0
detections = 0

total_inf = 0

# ===========================
# LOOP
# ===========================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    H, W = frame.shape[:2]

    # -----------------------
    # Preprocess
    # -----------------------

    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0

    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)

    # -----------------------
    # Inference
    # -----------------------

    start = time.perf_counter()

    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])

    inference = time.perf_counter() - start

    if frames >= SKIP_WARMUP:
        total_inf += inference

    fps = 1.0 / inference

    # -----------------------
    # Decode
    # -----------------------

    pred = output[0]

    cx = pred[0]
    cy = pred[1]
    bw = pred[2]
    bh = pred[3]
    conf = pred[4]

    mask = conf > CONF_THRESHOLD

    cx = cx[mask]
    cy = cy[mask]
    bw = bw[mask]
    bh = bh[mask]
    conf = conf[mask]

    boxes = []

    for x, y, w, h in zip(cx, cy, bw, bh):

        x1 = (x - w / 2) * W / IMG_SIZE
        y1 = (y - h / 2) * H / IMG_SIZE

        w = w * W / IMG_SIZE
        h = h * H / IMG_SIZE

        boxes.append([
            int(x1),
            int(y1),
            int(w),
            int(h)
        ])

    if len(boxes):

        indices = cv2.dnn.NMSBoxes(
            boxes,
            conf.tolist(),
            CONF_THRESHOLD,
            NMS_THRESHOLD
        )

        if len(indices):

            detections += 1

            idx = indices.flatten()[0]

            x, y, w, h = boxes[idx]

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            center_x = x + w // 2
            center_y = y + h // 2

            cv2.circle(
                frame,
                (center_x, center_y),
                4,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                frame,
                f"{conf[idx]:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"({center_x},{center_y})",
                (center_x + 5, center_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )

    # -----------------------
    # FPS
    # -----------------------

    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )

    cv2.imshow("TFLite Drone Detection", frame)

    frames += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ===========================
# RESULTS
# ===========================

cap.release()
cv2.destroyAllWindows()

effective_frames = max(frames - SKIP_WARMUP, 1)

avg_latency = total_inf / effective_frames
avg_fps = effective_frames / total_inf

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print(f"Frames                : {frames}")
print(f"Detections            : {detections}")
print(f"Detection Rate        : {(detections/frames)*100:.2f}%")
print(f"Average FPS           : {avg_fps:.2f}")
print(f"Average Latency (ms)  : {avg_latency*1000:.2f}")
print(f"CPU Usage             : {psutil.cpu_percent(interval=1):.1f}%")
print(f"RAM Usage             : {process.memory_info().rss/1024/1024:.2f} MB")