from ultralytics import YOLO
import cv2
import time
import psutil
import os

MODEL = "best_new.onnx"
VIDEO = "vv7.mp4"

IMG_SIZE = 640
CONF = 0.25
SKIP_WARMUP = 10

model = YOLO(MODEL, task="detect")

cap = cv2.VideoCapture(VIDEO)

process = psutil.Process(os.getpid())

frames = 0
missed = 0
total_inf = 0
conf_sum = 0
conf_count = 0

while True:

    ret, frame = cap.read()
    if not ret:
        break

    start = time.perf_counter()

    # Faster than model.predict()
    results = model(
        frame,
        imgsz=IMG_SIZE,
        conf=CONF,
        verbose=False
    )

    inference = time.perf_counter() - start

    # Ignore warm-up frames
    if frames >= SKIP_WARMUP:
        total_inf += inference

    annotated = frame

    boxes = results[0].boxes

    if len(boxes):

        # Highest confidence detection
        box = boxes[boxes.conf.argmax()]

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        conf = float(box.conf)

        conf_sum += conf
        conf_count += 1

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0,255,0),
            2
        )

        cv2.circle(
            annotated,
            (cx,cy),
            4,
            (0,0,255),
            -1
        )

        cv2.putText(
            annotated,
            f"Drone {conf:.2f}",
            (x1,y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )

        cv2.putText(
            annotated,
            f"({cx},{cy})",
            (cx+5,cy-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255,255,0),
            2
        )

    else:
        missed += 1

    if frames >= SKIP_WARMUP:

        fps = 1 / inference

        cv2.putText(
            annotated,
            f"FPS : {fps:.1f}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,255),
            2
        )

    cv2.imshow("ONNX Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frames += 1

cap.release()
cv2.destroyAllWindows()

effective_frames = max(1, frames - SKIP_WARMUP)

avg_latency = total_inf / effective_frames
avg_fps = effective_frames / total_inf

print("="*60)
print("ONNX RESULTS")
print("="*60)

print(f"Frames               : {frames}")
print(f"Warmup Ignored       : {SKIP_WARMUP}")
print(f"Detection Rate       : {(frames-missed)/frames*100:.2f}%")
print(f"Average FPS          : {avg_fps:.2f}")
print(f"Average Latency(ms)  : {avg_latency*1000:.2f}")

if conf_count:
    print(f"Average Confidence   : {conf_sum/conf_count:.3f}")

print(f"CPU Usage            : {psutil.cpu_percent(interval=1):.1f}%")
print(f"RAM Usage            : {process.memory_info().rss/1024/1024:.2f} MB")