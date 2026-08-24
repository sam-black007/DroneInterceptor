import cv2
import time
from ultralytics import YOLO

#########################################
# SETTINGS
#########################################

MODEL_PATH = "best_new.pt"
VIDEO_PATH = "vv8.mp4"
OUTPUT_VIDEO = "output_8.mp4"

CONFIDENCE = 0.55
IMGSZ = 512

#########################################

# Force CPU
model = YOLO(MODEL_PATH)
model.to("cpu")

cap = cv2.VideoCapture(VIDEO_PATH)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = cap.get(cv2.CAP_PROP_FPS)

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*'mp4v'),
    video_fps,
    (width, height)
)

frame_count = 0
total_inference = 0

print("=" * 70)
print("Running CPU Inference")
print("=" * 70)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    start = time.perf_counter()

    results = model.predict(
        frame,
        conf=CONFIDENCE,
        imgsz=IMGSZ,
        verbose=False,
        device="cpu"
    )

    inference_ms = (time.perf_counter() - start) * 1000
    total_inference += inference_ms

    annotated = frame.copy()

    if len(results[0].boxes) > 0:

        # highest confidence detection
        box = results[0].boxes[0]

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        conf = float(box.conf[0])

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Draw box
        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0,255,0),
            2
        )

        # Draw center
        cv2.circle(
            annotated,
            (cx,cy),
            5,
            (0,0,255),
            -1
        )

        # Draw coordinates
        cv2.putText(
            annotated,
            f"({cx},{cy})",
            (cx+10, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,0,0),
            2
        )

        # Confidence + class name
        cls_name = results[0].names[int(box.cls[0])]
        cv2.putText(
            annotated,
            f"{cls_name} {conf:.2f}",
            (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )

        print(
            f"Frame {frame_count:5d} | "
            f"Center=({cx:4d},{cy:4d}) | "
            f"Inference={inference_ms:7.2f} ms"
        )

    else:

        print(
            f"Frame {frame_count:5d} | "
            f"No Detection | "
            f"Inference={inference_ms:7.2f} ms"
        )

    fps = 1000 / inference_ms

    cv2.putText(
        annotated,
        f"FPS: {fps:.2f}",
        (20,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,255),
        2
    )

    cv2.putText(
        annotated,
        f"Inference: {inference_ms:.1f} ms",
        (20,60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,255),
        2
    )

    writer.write(annotated)

    cv2.imshow("Drone Detection", annotated)

    if cv2.waitKey(1) == 27:
        break

cap.release()
writer.release()
cv2.destroyAllWindows()

print("\nFinished")

print(f"Frames processed : {frame_count}")
print(f"Average inference: {total_inference/frame_count:.2f} ms")
print(f"Average FPS      : {1000/(total_inference/frame_count):.2f}")