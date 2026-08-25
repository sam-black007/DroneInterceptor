"""
webcam_test.py
==============
Live accuracy check for the 2-class (drone/bird) model using the laptop
webcam. Draws boxes + class + confidence, shows drone/bird counts and FPS.
Press 'q' to quit.

Usage:
    python scripts/webcam_test.py                 # uses best_new.pt, cam 0
    python scripts/webcam_test.py --conf 0.15     # lower conf to catch more
    python scripts/webcam_test.py --weights runs/detect/drone_bird_rtx/weights/best.pt
"""

import argparse
import sys

import cv2
from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="best_new.pt")
    ap.add_argument("--source", default="0", help="webcam index or video file")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    model = YOLO(args.weights)
    names = model.names  # {0: 'drone', 1: 'bird'}

    src = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"ERROR: cannot open source {args.source}")

    print(f"[WEBCAM] model={args.weights} conf={args.conf}  (press 'q' to quit)")
    frames_read = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            if frames_read == 0:
                sys.exit(
                    "ERROR: opened the camera but could not grab any frames. "
                    "The webcam may be in use by another app, disabled, or the "
                    "wrong --source index. Try --source 1 (or 2), or close other "
                    "camera apps (Zoom, Teams, OBS, browser)."
                )
            print("stream ended")
            break
        frames_read += 1

        res = model.predict(
            frame, conf=args.conf, imgsz=args.imgsz, verbose=False
        )[0]
        annotated = res.plot()

        n = {0: 0, 1: 0}
        for c in res.boxes.cls:
            n[int(c)] += 1
        h, w = annotated.shape[:2]
        cv2.putText(
            annotated,
            f"drone:{n[0]}  bird:{n[1]}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )
        cv2.imshow("webcam test (q to quit)", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
