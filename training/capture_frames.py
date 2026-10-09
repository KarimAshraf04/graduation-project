"""
Collect YOUR OWN training images (clean frames, no drawings) from the camera or a video file.
Saved to training/data/mine/images.

Usage (from C:\\vision):
  venv\\Scripts\\python training\\capture_frames.py                    # webcam 0, auto-save every 0.7s when a person is visible
  venv\\Scripts\\python training\\capture_frames.py --source clip.mp4  # extract frames from a video you filmed
  venv\\Scripts\\python training\\capture_frames.py --every 1.5 --all  # save even with nobody in frame

Keys:  SPACE = save this frame now   A = auto-save on/off   Q = quit
Tips: film from the back, side and front, near and far, different bags, and people WITHOUT bags too.
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "data" / "mine" / "images"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--every", type=float, default=0.7, help="seconds between auto-saves")
    ap.add_argument("--all", action="store_true", help="save even if no person is detected")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    person_model = None
    if not a.all:
        from ultralytics import YOLO
        person_model = YOLO(str(ROOT / "yolo26n-pose.pt"))

    live = a.source.isdigit()
    cap = cv2.VideoCapture(int(a.source), cv2.CAP_DSHOW) if live else cv2.VideoCapture(a.source)
    if live:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    auto, saved, last_save, frame_i = True, 0, -1e9, 0
    prefix = datetime.now().strftime("%Y%m%d_%H%M%S")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_i += 1
        t = time.time() if live else frame_i / src_fps

        has_person = True
        if person_model is not None:
            r = person_model.predict(frame, conf=0.4, imgsz=416, verbose=False)[0]
            has_person = r.boxes is not None and len(r.boxes) > 0

        key = cv2.waitKey(1) & 0xFF
        save_now = key == ord(" ")
        if key in (ord("a"), ord("A")):
            auto = not auto
        if key in (ord("q"), ord("Q"), 27):
            break
        if auto and has_person and t - last_save >= a.every:
            save_now = True
        if save_now:
            cv2.imwrite(str(OUT / f"{prefix}_{saved:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1
            last_save = t

        view = frame.copy()
        cv2.putText(view, f"saved: {saved}   auto: {'ON' if auto else 'OFF'}   person: {has_person}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(view, "SPACE=save  A=auto  Q=quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Capture training frames", view)

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {saved} images to {OUT}")


if __name__ == "__main__":
    main()
