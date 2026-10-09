"""
Simple label review / correction tool (no extra install needed).
Opens training/data/mine/images one by one with its boxes from training/data/mine/labels.

Usage (from C:\\vision):
  venv\\Scripts\\python training\\review_labels.py
  venv\\Scripts\\python training\\review_labels.py --start 120     # jump to image #120

MOUSE
  left-drag on empty area ....... draw a new box (with the current class)
  left-click on a box ........... select it
  right-click on a box .......... delete it
KEYS
  1 / 2 / 3 ..................... class: 1=backpack 2=handbag 3=suitcase (changes the selected box too)
  D or Right arrow / SPACE ...... save + next image
  A or Left arrow ............... save + previous image
  X or Delete ................... delete selected box
  C ............................. clear all boxes (image has NO bag = useful negative example)
  R ............................. reject image (moved to data/mine/rejected, not used for training)
  Q or Esc ...................... save + quit (progress is remembered)
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
IMG_DIR = HERE / "data" / "mine" / "images"
LBL_DIR = HERE / "data" / "mine" / "labels"
REJ_DIR = HERE / "data" / "mine" / "rejected"
PROGRESS = HERE / "data" / "mine" / "review_progress.txt"

NAMES = ["backpack", "handbag", "suitcase"]
COLORS = [(0, 165, 255), (255, 0, 255), (0, 255, 0)]
MAX_W, MAX_H = 1280, 760
KEY_LEFT, KEY_RIGHT, KEY_DEL = (2424832, 65361), (2555904, 65363), (3014656, 65535)


class State:
    boxes = []        # [cls, x1, y1, x2, y2] in original pixels
    sel = -1
    cur_cls = 0
    drag = None       # (x, y) start in original pixels
    mouse = (0, 0)
    scale = 1.0
    dirty = False


S = State()


def load_boxes(lbl_path, w, h):
    boxes = []
    if lbl_path.exists():
        for line in lbl_path.read_text().splitlines():
            p = line.split()
            if len(p) == 5:
                c, x, y, bw, bh = int(p[0]), *map(float, p[1:])
                boxes.append([c, (x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h])
    return boxes


def save_boxes(lbl_path, boxes, w, h):
    lines = []
    for c, x1, y1, x2, y2 in boxes:
        x1, x2 = sorted((max(0, x1), min(w, x2)))
        y1, y2 = sorted((max(0, y1), min(h, y2)))
        if x2 - x1 < 3 or y2 - y1 < 3:
            continue
        lines.append(f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    lbl_path.parent.mkdir(parents=True, exist_ok=True)
    lbl_path.write_text("\n".join(lines))


def box_at(x, y):
    hits = [i for i, (_, x1, y1, x2, y2) in enumerate(S.boxes) if x1 <= x <= x2 and y1 <= y <= y2]
    if not hits:
        return -1
    return min(hits, key=lambda i: (S.boxes[i][3] - S.boxes[i][1]) * (S.boxes[i][4] - S.boxes[i][2]))


def on_mouse(ev, x, y, flags, _):
    ox, oy = x / S.scale, y / S.scale
    S.mouse = (ox, oy)
    if ev == cv2.EVENT_LBUTTONDOWN:
        i = box_at(ox, oy)
        if i >= 0:
            S.sel = i
        else:
            S.drag = (ox, oy)
    elif ev == cv2.EVENT_LBUTTONUP and S.drag is not None:
        x0, y0 = S.drag
        S.drag = None
        if abs(ox - x0) > 4 and abs(oy - y0) > 4:
            S.boxes.append([S.cur_cls, min(x0, ox), min(y0, oy), max(x0, ox), max(y0, oy)])
            S.sel = len(S.boxes) - 1
            S.dirty = True
    elif ev == cv2.EVENT_RBUTTONDOWN:
        i = box_at(ox, oy)
        if i >= 0:
            S.boxes.pop(i)
            S.sel = -1
            S.dirty = True


def render(img, idx, total, name):
    view = cv2.resize(img, None, fx=S.scale, fy=S.scale) if S.scale != 1 else img.copy()
    for i, (c, x1, y1, x2, y2) in enumerate(S.boxes):
        p1 = (int(x1 * S.scale), int(y1 * S.scale))
        p2 = (int(x2 * S.scale), int(y2 * S.scale))
        cv2.rectangle(view, p1, p2, COLORS[c], 3 if i == S.sel else 2)
        cv2.putText(view, NAMES[c], (p1[0], max(15, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[c], 2)
    if S.drag is not None:
        x0, y0 = S.drag
        cv2.rectangle(view, (int(x0 * S.scale), int(y0 * S.scale)),
                      (int(S.mouse[0] * S.scale), int(S.mouse[1] * S.scale)), COLORS[S.cur_cls], 1)
    bar = np.full((56, view.shape[1], 3), 30, np.uint8)
    cv2.putText(bar, f"[{idx + 1}/{total}] {name}   boxes: {len(S.boxes)}   new-box class: "
                     f"{S.cur_cls + 1}={NAMES[S.cur_cls]}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(bar, "drag=new  rclick=del  1/2/3=class  D=next A=prev  X=del  C=clear  R=reject  Q=quit",
                (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    return np.vstack([view, bar])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=None, help="1-based image number to start from")
    a = ap.parse_args()

    imgs = sorted(p for p in IMG_DIR.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")) \
        if IMG_DIR.exists() else []
    if not imgs:
        print(f"No images in {IMG_DIR}. Run capture_frames.py first.")
        return
    idx = 0
    if a.start:
        idx = a.start - 1
    elif PROGRESS.exists():
        try:
            idx = int(PROGRESS.read_text().strip())
        except ValueError:
            idx = 0
    idx = max(0, min(idx, len(imgs) - 1))

    win = "Review labels"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)

    while 0 <= idx < len(imgs):
        path = imgs[idx]
        img = cv2.imread(str(path))
        if img is None:
            idx += 1
            continue
        h, w = img.shape[:2]
        S.scale = min(1.0, MAX_W / w, MAX_H / h)
        lbl = LBL_DIR / f"{path.stem}.txt"
        S.boxes, S.sel, S.dirty = load_boxes(lbl, w, h), -1, False
        if not lbl.exists():
            S.dirty = True   # first visit of an unlabeled image -> will be saved (maybe empty)

        step = 0
        while step == 0:
            cv2.imshow(win, render(img, idx, len(imgs), path.name))
            k = cv2.waitKeyEx(20)
            if k == -1:
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    step = 99
                continue
            ch = k & 0xFF
            if ch in (ord("1"), ord("2"), ord("3")):
                S.cur_cls = ch - ord("1")
                if 0 <= S.sel < len(S.boxes):
                    S.boxes[S.sel][0] = S.cur_cls
                    S.dirty = True
            elif k in KEY_RIGHT or ch in (ord("d"), ord("D"), ord(" ")):
                step = 1
            elif k in KEY_LEFT or ch in (ord("a"), ord("A")):
                step = -1
            elif k in KEY_DEL or ch in (ord("x"), ord("X")):
                if 0 <= S.sel < len(S.boxes):
                    S.boxes.pop(S.sel)
                    S.sel = -1
                    S.dirty = True
            elif ch in (ord("c"), ord("C")):
                S.boxes, S.sel, S.dirty = [], -1, True
            elif ch in (ord("r"), ord("R")):
                REJ_DIR.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(REJ_DIR / path.name))
                if lbl.exists():
                    lbl.unlink()
                imgs.pop(idx)
                S.dirty = False
                step = 2
            elif ch in (ord("q"), ord("Q"), 27):
                step = 99

        if S.dirty:
            save_boxes(lbl, S.boxes, w, h)
        if step == 99:
            break
        if step in (1, -1):
            idx = max(0, idx + step) if step == -1 else idx + 1
        if idx >= len(imgs):
            print("Reached the last image.")
            idx = len(imgs) - 1
            break

    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text(str(max(0, idx)))
    cv2.destroyAllWindows()
    labeled = sum(1 for p in imgs if (LBL_DIR / f"{p.stem}.txt").exists())
    print(f"Saved. {labeled}/{len(imgs)} images have labels. Next: venv\\Scripts\\python training\\train.py")


if __name__ == "__main__":
    main()
