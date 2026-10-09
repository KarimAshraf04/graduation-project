"""
Pre-label your images automatically, so you only FIX mistakes instead of drawing every box.
Reads training/data/mine/images, writes YOLO labels to training/data/mine/labels.
Classes: 0 = backpack, 1 = handbag, 2 = suitcase

Usage (from C:\\vision):
  venv\\Scripts\\python training\\auto_label.py                          # uses yolo26m.pt (COCO)
  venv\\Scripts\\python training\\auto_label.py --model models\\bags_best.pt   # after a first training: better pre-labels
  venv\\Scripts\\python training\\auto_label.py --overwrite              # redo images that already have labels
"""
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
IMG_DIR = HERE / "data" / "mine" / "images"
LBL_DIR = HERE / "data" / "mine" / "labels"
COCO_TO_OURS = {24: 0, 26: 1, 28: 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "yolo26m.pt"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    from ultralytics import YOLO
    import torch
    model = YOLO(a.model)
    custom = len(model.names) == 3          # our trained model -> ids already 0/1/2
    device = 0 if torch.cuda.is_available() else "cpu"

    LBL_DIR.mkdir(parents=True, exist_ok=True)
    imgs = sorted(p for p in IMG_DIR.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    todo = [p for p in imgs if a.overwrite or not (LBL_DIR / f"{p.stem}.txt").exists()]
    print(f"{len(imgs)} images, {len(todo)} to label with {a.model}")

    n_boxes = 0
    for i in range(0, len(todo), 16):
        batch = todo[i:i + 16]
        kwargs = dict(conf=a.conf, imgsz=a.imgsz, device=device, verbose=False)
        if not custom:
            kwargs["classes"] = sorted(COCO_TO_OURS)
        for path, r in zip(batch, model.predict([str(p) for p in batch], **kwargs)):
            lines = []
            for (x, y, w, h), c in zip(r.boxes.xywhn.cpu().numpy(), r.boxes.cls.int().cpu().tolist()):
                cid = c if custom else COCO_TO_OURS.get(c)
                if cid is not None:
                    lines.append(f"{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            (LBL_DIR / f"{path.stem}.txt").write_text("\n".join(lines))
            n_boxes += len(lines)
        print(f"  {min(i + 16, len(todo))}/{len(todo)}")
    print(f"Done: {n_boxes} boxes written to {LBL_DIR}")
    print("Next: review and fix them ->  venv\\Scripts\\python training\\review_labels.py")


if __name__ == "__main__":
    main()
