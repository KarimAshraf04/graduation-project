"""
Train (fine-tune) a bag detector on Open Images bags + finaldata (Roboflow) + your own labeled images.
Result is copied to C:\\vision\\models\\bags_best.pt and vision_pipeline.py uses it automatically.

Usage (from C:\\vision):
  venv\\Scripts\\python training\\train.py                       # defaults: yolo26s, 80 epochs, 640px, batch 8
  venv\\Scripts\\python training\\train.py --epochs 120 --model yolo26m.pt --batch 4
  venv\\Scripts\\python training\\train.py --mine-only            # only your own images
"""
import argparse
import hashlib
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / "data"
OI = DATA / "openimages"
MINE_IMG, MINE_LBL = DATA / "mine" / "images", DATA / "mine" / "labels"
MINE_SPLIT = DATA / "mine_split"
NAMES = ["backpack", "handbag", "suitcase"]


def split_mine(val_frac):
    """Copy YOUR labeled images into mine_split/{images,labels}/{train,val} (stable split by file name)."""
    if MINE_SPLIT.exists():
        shutil.rmtree(MINE_SPLIT)
    n = {"train": 0, "val": 0}
    if not MINE_IMG.exists():
        return n
    for img in sorted(MINE_IMG.iterdir()):
        lbl = MINE_LBL / f"{img.stem}.txt"
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png") or not lbl.exists():
            continue   # only reviewed / labeled images
        h = int(hashlib.md5(img.stem.encode()).hexdigest(), 16) % 1000
        sp = "val" if h < val_frac * 1000 else "train"
        for sub, src in (("images", img), ("labels", lbl)):
            d = MINE_SPLIT / sub / sp
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, d / src.name)
        n[sp] += 1
    return n


FD = DATA / "finaldata"
FD_REMAP = {0: 0, 1: 1, 3: 2}   # finaldata: Bagpack, Handbag, Person, Suitcase -> drop Person


def prepare_finaldata(split, out_name):
    """3-class copy of a finaldata split. Only images that still have bag boxes (unlabeled images are skipped,
    because in this dataset 'no label' often means 'nobody labeled it', not 'no bag')."""
    src = FD / split
    if not (src / "images").exists():
        return 0
    dst = DATA / out_name
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "images").mkdir(parents=True)
    (dst / "labels").mkdir(parents=True)
    n = 0
    for img in (src / "images").iterdir():
        lbl = src / "labels" / f"{img.stem}.txt"
        if not lbl.exists():
            continue
        out = []
        for line in lbl.read_text().splitlines():
            p = line.split()
            if p and int(p[0]) in FD_REMAP:
                out.append(" ".join([str(FD_REMAP[int(p[0])])] + p[1:]))
        if out:
            shutil.copy2(img, dst / "images" / img.name)
            (dst / "labels" / f"{img.stem}.txt").write_text("\n".join(out))
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26s.pt", help="starting weights (in C:\\vision)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8, help="8 fits a 6GB GPU with yolo26s; use 4 for yolo26m")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--mine-only", action="store_true")
    ap.add_argument("--no-finaldata", action="store_true", help="don't use training/data/finaldata")
    ap.add_argument("--repeat-oi", type=int, default=1,
                    help="show Open Images train N times per epoch (balances it against bigger datasets)")
    ap.add_argument("--repeat-mine", type=int, default=3,
                    help="show your own camera images N times per epoch (they matter most)")
    ap.add_argument("--name", default="bags")
    a = ap.parse_args()

    mine = split_mine(a.val_frac)
    print(f"Your images: {mine['train']} train / {mine['val']} val")

    train_dirs, val_dirs = [], []
    if not a.mine_only and (OI / "images" / "train").exists():
        train_dirs += [(OI / "images" / "train").as_posix()] * max(1, a.repeat_oi)
        val_dirs.append((OI / "images" / "val").as_posix())
        print(f"Open Images: {len(list((OI / 'images' / 'train').glob('*')))} train / "
              f"{len(list((OI / 'images' / 'val').glob('*')))} val")
    if not a.mine_only and not a.no_finaldata:
        n_tr = prepare_finaldata("train", "finaldata_train") + 0
        n_te = prepare_finaldata("test", "finaldata_test")
        n_va = prepare_finaldata("valid", "finaldata_eval")
        if n_tr:
            train_dirs.append((DATA / "finaldata_train" / "images").as_posix())
        if n_te:
            train_dirs.append((DATA / "finaldata_test" / "images").as_posix())
        if n_va:
            val_dirs.append((DATA / "finaldata_eval" / "images").as_posix())
        if n_tr or n_va:
            print(f"finaldata: {n_tr + n_te} train / {n_va} val (labeled images only)")
    if mine["train"]:
        train_dirs += [(MINE_SPLIT / "images" / "train").as_posix()] * max(1, a.repeat_mine)
    if mine["val"]:
        val_dirs.append((MINE_SPLIT / "images" / "val").as_posix())
    if not train_dirs or not val_dirs:
        print("Not enough data. Run download_openimages.py and/or capture_frames.py + review_labels.py first.")
        return

    yaml_path = DATA / "bags.yaml"
    lines = ["train:"] + [f"  - {d}" for d in train_dirs] + ["val:"] + [f"  - {d}" for d in val_dirs]
    lines += ["names:"] + [f"  {i}: {n}" for i, n in enumerate(NAMES)]
    yaml_path.write_text("\n".join(lines) + "\n")
    print(f"Dataset config: {yaml_path}")

    import torch
    from ultralytics import YOLO
    device = 0 if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU found - training on CPU will be VERY slow.")
    start = ROOT / a.model
    model = YOLO(str(start) if start.exists() else a.model)

    model.train(
        data=str(yaml_path), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=device,
        workers=2,                       # Windows-friendly
        project=str(HERE / "runs"), name=a.name, exist_ok=False,
        patience=30,                     # stop early if no improvement for 30 epochs
        close_mosaic=10, cos_lr=True,
        fliplr=0.5, hsv_h=0.015, hsv_s=0.6, hsv_v=0.4, scale=0.5, degrees=5,
        plots=True)

    best = Path(model.trainer.best)
    if best.exists():
        dst = ROOT / "models" / "bags_best.pt"
        dst.parent.mkdir(exist_ok=True)
        shutil.copy2(best, dst)
        print(f"\nBest model: {best}\nCopied to:  {dst}  (vision_pipeline.py will use it automatically)")
        m = YOLO(str(dst)).val(data=str(yaml_path), imgsz=a.imgsz, device=device, plots=False, verbose=False,
                               project=str(HERE / "runs"), name="final_val", exist_ok=True)
        print(f"Validation mAP50: {m.box.map50:.3f}   mAP50-95: {m.box.map:.3f}")
        for i, n in enumerate(NAMES):
            try:
                print(f"  {n:9s} AP50: {m.box.ap50[i]:.3f}")
            except Exception:
                pass


if __name__ == "__main__":
    main()
