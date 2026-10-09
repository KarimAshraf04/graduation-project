"""
Compare bag models on the SAME test images (fair comparison).
All models must have 3 classes in this order: 0=backpack, 1=handbag, 2=suitcase.

Usage (from C:\\vision):
  venv\\Scripts\\python training\\compare_models.py --models models\\bags_best.pt models\\bag_detector_v2.pt
  venv\\Scripts\\python training\\compare_models.py --models a.pt b.pt --imgsz 640

Test sets used (whatever exists):
  - Open Images val          training/data/openimages/images/val
  - finaldata (Roboflow) valid  training/data/finaldata/valid/images   (unzip finaldata.v1i.yolov8.zip there)
  - your own images val      training/data/mine_split/images/val
"""
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / "data"
NAMES = ["backpack", "handbag", "suitcase"]


def prepare_finaldata():
    """finaldata has 4 classes (Bagpack, Handbag, Person, Suitcase): make a 3-class copy without Person."""
    src = DATA / "finaldata" / "valid"
    if not (src / "images").exists():
        return None
    dst = DATA / "finaldata_eval"
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)
    remap = {0: 0, 1: 1, 3: 2}            # drop 2 = Person
    import shutil
    n = 0
    for img in (src / "images").iterdir():
        lbl = src / "labels" / f"{img.stem}.txt"
        if not lbl.exists() or lbl.stat().st_size == 0:
            continue                          # keep human-labeled images only
        out = []
        for line in lbl.read_text().splitlines():
            p = line.split()
            if p and int(p[0]) in remap:
                out.append(" ".join([str(remap[int(p[0])])] + p[1:]))
        if not out:
            continue
        shutil.copy2(img, dst / "images" / img.name)
        (dst / "labels" / f"{img.stem}.txt").write_text("\n".join(out))
        n += 1
    print(f"finaldata eval set: {n} human-labeled images")
    return dst / "images"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()

    sets = {}
    if (DATA / "openimages" / "images" / "val").exists():
        sets["OpenImages val"] = DATA / "openimages" / "images" / "val"
    fd = prepare_finaldata()
    if fd:
        sets["finaldata valid"] = fd
    if (DATA / "mine_split" / "images" / "val").exists():
        sets["your camera val"] = DATA / "mine_split" / "images" / "val"
    if not sets:
        print("No test sets found.")
        return

    import torch
    from ultralytics import YOLO
    device = 0 if torch.cuda.is_available() else "cpu"
    results = {}
    for set_name, img_dir in sets.items():
        yml = DATA / f"_eval_{set_name.split()[0].lower()}.yaml"
        yml.write_text(f"train: {img_dir.as_posix()}\nval: {img_dir.as_posix()}\n"
                       "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))
        for mp in a.models:
            mpath = Path(mp) if Path(mp).is_absolute() else ROOT / mp
            m = YOLO(str(mpath))
            if len(m.names) != 3:
                print(f"skip {mp}: has {len(m.names)} classes, need 3")
                continue
            r = m.val(data=str(yml), imgsz=a.imgsz, batch=4, device=device, plots=False, verbose=False,
                      project=str(HERE / "runs"), name="compare", exist_ok=True)
            results[(set_name, mpath.name)] = (r.box.map50, r.box.map, list(r.box.ap50))

    print("\n" + "=" * 70)
    print(f"{'test set':18s} {'model':28s} {'mAP50':>7s} {'mAP50-95':>9s}   per-class AP50")
    for (s, mname), (m50, m, per) in results.items():
        pc = " ".join(f"{x:.2f}" for x in per)
        print(f"{s:18s} {mname:28s} {m50:7.3f} {m:9.3f}   {pc}")
    print("=" * 70)
    print("Note: a model scores higher on the test set that matches the data it was trained on.")
    print("The set that matters most is 'your camera val'.")


if __name__ == "__main__":
    main()
