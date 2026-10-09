"""
Download bag images (Backpack / Handbag / Suitcase) with boxes from Open Images V7
and export them in YOLO format to training/data/openimages.

Usage (from C:\\vision):
  venv\\Scripts\\python training\\download_openimages.py                 # validation + test splits (~fast)
  venv\\Scripts\\python training\\download_openimages.py --max 500        # limit images per split
  venv\\Scripts\\python training\\download_openimages.py --use-train      # also the big train split (more images, slow first time)

Needs:  venv\\Scripts\\python -m pip install fiftyone
"""
import argparse
import shutil
from pathlib import Path

CLASSES = ["Backpack", "Handbag", "Suitcase"]   # order = class ids 0, 1, 2
HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "openimages"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=None, help="max images per split")
    ap.add_argument("--use-train", action="store_true", help="also download the train split")
    ap.add_argument("--val-frac", type=float, default=0.1)
    a = ap.parse_args()

    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
    except ImportError:
        print("fiftyone is not installed. Run:\n  venv\\Scripts\\python -m pip install fiftyone")
        return

    splits = ["validation", "test"] + (["train"] if a.use_train else [])
    merged = None
    for sp in splits:
        print(f"\n==> Downloading Open Images '{sp}' split for {CLASSES} ...")
        ds = foz.load_zoo_dataset(
            "open-images-v7", split=sp, label_types=["detections"], classes=CLASSES,
            max_samples=a.max, only_matching=True, shuffle=True, seed=51,
            dataset_name=f"oi-bags-{sp}", overwrite=True)
        print(f"    {len(ds)} images")
        if merged is None:
            merged = ds.clone("oi-bags-all", persistent=False)
        else:
            merged.merge_samples(ds)

    # شيل المربعات اللي بتغطي "مجموعة شنط" مرة واحدة (IsGroupOf) والرسومات/الصور المطبوعة (IsDepiction):
    # دي بتعلّم الموديل غلط (مثلاً مربع واحد على حيطة محل كلها شنط)
    from fiftyone import ViewField as F
    before = merged.count("ground_truth.detections")
    merged = merged.filter_labels(
        "ground_truth", (F("IsGroupOf") != True) & (F("IsDepiction") != True), only_matches=True)
    print(f"Removed group/depiction boxes: {before} -> {merged.count('ground_truth.detections')} boxes")

    n = len(merged)
    if n == 0:
        print("No images downloaded.")
        return
    val = merged.take(max(1, int(n * a.val_frac)), seed=51)
    train = merged.exclude(val.values("id"))

    if OUT.exists():
        shutil.rmtree(OUT)
    for view, split in ((train, "train"), (val, "val")):
        view.export(export_dir=str(OUT), dataset_type=fo.types.YOLOv5Dataset,
                    label_field="ground_truth", classes=CLASSES, split=split)
    print(f"\nDone. {len(train)} train + {len(val)} val images -> {OUT}")

    # quick per-class box count
    counts = {c: 0 for c in CLASSES}
    for lab in (OUT / "labels").rglob("*.txt"):
        for line in lab.read_text().splitlines():
            if line.strip():
                counts[CLASSES[int(line.split()[0])]] += 1
    print("Boxes per class:", counts)


if __name__ == "__main__":
    main()
