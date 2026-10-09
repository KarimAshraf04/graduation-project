"""
Vision pipeline: people count, face details, backpack size, held objects
(MediaPipe finger landmarks), and recording to results/ (video + CSV).

Run:
  python vision_pipeline.py --source 0              # webcam
  python vision_pipeline.py --source video.mp4      # video file
  python vision_pipeline.py --source 0 --no-video   # don't save video
  python vision_pipeline.py --source 0 --no-log     # don't save CSV
"""
import argparse
import csv
import os
import threading
import time
import urllib.request
from datetime import datetime

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ---------------- إعدادات السرعة ----------------
PERSON_MODEL = "yolo26n-pose.pt"
OBJECT_MODEL = "yolo26s.pt"
IMGSZ = 480
OBJECT_EVERY_N = 2
FACE_EVERY_N = 4
HAND_EVERY_N = 2
FACE_DET_SIZE = (320, 320)
CAM_W, CAM_H = 640, 480
MAX_HANDS = 6
OBJ_CONF = 0.35             # ثقة الحاجات العادية
BAG_CONF = 0.25             # ثقة أقل للشنط عشان متفوتش (بتتأكد بالتثبيت بالوقت)
BAG_ZOOM = True             # كشف تاني للشنط على صورة كل شخص مكبّرة (بيلقط الشنط الصغيرة/البعيدة)
BAG_ZOOM_MAX_PERSONS = 3    # أقصى عدد أشخاص يتعملهم zoom في الفريم (عشان السرعة)

DEVICE = 0 if torch.cuda.is_available() else "cpu"
HALF = DEVICE != "cpu"

# ---------------- إعدادات المنطق ----------------
AVG_PERSON_CM = 170
PANEL_W = 320
EXIT_AFTER_S = 2.0          # الشخص يعتبر خرج لو اختفى المدة دي
MIN_TRACK_S = 1.0           # الشخص لازم يفضل ظاهر ثانية عشان يتعد (بيمنع العد الغلط)
TRACK_BUFFER = 90           # عدد الفريمات اللي الـ tracker يفتكر فيها الشخص لو اختفى
HELD_MIN_POINTS = 4         # أقل عدد نقط إيد (من 21) جوه الحاجة عشان تعتبر ماسكها
HOLD_ON_S = 0.3             # لازم تبان ماسكها المدة دي عشان تتأكد
HOLD_OFF_S = 1.5            # لازم تختفي المدة دي عشان تعتبر سبتها (بيمنع الرعشة)

HAND_MODEL_PATH = os.path.join("models", "hand_landmarker.task")
# موديل الشنط المتدرب (training\train.py بيحطه هنا). لو موجود، بيتستخدم للشنط بدل YOLO العادي
BAG_MODEL_PATH = os.path.join("models", "bags_best.pt")
BAG_IMGSZ = 640                   # حجم الصورة لموديل الشنط المتدرب (أكبر = أدق للشنط الصغيرة، أبطأ)
BAG_MAP = {0: 24, 1: 26, 2: 28}   # backpack / handbag / suitcase -> أرقام COCO
HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                  "hand_landmarker/float16/latest/hand_landmarker.task")

BACKPACK, HANDBAG, SUITCASE = 24, 26, 28
BAG_CLASSES = {BACKPACK, HANDBAG, SUITCASE}
L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 7, 8, 9, 10

# حاجات مستحيل تبقى "في الإيد" (عشان الإيد على الترابيزة متتحسبش)
NOT_HOLDABLE = {"bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
                "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
                "bench", "chair", "couch", "potted plant", "bed", "dining table",
                "toilet", "tv", "refrigerator", "oven", "sink", "microwave",
                "elephant", "horse", "cow", "sheep", "bear", "giraffe", "zebra"}

FINGER_TIPS = [4, 8, 12, 16, 20]
HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
                    (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


# ======================= الكاميرا =======================
class LatestFrameReader:
    """بيقرا الكاميرا في thread لوحده ويحتفظ بآخر فريم بس."""

    def __init__(self, source):
        if source.isdigit():
            self.cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.live = True
        else:
            self.cap = cv2.VideoCapture(source)
            self.live = False
        self.src_fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        self.frame, self.ok, self.stopped = None, True, False
        if self.live:
            threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while not self.stopped:
            ok, f = self.cap.read()
            self.ok = ok
            if ok:
                self.frame = f

    def read(self):
        if not self.live:
            return self.cap.read()
        while self.frame is None and self.ok:
            time.sleep(0.005)
        return self.ok, None if self.frame is None else self.frame.copy()

    def release(self):
        self.stopped = True
        self.cap.release()


# ======================= تحميل الموديلات =======================
def load_face_app():
    try:
        import onnxruntime as ort
        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception:
                pass
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l",
                           allowed_modules=["detection", "landmark_2d_106",
                                            "landmark_3d_68", "genderage"],
                           providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=FACE_DET_SIZE)
        prov = app.models["detection"].session.get_providers()[0]
        print(f"[i] Face models running on: {prov}")
        return app
    except Exception as e:
        print(f"[!] InsightFace not available, faces disabled: {e}")
        return None


def load_hand_landmarker():
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision as mpv
    except Exception as e:
        print(f"[!] MediaPipe not installed ({e}). Using wrist estimate instead.")
        print("    Install: venv\\Scripts\\python -m pip install mediapipe")
        return None, None
    try:
        if not os.path.exists(HAND_MODEL_PATH):
            os.makedirs(os.path.dirname(HAND_MODEL_PATH), exist_ok=True)
            print("[i] Downloading MediaPipe hand model...")
            urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
        opts = mpv.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
            running_mode=mpv.RunningMode.VIDEO,
            num_hands=MAX_HANDS,
            min_hand_detection_confidence=0.4,
            min_hand_presence_confidence=0.4,
            min_tracking_confidence=0.4)
        lm = mpv.HandLandmarker.create_from_options(opts)
        print("[i] MediaPipe Hands ready (finger landmarks)")
        return lm, mp
    except Exception as e:
        print(f"[!] MediaPipe Hands failed ({e}). Using wrist estimate instead.")
        return None, None


# ======================= دوال مساعدة =======================
def wrist_estimates(kpts, kconf, thr=0.4):
    """تقدير مكان الكف من الرسغ والكوع (fallback لو MediaPipe مشافش الإيد)."""
    pts = []
    for wrist, elbow in ((L_WRIST, L_ELBOW), (R_WRIST, R_ELBOW)):
        if kconf[wrist] < thr:
            continue
        w = kpts[wrist]
        if kconf[elbow] >= thr:
            w = w + 0.3 * (w - kpts[elbow])
        pts.append(w)
    return pts


def overlap_ratio(inner, outer):
    x1, y1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    x2, y2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = max(1, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / area


def expand(box, frac):
    p = frac * max(box[2] - box[0], box[3] - box[1])
    return (box[0] - p, box[1] - p, box[2] + p, box[3] + p)


def points_inside(pts, box):
    return int(np.sum((pts[:, 0] >= box[0]) & (pts[:, 0] <= box[2]) &
                      (pts[:, 1] >= box[1]) & (pts[:, 1] <= box[3])))


def bag_size(bag_box, person_box):
    ratio = (bag_box[3] - bag_box[1]) / max(1, person_box[3] - person_box[1])
    cm = ratio * AVG_PERSON_CM
    label = "Small" if ratio < 0.25 else "Medium" if ratio < 0.38 else "Large"
    return label, cm


def is_holding(person, obj_box):
    """True لو الإيد ماسكة الحاجة. بيستخدم 21 نقطة لو موجودة، وإلا تقدير الرسغ."""
    box = expand(obj_box, 0.10)
    for hand in person["mp_hands"]:
        if points_inside(hand, box) >= HELD_MIN_POINTS and points_inside(hand[FINGER_TIPS], box) >= 1:
            return True
    if not person["mp_hands"]:
        box2 = expand(obj_box, 0.15)
        for p in person["wrist_pts"]:
            if box2[0] <= p[0] <= box2[2] and box2[1] <= p[1] <= box2[3]:
                return True
    return False


def debounce(state, present, t):
    """تثبيت بالوقت: الحاجة تتأكد بعد HOLD_ON_S، وتتشال بس لو غابت HOLD_OFF_S.
    بيشتغل صح مهما كان الـ FPS، وبيمنع start/stop كل شوية."""
    for k in present:
        st = state.setdefault(k, {"first": t, "last": t, "on": False})
        st["last"] = t
        if not st["on"] and t - st["first"] >= HOLD_ON_S:
            st["on"] = True
    for k in list(state):
        st = state[k]
        gone = t - st["last"]
        if (st["on"] and gone > HOLD_OFF_S) or (not st["on"] and gone > 0.5):
            del state[k]
    return {k for k, st in state.items() if st["on"]}


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def make_tracker_cfg():
    """نسخة من bytetrack بذاكرة أطول، عشان نفس الشخص ميخدش ID جديد كل ما يتداري."""
    try:
        from ultralytics.utils import ROOT
        import yaml
        with open(ROOT / "cfg" / "trackers" / "bytetrack.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg["track_buffer"] = TRACK_BUFFER
        path = os.path.join("models", "tracker.yaml")
        os.makedirs("models", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f)
        return path
    except Exception as e:
        print(f"[!] Custom tracker config failed ({e}), using default bytetrack.")
        return "bytetrack.yaml"


# ======================= التسجيل =======================
class Recorder:
    def __init__(self, save_video, save_log, save_crops=True):
        self.dir = os.path.join("results", datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.dir, exist_ok=True)
        self.save_video, self.save_log, self.save_crops = save_video, save_log, save_crops
        self.crop_best = {}   # (kind, pid, key) -> (score, last_write_time)
        if save_crops:
            os.makedirs(os.path.join(self.dir, "faces"), exist_ok=True)
            os.makedirs(os.path.join(self.dir, "bags"), exist_ok=True)
        self.writer, self.events_f = None, None
        self.people = {}  # pid -> summary
        self.objects = {}  # class -> {"max": n, "frames": n, "first": t, "last": t}
        if save_log:
            self.events_f = open(os.path.join(self.dir, "events.csv"), "w",
                                 newline="", encoding="utf-8-sig")
            self.events = csv.writer(self.events_f)
            self.events.writerow(["clock_time", "video_sec", "person_id", "event", "detail"])
        print(f"[i] Saving results to: {os.path.abspath(self.dir)}")

    def event(self, t_wall, t_vid, pid, ev, detail=""):
        if self.events_f:
            self.events.writerow([datetime.fromtimestamp(t_wall).strftime("%H:%M:%S"),
                                  f"{t_vid:.1f}", pid, ev, detail])
            self.events_f.flush()

    def frame(self, img, fps):
        if not self.save_video:
            return
        if self.writer is None:
            h, w = img.shape[:2]
            self.writer = cv2.VideoWriter(os.path.join(self.dir, "video.mp4"),
                                          cv2.VideoWriter_fourcc(*"mp4v"),
                                          max(5.0, min(30.0, fps)), (w, h))
        self.writer.write(img)

    def save_crop(self, kind, pid, key, img, box, score, pad=0.2):
        """بيقص الصورة ويحفظها، وبيبدّلها بس لو لقى صورة أوضح لنفس الشخص (أكبر وأوضح وأقرب للمواجهة)."""
        if not self.save_crops or pid not in self.people:
            return
        k = (kind, pid, key)
        old_score, last_t = self.crop_best.get(k, (0.0, 0.0))
        now = time.time()
        if score <= old_score * 1.1 or now - last_t < 0.5:
            return
        H, W = img.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        pw, ph = (x2 - x1) * pad, (y2 - y1) * pad
        x1, y1 = max(0, int(x1 - pw)), max(0, int(y1 - ph))
        x2, y2 = min(W, int(x2 + pw)), min(H, int(y2 + ph))
        if x2 - x1 < 24 or y2 - y1 < 24:
            return
        name = f"person_{pid}.jpg" if kind == "faces" else f"person_{pid}_{key.replace(' ', '_')}.jpg"
        path = os.path.join(self.dir, kind, name)
        cv2.imwrite(path, img[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 95])
        self.crop_best[k] = (score, now)
        s = self.people[pid]
        if kind == "faces":
            s["face_img"] = os.path.join(kind, name)
        else:
            s.setdefault("bag_imgs", set()).add(os.path.join(kind, name))

    def update_objects(self, names_list, t_vid):
        counts = {}
        for n in names_list:
            counts[n] = counts.get(n, 0) + 1
        for n, c in counts.items():
            o = self.objects.setdefault(n, {"max": 0, "frames": 0, "first": t_vid, "last": t_vid})
            o["max"] = max(o["max"], c)
            o["frames"] += 1
            o["last"] = t_vid

    def write_objects(self):
        if not self.save_log:
            return
        with open(os.path.join(self.dir, "objects.csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["object", "max_in_one_frame", "detections", "first_seen_s", "last_seen_s"])
            for n, o in sorted(self.objects.items(), key=lambda x: -x[1]["frames"]):
                w.writerow([n, o["max"], o["frames"], f"{o['first']:.1f}", f"{o['last']:.1f}"])

    def write_people(self):
        if not self.save_log:
            return
        with open(os.path.join(self.dir, "people.csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["person_id", "first_seen", "last_seen", "duration_s",
                        "gender", "bags", "held_items", "face_image", "bag_images"])
            for pid, s in sorted(self.people.items()):
                g = s.get("genders", [])
                w.writerow([pid, f"{s['first']:.1f}", f"{s['last']:.1f}",
                            f"{s['last'] - s['first']:.1f}",
                            {"M": "Male", "F": "Female"}.get(max(set(g), key=g.count), "") if g else "",
                            "; ".join(sorted(s["bags"])), "; ".join(sorted(s["held"])),
                            s.get("face_img", ""), "; ".join(sorted(s.get("bag_imgs", [])))])

    def close(self):
        if self.writer:
            self.writer.release()
        if self.events_f:
            self.events_f.close()
        self.write_people()
        self.write_objects()
        print(f"[i] Results saved in: {os.path.abspath(self.dir)}")


# ======================= البرنامج =======================
def main(source, save_video, save_log, save_crops=True, coco_bags=False, bag_model_path=None):
    print(f"[i] YOLO device: {'GPU ' + torch.cuda.get_device_name(0) if HALF else 'CPU (slow!)'}")
    person_model = YOLO(PERSON_MODEL)
    object_model = YOLO(OBJECT_MODEL)
    names = object_model.names
    bag_model = None
    bag_path = bag_model_path or BAG_MODEL_PATH
    if os.path.exists(bag_path) and not coco_bags:
        bag_model = YOLO(bag_path)
        print(f"[i] Custom bag model loaded: {bag_path}  classes={bag_model.names}")

    def detect_bags(img, imgsz, offset=(0, 0)):
        """بيرجع الشنط [(box, coco_cls, conf)] من الموديل المتدرب لو موجود، وإلا من YOLO العادي."""
        if bag_model is not None:
            r = bag_model.predict(img, conf=BAG_CONF, imgsz=imgsz, device=DEVICE, half=HALF, verbose=False)[0]
            cmap = BAG_MAP
        else:
            r = object_model.predict(img, conf=BAG_CONF, imgsz=imgsz, classes=sorted(BAG_CLASSES),
                                     device=DEVICE, half=HALF, verbose=False)[0]
            cmap = {c: c for c in BAG_CLASSES}
        out = []
        off = np.array([offset[0], offset[1], offset[0], offset[1]], dtype=np.float32)
        for bb, c, f in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.int().cpu().tolist(),
                            r.boxes.conf.cpu().tolist()):
            if c in cmap:
                out.append((bb.astype(np.float32) + off, cmap[c], f))
        return out
    face_app = load_face_app()
    hand_lm, mp = load_hand_landmarker()

    cap = LatestFrameReader(source)
    rec = (Recorder(save_video, save_log, save_crops)
           if (save_video or save_log or save_crops) else None)

    tracker_cfg = make_tracker_cfg()
    seen_ids, faces, objects, mp_hands_px = set(), [], [], []
    held_deb, bag_deb = {}, {}             # pid -> debounce state
    held_state, bag_state = {}, {}         # pid -> آخر حالة اتسجلت
    last_seen = {}                         # pid -> video time
    track_first = {}                       # pid -> أول ظهور (قبل ما يتعد)
    frame_i, fps, t_prev, last_ts = 0, 0.0, time.time(), 0
    t_start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frame_i += 1
        H, W = frame.shape[:2]
        raw = frame.copy()   # نسخة نضيفة من غير رسومات: منها بنقص صور الوش والشنط
        now = time.time()
        # وقت الفيديو: للكاميرا = الوقت من أول التشغيل، للفيديو = رقم الفريم / fps
        t_vid = (now - t_start) if cap.live else frame_i / cap.src_fps

        # ---- 1) الأشخاص + tracking ----
        pr = person_model.track(frame, persist=True, conf=0.4, imgsz=IMGSZ,
                                device=DEVICE, half=HALF,
                                tracker=tracker_cfg, verbose=False)[0]
        persons = []
        if pr.boxes is not None and len(pr.boxes):
            boxes = pr.boxes.xyxy.cpu().numpy()
            ids = pr.boxes.id.int().cpu().tolist() if pr.boxes.id is not None else [-1] * len(boxes)
            kxy = pr.keypoints.xy.cpu().numpy()
            kcf = pr.keypoints.conf.cpu().numpy() if pr.keypoints.conf is not None else np.ones(kxy.shape[:2])
            for b, tid, k, c in zip(boxes, ids, kxy, kcf):
                persons.append({"box": b, "id": tid,
                                "wrists": [k[i] for i in (L_WRIST, R_WRIST) if c[i] >= 0.4],
                                "wrist_pts": wrist_estimates(k, c),
                                "mp_hands": [], "held": set(), "bags": {}})

        # ---- MediaPipe Hands: 21 نقطة لكل إيد ----
        if hand_lm is not None and (frame_i % HAND_EVERY_N == 1 or HAND_EVERY_N == 1):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts = max(last_ts + 1, int(time.monotonic() * 1000))
            last_ts = ts
            res = hand_lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
            mp_hands_px = [np.array([[p.x * W, p.y * H] for p in hl], dtype=np.float32)
                           for hl in res.hand_landmarks]

        # ربط كل إيد بالشخص اللي رسغه (من YOLO) أقرب لرسغ الإيد (نقطة 0)
        # الإيد اللي ملهاش جسم ظاهر (دراع بس في الكادر) بتروح لـ "hand"
        hand_only = {"id": "hand", "mp_hands": [], "wrist_pts": [], "held": set()}
        for hand in mp_hands_px:
            best, best_d = None, 1e9
            for p in persons:
                ph = p["box"][3] - p["box"][1]
                for wpt in p["wrists"]:
                    d = float(np.linalg.norm(hand[0] - wpt))
                    if d < best_d and d < 0.2 * ph:
                        best, best_d = p, d
            if best is None:  # مفيش رسغ قريب: خد الشخص اللي الإيد جوه البوكس بتاعه
                hb = (*hand.min(0), *hand.max(0))
                cands = [p for p in persons if overlap_ratio(hb, p["box"]) > 0.6]
                best = cands[0] if cands else None
            if best is not None:
                best["mp_hands"].append(hand)
            else:
                hand_only["mp_hands"].append(hand)

        # ---- الأوبجكتس ----
        if frame_i % OBJECT_EVERY_N == 1 or OBJECT_EVERY_N == 1:
            orr = object_model.predict(raw, conf=BAG_CONF, imgsz=IMGSZ, device=DEVICE,
                                       half=HALF, verbose=False)[0]
            objects = [(b, c, f) for b, c, f in zip(orr.boxes.xyxy.cpu().numpy(),
                                                    orr.boxes.cls.int().cpu().tolist(),
                                                    orr.boxes.conf.cpu().tolist())
                       if c != 0 and (f >= OBJ_CONF or (c in BAG_CLASSES and bag_model is None))
                       and not (c in BAG_CLASSES and bag_model is not None)]
            if bag_model is not None:   # الشنط من الموديل المتدرب
                objects += detect_bags(raw, BAG_IMGSZ)

            # zoom على كل شخص مفيش عليه شنطة: نقص صورته ونكشف شنط بس عليها.
            # الشخص بيبقى أكبر في الصورة المقصوصة، فالشنطة الصغيرة/البعيدة بتبان أوضح.
            if BAG_ZOOM and persons:
                bag_boxes_now = [b for b, c, _ in objects if c in BAG_CLASSES]
                need = [p for p in persons
                        if (p["box"][3] - p["box"][1]) > 80
                        and not any(overlap_ratio(bb, p["box"]) > 0.3 for bb in bag_boxes_now)]
                need.sort(key=lambda p: -(p["box"][3] - p["box"][1]))
                for p in need[:BAG_ZOOM_MAX_PERSONS]:
                    ex = expand(p["box"], 0.08)
                    cx1, cy1 = max(0, int(ex[0])), max(0, int(ex[1]))
                    cx2, cy2 = min(W, int(ex[2])), min(H, int(ex[3]))
                    if cx2 - cx1 < 40 or cy2 - cy1 < 40:
                        continue
                    for gb, zc, zf in detect_bags(raw[cy1:cy2, cx1:cx2], 320, (cx1, cy1)):
                        if all(iou(gb, ob) < 0.5 for ob, _, _ in objects):
                            objects.append((gb, zc, zf))

        unowned_bags, scene_objects = [], []
        for b, cls, cf in objects:
            name = names[cls]
            # كل حاجة متعرفة بتترسم، حتى لو مفيش أي شخص أو وش في الصورة
            if cls not in BAG_CLASSES and cf >= OBJ_CONF:
                scene_objects.append(name)
                cv2.rectangle(frame, tuple(map(int, b[:2])), tuple(map(int, b[2:])), (0, 255, 255), 1)
                cv2.putText(frame, f"{name} {cf:.2f}", (int(b[0]), max(12, int(b[1]) - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            owner = max(persons, key=lambda p: overlap_ratio(b, p["box"]), default=None)
            if cls in BAG_CLASSES and not (owner and overlap_ratio(b, owner["box"]) > 0.3):
                # شنطة من غير جسم ظاهر (مثلاً جزء من الضهر بس في الكادر)
                if cf >= OBJ_CONF:
                    unowned_bags.append(name)
                    scene_objects.append(name)
                    cv2.rectangle(frame, tuple(map(int, b[:2])), tuple(map(int, b[2:])), (0, 165, 255), 1)
                    cv2.putText(frame, f"{name} (no body)", (int(b[0]), int(b[1]) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
            if cls in BAG_CLASSES and owner and overlap_ratio(b, owner["box"]) > 0.3:
                label, cm = bag_size(b, owner["box"])
                owner["bags"][f"{name} {label}"] = cm
                scene_objects.append(name)
                owner.setdefault("bag_boxes", []).append((name, cf, b))
                cv2.rectangle(frame, tuple(map(int, b[:2])), tuple(map(int, b[2:])), (0, 165, 255), 2)
                cv2.putText(frame, f"{name} {label}", (int(b[0]), int(b[1]) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            if name in NOT_HOLDABLE:
                continue
            for p in persons + [hand_only]:
                if is_holding(p, b):
                    p.setdefault("cands", []).append((name, cf, b))
                    break

        # لو نفس الحاجة اتعرفت بنوعين (موبايل/ريموت) خد اللي ثقته أعلى بس
        for p in persons + [hand_only]:
            cands = sorted(p.pop("cands", []), key=lambda x: -x[1])
            kept = []
            for name, cf, b in cands:
                if all(iou(b, kb) < 0.5 for _, _, kb in kept):
                    kept.append((name, cf, b))
            for name, cf, b in kept:
                p["held"].add(name)
                cv2.rectangle(frame, tuple(map(int, b[:2])), tuple(map(int, b[2:])), (0, 0, 255), 2)

        # ---- تثبيت النتايج + الأحداث ----
        hand_only["stable_held"] = debounce(held_deb.setdefault("hand", {}), hand_only["held"], t_vid)
        if rec:
            old_h = held_state.get("hand", set())
            for item in hand_only["stable_held"] - old_h:
                rec.event(now, t_vid, "hand", "start_holding", item)
            for item in old_h - hand_only["stable_held"]:
                rec.event(now, t_vid, "hand", "stop_holding", item)
        held_state["hand"] = hand_only["stable_held"]

        for p in persons:
            pid = p["id"]
            p["stable_held"] = debounce(held_deb.setdefault(pid, {}), p["held"], t_vid)
            p["stable_bags"] = debounce(bag_deb.setdefault(pid, {}), set(p["bags"]), t_vid)
            if pid == -1:
                continue
            last_seen[pid] = t_vid
            if pid not in seen_ids:
                # مبيتعدش غير لما يفضل ظاهر MIN_TRACK_S
                first = track_first.setdefault(pid, t_vid)
                if t_vid - first < MIN_TRACK_S:
                    continue
                seen_ids.add(pid)
                if rec:
                    rec.event(now, first, pid, "enter")
                    rec.people[pid] = {"first": first, "last": t_vid, "bags": set(), "held": set()}
            if rec:
                s = rec.people.setdefault(pid, {"first": t_vid, "last": t_vid, "bags": set(), "held": set()})
                s["last"] = t_vid
                s["bags"] |= p["stable_bags"]
                s["held"] |= p["stable_held"]
                old_h, old_b = held_state.get(pid, set()), bag_state.get(pid, set())
                for item in p["stable_held"] - old_h:
                    rec.event(now, t_vid, pid, "start_holding", item)
                for item in old_h - p["stable_held"]:
                    rec.event(now, t_vid, pid, "stop_holding", item)
                for bag in p["stable_bags"] - old_b:
                    rec.event(now, t_vid, pid, "bag_detected", bag)
                # صورة الشنطة لوحدها: الأوضح = ثقة أعلى × حجم أكبر
                for bname, bcf, bb in p.get("bag_boxes", []):
                    area = (bb[2] - bb[0]) * (bb[3] - bb[1])
                    rec.save_crop("bags", pid, bname, raw, bb, bcf * np.sqrt(area), pad=0.1)
            held_state[pid], bag_state[pid] = p["stable_held"], p["stable_bags"]

        for pid in [k for k, t in last_seen.items() if t_vid - t > EXIT_AFTER_S]:
            if rec and pid in seen_ids:
                for item in held_state.get(pid, set()):
                    rec.event(now, t_vid, pid, "stop_holding", item)
                rec.event(now, t_vid, pid, "exit")
            for d in (last_seen, held_deb, bag_deb, held_state, bag_state, track_first):
                d.pop(pid, None)

        # ---- 2) الوش ----
        face_fresh = False   # بنقص صورة الوش بس في الفريم اللي الكشف اشتغل فيه فعلًا
        if face_app and (frame_i % FACE_EVERY_N == 1 or FACE_EVERY_N == 1):
            faces = face_app.get(raw)
            face_fresh = True
        for f in faces:
            x1, y1, x2, y2 = f.bbox.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
            lm = getattr(f, "landmark_2d_106", None)
            if lm is not None:
                for x, y in lm.astype(int):
                    cv2.circle(frame, (x, y), 1, (255, 255, 0), -1)
            sex = getattr(f, "sex", "?")
            pose = getattr(f, "pose", None)
            txt = {"M": "Male", "F": "Female"}.get(sex, "?")   # النوع بس (من غير السن)
            cv2.putText(frame, txt, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)
            if rec:  # ربط الوش بالشخص عشان people.csv
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                for p in persons:
                    bx = p["box"]
                    if bx[0] <= cx <= bx[2] and bx[1] <= cy <= bx[3] and p["id"] in rec.people:
                        rec.people[p["id"]].setdefault("genders", []).append(sex)
                        # صورة الوش لوحده: الأوضح = ثقة × حجم × قد إيه الوش باصص للكاميرا
                        yaw = abs(float(pose[1])) if pose is not None else 0.0
                        area = (x2 - x1) * (y2 - y1)
                        q = float(getattr(f, "det_score", 1.0)) * np.sqrt(area) * max(0.2, 1 - yaw / 90)
                        if face_fresh:
                            rec.save_crop("faces", p["id"], "face", raw, f.bbox, q, pad=0.25)
                        break

        # ---- رسم الأشخاص والإيدين ----
        for p in persons:
            x1, y1, x2, y2 = p["box"].astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {p['id']}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            for hand in p["mp_hands"]:
                hp = hand.astype(int)
                for a, c in HAND_CONNECTIONS:
                    cv2.line(frame, tuple(hp[a]), tuple(hp[c]), (0, 0, 255), 1)
                for pt in hp:
                    cv2.circle(frame, tuple(pt), 2, (255, 255, 255), -1)
            if not p["mp_hands"]:
                for h in p["wrist_pts"]:
                    cv2.circle(frame, tuple(map(int, h)), 5, (0, 0, 255), -1)
            if p["stable_held"]:
                cv2.putText(frame, "HELD: " + ", ".join(sorted(p["stable_held"])),
                            (x1, min(H - 5, y2 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        for hand in hand_only["mp_hands"]:
            hp = hand.astype(int)
            for a, c in HAND_CONNECTIONS:
                cv2.line(frame, tuple(hp[a]), tuple(hp[c]), (0, 0, 255), 1)
            for pt in hp:
                cv2.circle(frame, tuple(pt), 2, (255, 255, 255), -1)
        if hand_only["stable_held"] and hand_only["mp_hands"]:
            hx, hy = hand_only["mp_hands"][0].max(0).astype(int)
            cv2.putText(frame, "HELD: " + ", ".join(sorted(hand_only["stable_held"])),
                        (max(0, hx - 120), min(H - 5, hy + 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        # ---- FPS ----
        t_now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, t_now - t_prev))
        t_prev = t_now

        # ---- البانل ----
        panel = np.full((H, PANEL_W, 3), 30, np.uint8)
        lines = [f"FPS        : {fps:.1f}",
                 f"Device     : {'GPU' if HALF else 'CPU'}",
                 f"Hands      : {'fingers' if hand_lm else 'wrist est.'}",
                 f"Bag model  : {'custom' if bag_model is not None else 'COCO'}",
                 f"People now : {len(persons)}",
                 f"Total seen : {len(seen_ids)}",
                 f"Faces      : {len(faces)}",
                 f"REC        : {'ON' if rec else 'OFF'}", ""]
        counts = {}
        for o in scene_objects:
            counts[o] = counts.get(o, 0) + 1
        lines.append(f"Objects    : {len(scene_objects)}")
        items = [f"{k} x{v}" if v > 1 else k for k, v in sorted(counts.items(), key=lambda x: -x[1])]
        for i in range(0, len(items), 3):
            lines.append("  " + ", ".join(items[i:i + 3]))
        if items:
            lines.append("")
        if unowned_bags:
            lines.append("Bags (no body): " + ", ".join(sorted(set(unowned_bags))))
        if hand_only["stable_held"]:
            lines.append("Hand (no body):")
            lines.append("  hand: " + ", ".join(sorted(hand_only["stable_held"])))
        for p in persons:
            lines.append(f"ID {p['id']}:")
            bags = [f"{k} ~{p['bags'][k]:.0f}cm" if k in p["bags"] else k
                    for k in sorted(p["stable_bags"])]
            lines.append("  bag : " + (", ".join(bags) or "-"))
            lines.append("  hand: " + (", ".join(sorted(p["stable_held"])) or "-"))
        for i, t in enumerate(lines):
            if 30 + i * 24 > H - 10:
                break
            color = (0, 0, 255) if t.startswith("REC") and rec else (255, 255, 255)
            cv2.putText(panel, t, (10, 30 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

        out = np.hstack([frame, panel])
        if rec:
            # للكاميرا: نستنى 20 فريم لحد ما الـ FPS يستقر عشان الفيديو المتسجل يبقى بسرعته الطبيعية
            if not cap.live or frame_i > 20:
                rec.frame(out, fps if cap.live else cap.src_fps)
            if frame_i % OBJECT_EVERY_N == 1 or OBJECT_EVERY_N == 1:
                rec.update_objects(scene_objects, t_vid)
            if frame_i % 150 == 0:
                rec.write_people()
                rec.write_objects()
        cv2.imshow("Vision", out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if rec:
        for pid in list(last_seen):
            rec.event(time.time(), t_vid if frame_i else 0, pid, "exit")
        rec.close()
    if hand_lm is not None:
        hand_lm.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--no-video", action="store_true", help="don't save video")
    ap.add_argument("--no-log", action="store_true", help="don't save CSV")
    ap.add_argument("--no-crops", action="store_true", help="don't save face/bag images")
    ap.add_argument("--coco-bags", action="store_true", help="ignore models/bags_best.pt, use normal YOLO for bags")
    ap.add_argument("--bag-model", default=None, help="path to a bag model .pt to try (default models/bags_best.pt)")
    a = ap.parse_args()
    main(a.source, save_video=not a.no_video, save_log=not a.no_log, save_crops=not a.no_crops, coco_bags=a.coco_bags, bag_model_path=a.bag_model)
