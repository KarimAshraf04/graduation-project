# تدريب موديل الشنط

الهدف: موديل بيعرف الشنط أدق من YOLO العادي، ومتدرب على **صور Open Images** وكمان على **صور من الكاميرا بتاعتك**.
بعد التدريب، الموديل الجديد بيتحفظ في `models\bags_best.pt`، والبرنامج الأساسي بيستخدمه لوحده من غير ما تعمل حاجة. هتلاقي في البانل `Bag model: custom`.

الكلاسات: `backpack` (شنطة ضهر)، و`handbag` (شنطة إيد)، و`suitcase` (شنطة سفر).

كل الأوامر دي بتتكتب من `C:\vision`.

## 0) مرة واحدة: تسطيب FiftyOne

```powershell
.\venv\Scripts\python -m pip install fiftyone
```

## 1) نزّل صور الشنط من Open Images

```powershell
.\venv\Scripts\python training\download_openimages.py
```

- بينزّل صور فيها Backpack وHandbag وSuitcase، ومعاها المربعات جاهزة، ويحطها في `training\data\openimages`.
- لو عايز صور أكتر ضيف `--use-train`. أول مرة هيكون أبطأ بكتير، لأنه بينزّل ملف annotations كبير.
- لو عايز تجرّب بسرعة الأول ضيف `--max 300`.

## 2) صوّر صورك انت

```powershell
.\venv\Scripts\python training\capture_frames.py
```

- بيحفظ فريم نضيف (من غير رسومات) كل 0.7 ثانية لما يكون فيه حد قدام الكاميرا.
- `SPACE` يحفظ الفريم ده دلوقتي، و`A` يشغّل أو يقفل الحفظ التلقائي، و`Q` يخرج.
- لو عندك فيديو صوّرته بالموبايل: `--source clip.mp4`
- **صوّر من كذا زاوية:** من ضهر ومن جنب ومن قدام، قريب وبعيد، بشنط مختلفة، **وكمان ناس من غير شنط**.
- الهدف حوالي **300 لـ 1000 صورة**.

## 3) labels تلقائية

```powershell
.\venv\Scripts\python training\auto_label.py
```

الموديل `yolo26m` بيرسم المربعات مبدئيًا لوحده.

## 4) راجع وصلّح

```powershell
.\venv\Scripts\python training\review_labels.py
```

| الحركة | بتعمل إيه |
|---|---|
| سحب بالماوس | تعمل مربع جديد |
| كليك يمين على مربع | تمسحه |
| `1` / `2` / `3` | تغيّر الكلاس: backpack / handbag / suitcase |
| `D` أو `SPACE` أو ← → | تحفظ وتروح للصورة اللي بعدها، أو `A` للي قبلها |
| `C` | الصورة مفيهاش شنط (مثال سلبي مفيد) |
| `R` | ترفض الصورة (مش هتدخل التدريب) |
| `Q` | تحفظ وتخرج، وبيفتكر انت وقفت فين |

**مهم:** راجع **كل** الصور. مربع ناقص أو غلط بيعلّم الموديل غلط.

## 5) درّب

```powershell
.\venv\Scripts\python training\train.py
```

- على RTX 3060 بياخد حوالي **1 لـ 3 ساعات**، على حسب عدد الصور.
- لو التدريب مش بيتحسن 20 epoch ورا بعض بيقف لوحده.
- في الآخر بيطبع **mAP50**: فوق 0.7 كويس، وفوق 0.85 ممتاز.
- الرسومات والنتايج التفصيلية بتتحفظ في `training\runs\bags\`.
- لو ظهر error بتاع out of memory: `--batch 4`

## 6) شغّل البرنامج عادي

```powershell
.\run.bat
```

لازم البانل يكتب `Bag model: custom`.

## عشان تحسّن أكتر (دورة تانية)

1. صوّر صور جديدة بـ `capture_frames.py`، خصوصًا في المواقف اللي الموديل غلط فيها.
2. اعمل labels بالموديل الجديد بتاعك، وهتطلع أدق بكتير:
   ```powershell
   .\venv\Scripts\python training\auto_label.py --model models\bags_best.pt
   ```
3. راجع، وبعدين درّب تاني.
