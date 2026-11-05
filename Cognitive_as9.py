# UCS420 Assignment 9
# Vision based Cognitive Systems using Landing AI
# Gurkirpa Singh, 3Q34, 1024170378


# pip install requests opencv-python easyocr kagglehub matplotlib pandas
import os, re, json, time, random, shutil
import cv2
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

random.seed(42)

with open("landing_keys.json") as f:
    keys = json.load(f)

API_KEY = keys["api_key"]
MED_ENDPOINT = keys["medicine_endpoint"]
EMO_ENDPOINT = keys["emotion_endpoint"]
URL = "https://predict.app.landing.ai/inference/v1/predict"


def predict(img_path, endpoint):
    with open(img_path, "rb") as f:
        res = requests.post(URL, params={"endpoint_id": endpoint},
                            headers={"apikey": API_KEY},
                            files={"file": f})
    res.raise_for_status()
    time.sleep(0.3)
    return res.json()

def get_preds(obj):
    out = []
    if isinstance(obj, dict):
        if "labelName" in obj:
            out.append(obj)
        for v in obj.values():
            out += get_preds(v)
    elif isinstance(obj, list):
        for v in obj:
            out += get_preds(v)
    return out

def list_images(folder):
    exts = (".jpg", ".jpeg", ".png")
    return sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts))


# Q1
raw_dir = "data/medicine/raw"
up_dir = "data/medicine/upload"
test_dir = "data/medicine/test"

for d in (up_dir, test_dir):
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)

def resize_max(img, side=1024):
    h, w = img.shape[:2]
    scale = side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img

imgs = list_images(raw_dir)
random.shuffle(imgs)
cut = int(0.8 * len(imgs))

for i, path in enumerate(imgs):
    img = cv2.imread(path)
    if img is None:
        continue
    dest = up_dir if i < cut else test_dir
    cv2.imwrite(os.path.join(dest, f"med_{i:03d}.jpg"), resize_max(img))

print("total images:", len(imgs))
print("for LandingLens:", len(os.listdir(up_dir)))
print("kept for testing:", len(os.listdir(test_dir)))


test_imgs = list_images(test_dir)

# raw response for one image, to see the format
sample = predict(test_imgs[0], MED_ENDPOINT)
print(json.dumps(sample, indent=2)[:1500])


def detect_labels(path, min_score=0.4):
    preds = get_preds(predict(path, MED_ENDPOINT))
    boxes = []
    for p in preds:
        c = p.get("coordinates")
        if c and p.get("score", 0) >= min_score:
            boxes.append((c["xmin"], c["ymin"], c["xmax"], c["ymax"], p["score"]))
    return sorted(boxes, key=lambda b: -b[4])

detections = {path: detect_labels(path) for path in test_imgs}

n = min(6, len(test_imgs))
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, path in zip(axes.flat, test_imgs[:n]):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    for x1, y1, x2, y2, s in detections[path]:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 4)
        cv2.putText(img, f"Label {s:.2f}", (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)
    ax.imshow(img)
    ax.set_title(os.path.basename(path))
    ax.axis("off")
for ax in axes.flat[n:]:
    ax.axis("off")
plt.tight_layout()
plt.show()


import easyocr, certifi
# easyocr downloads its models over https, point it at certifi's certs so the download works
os.environ["SSL_CERT_FILE"] = certifi.where()
reader = easyocr.Reader(["th", "en"], gpu=False)

def read_text(img):
    res = reader.readtext(img)
    # top to bottom, left to right
    res.sort(key=lambda r: (r[0][0][1], r[0][0][0]))
    return [(t, abs(b[2][1] - b[0][1])) for b, t, conf in res if conf > 0.3]

def parse_label(lines):
    text = " ".join(t for t, h in lines)

    skip = ("exp", "mfg", "batch", "b.no", "lot", "mrp", "rx", "price")
    cands = [(t, h) for t, h in lines
             if re.match(r"[A-Za-z]{3,}", t.strip()) and not t.lower().startswith(skip)]
    name = max(cands, key=lambda c: c[1])[0].strip() if cands else None

    strength = re.search(r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|g|ml|iu)\b", text, re.I)
    expiry = re.search(r"(?:exp\.?|expiry(?: date)?|use before)\s*[:.]?\s*([0-9]{1,2}\s?[/.\-]\s?[0-9]{2,4}|[a-z]{3}\.?\s?[-/]?\s?[0-9]{2,4})", text, re.I)
    batch = re.search(r"(?:b\.?\s?no\.?|batch(?:\s?no\.?)?|lot)\s*[:.]?\s*([a-z0-9\-]{3,})", text, re.I)

    return {
        "name": name,
        "strength": strength.group(0) if strength else None,
        "expiry": expiry.group(1) if expiry else None,
        "batch": batch.group(1) if batch else None,
        "text": text,
    }


def medicine_assistant(path, show=True):
    img = cv2.imread(path)
    boxes = detections.get(path) or detect_labels(path)

    if boxes:
        x1, y1, x2, y2, s = boxes[0]
        pad = 15
        crop = img[max(y1 - pad, 0):y2 + pad, max(x1 - pad, 0):x2 + pad]
        found = True
    else:
        crop = img      # no label found, read the whole image instead
        found = False

    info = parse_label(read_text(crop))

    if show:
        plt.figure(figsize=(5, 4))
        plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        plt.title("label crop" if found else "no label detected, full image")
        plt.axis("off")
        plt.show()

        print("Assistant:")
        if info["name"]:
            print(f"  This looks like {info['name']}.")
        else:
            print("  I couldn't make out the medicine name clearly.")
        if info["strength"]:
            print(f"  Strength: {info['strength']}.")
        if info["expiry"]:
            print(f"  It expires on {info['expiry']}.")
        if info["batch"]:
            print(f"  Batch number {info['batch']}.")
        print("  OCR text:", info["text"][:200])

    info["label_found"] = found
    return info

for path in test_imgs[:3]:
    print(os.path.basename(path))
    medicine_assistant(path)
    print()


rows = []
for path in test_imgs:
    info = medicine_assistant(path, show=False)
    rows.append({"image": os.path.basename(path), "label_found": info["label_found"],
                 "name": info["name"], "strength": info["strength"],
                 "expiry": info["expiry"], "batch": info["batch"]})

med_df = pd.DataFrame(rows)
print("labels detected:", med_df["label_found"].sum(), "/", len(med_df))
print(med_df)


# Q2
import kagglehub
fer_path = kagglehub.dataset_download("msambare/fer2013")
print(fer_path, os.listdir(fer_path))


emotions = ["happy", "sad", "angry", "surprise"]
n_train, n_test = 40, 10
emo_up = "data/emotion/upload"
emo_test = "data/emotion/test"

for base in (emo_up, emo_test):
    shutil.rmtree(base, ignore_errors=True)

def copy_faces(files, dest):
    os.makedirs(dest)
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (192, 192), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(dest, os.path.basename(f)), img)

for emo in emotions:
    train_files = random.sample(list_images(os.path.join(fer_path, "train", emo)), n_train)
    test_files = random.sample(list_images(os.path.join(fer_path, "test", emo)), n_test)
    copy_faces(train_files, os.path.join(emo_up, emo))
    copy_faces(test_files, os.path.join(emo_test, emo))
    print(emo, "-> upload:", n_train, " test:", n_test)


def predict_emotion(path):
    preds = get_preds(predict(path, EMO_ENDPOINT))
    if not preds:
        return None, 0.0
    top = max(preds, key=lambda p: p.get("score", 0))
    return top["labelName"].lower(), top.get("score", 0)

results = []
for emo in emotions:
    for path in list_images(os.path.join(emo_test, emo)):
        pred, score = predict_emotion(path)
        results.append({"image": path, "actual": emo, "predicted": pred, "score": round(score, 3)})

emo_df = pd.DataFrame(results)
acc = (emo_df["actual"] == emo_df["predicted"]).mean()
print(f"accuracy on {len(emo_df)} test faces: {acc:.2%}")
print(pd.crosstab(emo_df["actual"], emo_df["predicted"], rownames=["actual"], colnames=["predicted"]))


fig, axes = plt.subplots(2, 4, figsize=(12, 6))
picks = emo_df.groupby("actual").head(2).reset_index(drop=True)
for ax, (_, r) in zip(axes.flat, picks.iterrows()):
    ax.imshow(cv2.imread(r["image"], cv2.IMREAD_GRAYSCALE), cmap="gray")
    color = "green" if r["actual"] == r["predicted"] else "red"
    ax.set_title(f"actual: {r['actual']}\npred: {r['predicted']} ({r['score']:.2f})", color=color)
    ax.axis("off")
plt.tight_layout()
plt.show()


replies = {
    "happy": "You look happy! Want me to play something upbeat to keep the mood going?",
    "sad": "You seem a bit low. Want to take a short break, or talk about what's on your mind?",
    "angry": "You look frustrated. Maybe take a few deep breaths, I can wait.",
    "surprise": "Something caught you off guard! Anything you want me to look up?",
}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def emotion_assistant(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face = cv2.resize(gray[y:y + h, x:x + w], (192, 192))
    else:
        face = cv2.resize(gray, (192, 192))   # already a face crop

    cv2.imwrite("face_tmp.jpg", face)
    emo, score = predict_emotion("face_tmp.jpg")

    plt.figure(figsize=(3, 3))
    plt.imshow(face, cmap="gray")
    plt.title(f"{emo} ({score:.2f})")
    plt.axis("off")
    plt.show()

    print("Assistant:", replies.get(emo, "I couldn't read your expression clearly, could you face the camera?"))
    return emo

# one test face from each class
for emo in emotions:
    emotion_assistant(list_images(os.path.join(emo_test, emo))[0])


# my own photo
my_photo = "as9_images/my_face.jpg"
if os.path.exists(my_photo):
    emotion_assistant(my_photo)
else:
    print("no photo of my own added yet")
