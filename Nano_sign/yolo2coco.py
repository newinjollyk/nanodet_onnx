import os
import glob
import json
from PIL import Image

# ======================
# PATHS (EDIT IF NEEDED)
# ======================
IMAGES_DIR = "yolodark/valid/images"
LABELS_DIR = "yolodark/valid/labels"
OUTPUT_JSON = "dataset/annotations/instances_val.json"

# ======================
# CLASS NAMES (MUST MATCH YOLO)
# ======================
CLASS_NAMES = [
    "Keep left", "Keep right", "No U-turn", "No left turn", "No parking",
    "No right turn", "No stopping", "Parking", "Pedestrian Crossing",
    "Speed Limit -100-", "Speed Limit -30-", "Speed Limit -60-",
    "Stop Sign", "Traffic Light -Green-", "Traffic Light -Red-",
    "Traffic Light -Yellow-", "U-turn", "bike", "motobike",
    "person", "vehicle"
]

coco = {
    "images": [],
    "annotations": [],
    "categories": [{"id": i + 1, "name": n} for i, n in enumerate(CLASS_NAMES)]
}

ann_id = 1
img_id = 1

for label_file in sorted(os.listdir(LABELS_DIR)):
    if not label_file.endswith(".txt"):
        continue

    stem = os.path.splitext(label_file)[0]

    # 🔑 Roboflow image matching
    matches = (
        glob.glob(os.path.join(IMAGES_DIR, stem + "*.jpg")) +
        glob.glob(os.path.join(IMAGES_DIR, stem + "*.png"))
    )

    if not matches:
        print(f"⚠️ Missing image for {label_file}")
        continue

    image_path = matches[0]
    img = Image.open(image_path)
    w, h = img.size

    coco["images"].append({
        "id": img_id,
        "file_name": os.path.basename(image_path),
        "width": w,
        "height": h
    })

    with open(os.path.join(LABELS_DIR, label_file)) as f:
        for line in f:
            cls, xc, yc, bw, bh = map(float, line.split())

            x = (xc - bw / 2) * w
            y = (yc - bh / 2) * h
            bw *= w
            bh *= h

            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(cls) + 1,
                "bbox": [x, y, bw, bh],
                "area": bw * bh,
                "iscrowd": 0
            })
            ann_id += 1

    img_id += 1

os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, "w") as f:
    json.dump(coco, f, indent=2)

print("✅ COCO created successfully")
print("Images:", len(coco["images"]))
print("Annotations:", len(coco["annotations"]))
