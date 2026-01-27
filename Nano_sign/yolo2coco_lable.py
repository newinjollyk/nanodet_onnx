import os
import glob
import json
from PIL import Image

# ============================
# ROOT DATASET PATH
# ============================
ROOT = "/home/newin/projects/Nano_sign/yolodark_simple"

OUT_DIR = "/home/newin/projects/Nano_sign/coco/annotations"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================
# CLASS NAMES (ORDER MATTERS)
# ============================
CLASS_NAMES = [
    "Keep left", "Keep right", "No U-turn", "No left turn", "No parking",
    "No right turn", "No stopping", "Parking", "Pedestrian Crossing",
    "Speed Limit -100-", "Speed Limit -30-", "Speed Limit -60-",
    "Stop Sign", "Traffic Light -Green-", "Traffic Light -Red-",
    "Traffic Light -Yellow-", "U-turn", "bike", "motobike",
    "person", "vehicle"
]

CATEGORIES = [{"id": i + 1, "name": n} for i, n in enumerate(CLASS_NAMES)]

# ============================
# CORE FUNCTION
# ============================
def convert_split(split):
    images_dir = os.path.join(ROOT, split, "images")
    labels_dir = os.path.join(ROOT, split, "labels")
    out_json = os.path.join(OUT_DIR, f"instances_{split}.json")

    coco = {
        "images": [],
        "annotations": [],
        "categories": CATEGORIES
    }

    ann_id = 1
    img_id = 1

    for label_file in sorted(os.listdir(labels_dir)):
        if not label_file.endswith(".txt"):
            continue

        stem = os.path.splitext(label_file)[0]

        # Match Roboflow-style filenames
        matches = (
            glob.glob(os.path.join(images_dir, stem + "*.jpg")) +
            glob.glob(os.path.join(images_dir, stem + "*.png"))
        )

        if not matches:
            print(f"⚠️ Missing image for {label_file}")
            continue

        image_path = matches[0]
        img = Image.open(image_path)
        width, height = img.size

        coco["images"].append({
            "id": img_id,
            "file_name": os.path.basename(image_path),
            "width": width,
            "height": height
        })

        with open(os.path.join(labels_dir, label_file)) as f:
            for line in f:
                cls, xc, yc, w, h = map(float, line.strip().split())

                x = (xc - w / 2) * width
                y = (yc - h / 2) * height
                w *= width
                h *= height

                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(cls) + 1,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0
                })
                ann_id += 1

        img_id += 1

    with open(out_json, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"✅ {split}: {len(coco['images'])} images, {len(coco['annotations'])} annotations")


# ============================
# RUN FOR ALL SPLITS
# ============================
for split in ["train", "valid", "test"]:
    convert_split(split)
