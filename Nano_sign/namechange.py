import os
import shutil

# ====== CHANGE THESE PATHS ======
IMAGE_DIR = "/home/newin/Projects/nanodet_sign/Nano_sign/yolodark/valid/images"
LABEL_DIR = "/home/newin/Projects/nanodet_sign/Nano_sign/yolodark/valid/labels"

OUT_IMAGE_DIR = "/home/newin/Projects/nanodet_sign/Nano_sign/yolodark/valid/images_simple"
OUT_LABEL_DIR = "/home/newin/Projects/nanodet_sign/Nano_sign/yolodark/valid/labels_simple"
# =================================

os.makedirs(OUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUT_LABEL_DIR, exist_ok=True)

image_exts = [".jpg", ".jpeg", ".png"]

count = 0
skipped = 0

for label_file in sorted(os.listdir(LABEL_DIR)):
    if not label_file.endswith(".txt"):
        continue

    base = label_file[:-4]  # remove .txt
    image_path = None
    image_ext = None

    for ext in image_exts:
        candidate = os.path.join(IMAGE_DIR, base + ext)
        if os.path.exists(candidate):
            image_path = candidate
            image_ext = ext
            break

    if image_path is None:
        skipped += 1
        print(f"[SKIP] No image for label: {label_file}")
        continue

    count += 1
    new_name = f"{count:06d}"

    # Copy image
    shutil.copy(
        image_path,
        os.path.join(OUT_IMAGE_DIR, new_name + image_ext)
    )

    # Copy label
    shutil.copy(
        os.path.join(LABEL_DIR, label_file),
        os.path.join(OUT_LABEL_DIR, new_name + ".txt")
    )

print("\n=== DONE ===")
print(f"Valid pairs copied : {count}")
print(f"Skipped labels     : {skipped}")
