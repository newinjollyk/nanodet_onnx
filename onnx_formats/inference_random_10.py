import os
import random
import cv2
import numpy as np
import onnxruntime as ort

# ================= CONFIG =================
ONNX_PATH = "/home/newin/Projects/nanodet_sign/onnx_formats/nanodet_decoded.onnx"
TEST_FOLDER = "/home/newin/Projects/nanodet_sign/Nano_sign/yolodark_nano_dataset/test/images"
OUTPUT_DIR = "/home/newin/Projects/nanodet_sign/onnx_formats/predicted_out"

INPUT_SIZE = 416
NUM_CLASSES = 21
REG_MAX = 7
STRIDES = [8, 16, 32, 64]

SCORE_THRESH = 0.35
NMS_THRESH = 0.6
NUM_IMAGES = 10

CLASS_NAMES = [
    "Keep left", "Keep right", "No U-turn", "No left turn",
    "No parking", "No right turn", "No stopping", "Parking",
    "Pedestrian Crossing", "Speed Limit -100-", "Speed Limit -30-",
    "Speed Limit -60-", "Stop Sign",
    "Traffic Light -Green-", "Traffic Light -Red-", "Traffic Light -Yellow-",
    "U-turn", "bike", "motobike", "person", "vehicle"
]

MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32)
STD  = np.array([57.375, 57.12, 58.395], dtype=np.float32)

# ================= UTILS =================
def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)

def compute_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    return inter / (area1 + area2 - inter + 1e-6)

def nms(boxes, scores, iou_thresh):
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break

        ious = compute_iou(boxes[i], boxes[order[1:]])
        order = order[1:][ious < iou_thresh]

    return keep

def get_random_images(folder, num_images):
    exts = (".jpg", ".jpeg", ".png")
    images = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(exts)
    ]

    if not images:
        raise RuntimeError("❌ No images found in test folder")

    return random.sample(images, min(num_images, len(images)))

# ================= DECODE =================
def decode_nanodet(pred):
    boxes_all, scores_all, labels_all = [], [], []
    offset = 0

    for stride in STRIDES:
        feat = INPUT_SIZE // stride
        num_points = feat * feat

        cls_pred = pred[:, offset:offset+num_points, :NUM_CLASSES]
        reg_pred = pred[:, offset:offset+num_points, NUM_CLASSES:]
        offset += num_points

        cls_pred = cls_pred[0]
        reg_pred = reg_pred[0]

        cls_scores = cls_pred.max(axis=1)
        cls_labels = cls_pred.argmax(axis=1)

        keep = cls_scores > SCORE_THRESH
        if not np.any(keep):
            continue

        cls_scores = cls_scores[keep]
        cls_labels = cls_labels[keep]
        reg_pred = reg_pred[keep]

        reg_pred = reg_pred.reshape(-1, 4, REG_MAX + 1)
        prob = softmax(reg_pred, axis=2)
        dist = np.sum(prob * np.arange(REG_MAX + 1), axis=2)

        idxs = np.where(keep)[0]
        gy, gx = np.divmod(idxs, feat)

        cx = (gx + 0.5) * stride
        cy = (gy + 0.5) * stride

        x1 = cx - dist[:, 0] * stride
        y1 = cy - dist[:, 1] * stride
        x2 = cx + dist[:, 2] * stride
        y2 = cy + dist[:, 3] * stride

        boxes_all.append(np.stack([x1, y1, x2, y2], axis=1))
        scores_all.append(cls_scores)
        labels_all.append(cls_labels)

    if not boxes_all:
        return [], [], []

    boxes = np.concatenate(boxes_all)
    scores = np.concatenate(scores_all)
    labels = np.concatenate(labels_all)

    final_boxes, final_scores, final_labels = [], [], []

    for c in range(NUM_CLASSES):
        idx = labels == c
        if not np.any(idx):
            continue

        keep = nms(boxes[idx], scores[idx], NMS_THRESH)
        final_boxes.append(boxes[idx][keep])
        final_scores.append(scores[idx][keep])
        final_labels.extend([c] * len(keep))

    if not final_boxes:
        return [], [], []

    return (
        np.concatenate(final_boxes),
        np.concatenate(final_scores),
        final_labels
    )

# ================= INFERENCE =================
def run_inference(image_path, session):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    inp = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32)
    inp = (inp - MEAN) / STD
    inp = inp.transpose(2, 0, 1)[None]

    pred = session.run(None, {"data": inp})[0]
    boxes, scores, labels = decode_nanodet(pred)

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        x1 = int(x1 * w / INPUT_SIZE)
        y1 = int(y1 * h / INPUT_SIZE)
        x2 = int(x2 * w / INPUT_SIZE)
        y2 = int(y2 * h / INPUT_SIZE)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{CLASS_NAMES[label]} {score:.2f}"
        cv2.putText(img, text, (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return img

# ================= MAIN =================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    session = ort.InferenceSession(
        ONNX_PATH, providers=["CPUExecutionProvider"]
    )

    images = get_random_images(TEST_FOLDER, NUM_IMAGES)

    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {img_path}")

        out = run_inference(img_path, session)

        save_path = os.path.join(OUTPUT_DIR, f"result_{i}.jpg")
        cv2.imwrite(save_path, out)

        cv2.imshow("NanoDet Random Inference", out)
        key = cv2.waitKey(0)

        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
