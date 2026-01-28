from concurrent.futures import process
import os
import csv
import time
import random
import psutil
import cv2
import numpy as np
import onnxruntime as ort

# ================= CONFIG =================
ONNX_PATH = "/home/newin/Projects/nanodet_sign/onnx_formats/nanodet_decoded.onnx"
TEST_FOLDER = "/home/newin/Projects/nanodet_sign/Nano_sign/yolodark_nano_dataset/test/images"
OUTPUT_DIR = "/home/newin/Projects/nanodet_sign/onnx_formats/predicted_out"
CSV_PATH = "/home/newin/Projects/nanodet_sign/onnx_formats/benchmark.csv"

INPUT_SIZE = 416
NUM_CLASSES = 21
REG_MAX = 7
STRIDES = [8, 16, 32, 64]

SCORE_THRESH = 0.6        # 🔴 IMPORTANT
NMS_THRESH = 0.6
TOP_K = 100               # 🔴 IMPORTANT
MAX_FINAL = 50            # 🔴 FINAL CAP
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
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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

def nms(boxes, scores, thresh):
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        iou = compute_iou(boxes[i], boxes[order[1:]])
        order = order[1:][iou < thresh]
    return keep

def get_random_images(folder, n):
    imgs = [os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    return random.sample(imgs, min(n, len(imgs)))

# ================= DECODE =================
def decode_nanodet(pred):
    boxes_all, scores_all, labels_all = [], [], []
    offset = 0

    for stride in STRIDES:
        feat = INPUT_SIZE // stride
        num_points = feat * feat

        cls_logits = pred[:, offset:offset+num_points, :NUM_CLASSES][0]
        reg_pred = pred[:, offset:offset+num_points, NUM_CLASSES:][0]
        offset += num_points

        cls_prob = sigmoid(cls_logits)
        scores = cls_prob.max(axis=1)
        labels = cls_prob.argmax(axis=1)

        keep = scores > SCORE_THRESH
        if not np.any(keep):
            continue

        scores = scores[keep]
        labels = labels[keep]
        reg_pred = reg_pred[keep]
        idxs = np.where(keep)[0]

        # 🔴 TOP-K
        if len(scores) > TOP_K:
            top = np.argsort(scores)[::-1][:TOP_K]
            scores = scores[top]
            labels = labels[top]
            reg_pred = reg_pred[top]
            idxs = idxs[top]

        reg_pred = reg_pred.reshape(-1, 4, REG_MAX + 1)
        prob = softmax(reg_pred, axis=2)
        dist = np.sum(prob * np.arange(REG_MAX + 1), axis=2)

        gy, gx = np.divmod(idxs, feat)
        cx = (gx + 0.5) * stride
        cy = (gy + 0.5) * stride

        x1 = cx - dist[:, 0] * stride
        y1 = cy - dist[:, 1] * stride
        x2 = cx + dist[:, 2] * stride
        y2 = cy + dist[:, 3] * stride

        boxes_all.append(np.stack([x1, y1, x2, y2], axis=1))
        scores_all.append(scores)
        labels_all.append(labels)

    if not boxes_all:
        return np.empty((0,4)), np.array([]), []

    boxes = np.concatenate(boxes_all)
    scores = np.concatenate(scores_all)
    labels = np.concatenate(labels_all)

    final_b, final_s, final_l = [], [], []

    for c in range(NUM_CLASSES):
        idx = labels == c
        if not np.any(idx):
            continue
        keep = nms(boxes[idx], scores[idx], NMS_THRESH)
        final_b.append(boxes[idx][keep])
        final_s.append(scores[idx][keep])
        final_l.extend([c]*len(keep))

    boxes = np.concatenate(final_b)
    scores = np.concatenate(final_s)
    labels = final_l

    # 🔴 FINAL CAP
    if len(scores) > MAX_FINAL:
        top = np.argsort(scores)[::-1][:MAX_FINAL]
        boxes = boxes[top]
        scores = scores[top]
        labels = [labels[i] for i in top]

    return boxes, scores, labels

# ================= INFERENCE =================
def run_inference(img_path, session, process):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    inp = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32)
    inp = (inp - MEAN) / STD
    inp = inp.transpose(2, 0, 1)[None]

    t0 = time.perf_counter()
    pred = session.run(None, {"data": inp})[0]
    t1 = time.perf_counter()

    boxes, scores, labels = decode_nanodet(pred)
    t2 = time.perf_counter()

    infer_ms = (t2 - t0) * 1000
    pure_ms = (t1 - t0) * 1000
    mem = process.memory_info().rss / (1024 * 1024)
    cpu = process.cpu_percent(interval=None) / psutil.cpu_count()

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        x1 = int(x1 * w / INPUT_SIZE)
        y1 = int(y1 * h / INPUT_SIZE)
        x2 = int(x2 * w / INPUT_SIZE)
        y2 = int(y2 * h / INPUT_SIZE)

        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(img, f"{CLASS_NAMES[label]} {score:.2f}",
                    (x1, max(0,y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    avg_conf = float(scores.mean()) if len(scores) else 0.0
    return img, infer_ms, pure_ms, cpu, mem, len(scores), avg_conf

# ================= MAIN =================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    process = psutil.Process(os.getpid())
    session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    model_size_mb = os.path.getsize(ONNX_PATH) / (1024*1024)

    images = get_random_images(TEST_FOLDER, NUM_IMAGES)
    rows = []
    total_time = 0.0
    total_pure = 0.0

    for i, img in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {os.path.basename(img)}")
        out, infer_ms, pure_ms, cpu, mem, dets, avg_conf = run_inference(img, session, process)
        total_time += infer_ms
        total_pure += pure_ms

        rows.append([os.path.basename(img), infer_ms, cpu, mem, dets, avg_conf])
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"result_{i}.jpg"), out)

    avg_time = total_time / len(rows)
    avg_pure = total_pure / len(rows)
    fps = 1000.0 / avg_time

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image","inference_ms","cpu_percent","memory_mb","num_detections","avg_confidence"])
        writer.writerows(rows)
        writer.writerow([])
        writer.writerow(["model_size_mb", model_size_mb])
        writer.writerow(["avg_pure_inference_ms", avg_pure])
        writer.writerow(["avg_end_to_end_ms", avg_time])
        writer.writerow(["fps", fps])

    print("\n✅ Benchmark complete")
    print(f"Model size: {model_size_mb:.2f} MB")
    print(f"Avg pure inference: {avg_pure:.2f} ms")
    print(f"Avg end-to-end: {avg_time:.2f} ms")
    print(f"FPS: {fps:.2f}")
    print(f"CSV saved to: {CSV_PATH}")

if __name__ == "__main__":
    main()
