import os
import random
import cv2
import csv
import time
import psutil
import numpy as np
import onnxruntime as ort



# ================= CONFIG =================
ONNX_PATH = "/home/newin/Projects/nanodet_sign/onnx_formats/nanodet_decoded.onnx"
TEST_FOLDER = "/home/newin/Projects/nanodet_sign/Nano_sign/yolodark_nano_dataset/test/images"
OUTPUT_DIR = "/home/newin/Projects/nanodet_sign/onnx_formats/predicted_out"
CSV_PATH = "/home/newin/Projects/nanodet_sign/onnx_formats/metrics.csv"

INPUT_SIZE = 416
NUM_CLASSES = 21
REG_MAX = 7
STRIDES = [8, 16, 32, 64]

SCORE_THRESH = 0.5
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
    return random.sample(images, min(num_images, len(images)))

# ================= DECODE =================
def decode_nanodet(pred):
    boxes_all, scores_all, labels_all = [], [], []
    offset = 0

    TOP_K = 200  # 🔴 global cap per feature level

    for stride in STRIDES:
        feat = INPUT_SIZE // stride
        num_points = feat * feat

        cls_pred = pred[:, offset:offset + num_points, :NUM_CLASSES]
        reg_pred = pred[:, offset:offset + num_points, NUM_CLASSES:]
        offset += num_points

        cls_pred = cls_pred[0]
        reg_pred = reg_pred[0]

        # Sigmoid on class logits
        cls_prob = 1 / (1 + np.exp(-cls_pred))
        cls_scores = cls_prob.max(axis=1)
        cls_labels = cls_prob.argmax(axis=1)

        # Score threshold
        keep = cls_scores > SCORE_THRESH
        if not np.any(keep):
            continue

        cls_scores = cls_scores[keep]
        cls_labels = cls_labels[keep]
        reg_pred = reg_pred[keep]
        idxs = np.where(keep)[0]

        # 🔴 TOP-K APPLIED HERE (CORRECT PLACE)
        if len(cls_scores) > TOP_K:
            topk_idx = np.argsort(cls_scores)[::-1][:TOP_K]
            cls_scores = cls_scores[topk_idx]
            cls_labels = cls_labels[topk_idx]
            reg_pred = reg_pred[topk_idx]
            idxs = idxs[topk_idx]

        # Decode bbox distances
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
        scores_all.append(cls_scores)
        labels_all.append(cls_labels)

    if not boxes_all:
        return [], [], []

    boxes = np.concatenate(boxes_all)
    scores = np.concatenate(scores_all)
    labels = np.concatenate(labels_all)

    final_boxes, final_scores, final_labels = [], [], []

    # Class-wise NMS
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

def run_inference(image_path, session, process):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # ---------------------------
    # Preprocess
    # ---------------------------
    inp = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32)
    inp = (inp - MEAN) / STD
    inp = inp.transpose(2, 0, 1)[None]

    # ---------------------------
    # Inference timing
    # ---------------------------
    t0 = time.perf_counter()
    pred = session.run(None, {"data": inp})[0]
    t1 = time.perf_counter()

    pure_infer_ms = (t1 - t0) * 1000  # model only

    # ---------------------------
    # Decode + postprocess timing
    # ---------------------------
    t2 = time.perf_counter()
    boxes, scores, labels = decode_nanodet(pred)
    t3 = time.perf_counter()

    infer_ms = (t3 - t0) * 1000  # end-to-end

    # ---------------------------
    # Metrics
    # ---------------------------
    mem_mb = process.memory_info().rss / (1024 * 1024)
    cpu_pct = process.cpu_percent(interval=None)

    num_detections = len(boxes)
    avg_confidence = float(np.mean(scores)) if num_detections > 0 else 0.0

    # ---------------------------
    # Draw results
    # ---------------------------
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        x1 = int(x1 * w / INPUT_SIZE)
        y1 = int(y1 * h / INPUT_SIZE)
        x2 = int(x2 * w / INPUT_SIZE)
        y2 = int(y2 * h / INPUT_SIZE)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{CLASS_NAMES[label]} {score:.2f}"
        cv2.putText(
            img,
            text,
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    return (
        img,
        pure_infer_ms,     # model latency only
        infer_ms,          # full pipeline latency
        cpu_pct,
        mem_mb,
        num_detections,
        avg_confidence
    )

# ================= MAIN =================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    process = psutil.Process(os.getpid())
    process.cpu_percent(interval=None)  # warm-up CPU meter

    session = ort.InferenceSession(
        ONNX_PATH, providers=["CPUExecutionProvider"]
    )

    model_size_mb = os.path.getsize(ONNX_PATH) / (1024 * 1024)
    images = get_random_images(TEST_FOLDER, NUM_IMAGES)

    rows = []
    total_infer_time = 0.0
    total_pure_time = 0.0

    for i, img_path in enumerate(images, 1):
        img_name = os.path.basename(img_path)
        print(f"[{i}/{len(images)}] {img_name}")

        (
            out,
            pure_infer_ms,
            infer_ms,
            cpu_pct,
            mem_mb,
            num_dets,
            avg_conf
        ) = run_inference(img_path, session, process)

        total_infer_time += infer_ms
        total_pure_time += pure_infer_ms

        rows.append([
            img_name,
            round(pure_infer_ms, 3),
            round(infer_ms, 3),
            round(cpu_pct, 2),
            round(mem_mb, 2),
            num_dets,
            round(avg_conf, 4),
            round(model_size_mb, 3),
        ])

        cv2.imwrite(
            os.path.join(OUTPUT_DIR, f"result_{i}.jpg"),
            out
        )

    avg_infer = total_infer_time / len(rows)
    avg_pure = total_pure_time / len(rows)
    fps = 1000.0 / avg_infer

    # ---------------------------
    # Write CSV
    # ---------------------------
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image",
            "pure_inference_ms",
            "end_to_end_ms",
            "cpu_percent",
            "memory_mb",
            "num_detections",
            "avg_confidence",
            "model_size_mb",
        ])
        writer.writerows(rows)
        writer.writerow([])
        writer.writerow(["avg_pure_inference_ms", round(avg_pure, 3)])
        writer.writerow(["avg_end_to_end_ms", round(avg_infer, 3)])
        writer.writerow(["fps", round(fps, 2)])

    print("\n✅ Benchmark complete")
    print(f"Model size: {model_size_mb:.2f} MB")
    print(f"Avg pure inference: {avg_pure:.2f} ms")
    print(f"Avg end-to-end: {avg_infer:.2f} ms")
    print(f"FPS: {fps:.2f}")
    print(f"CSV saved to: {CSV_PATH}")

if __name__ == "__main__":
    main()
