import cv2
import numpy as np
import onnxruntime as ort

# ================= CONFIG =================
ONNX_PATH = "/home/newin/Projects/nanodet_sign/onnx_formats/nanodet_decoded2.onnx"
INPUT_SIZE = 416
NUM_CLASSES = 21
REG_MAX = 7
STRIDES = [8, 16, 32, 64]
SCORE_THRESH = 0.35
NMS_THRESH = 0.6

CLASS_NAMES = [
    "Keep left", "Keep right", "No U-turn", "No left turn",
    "No parking", "No right turn", "No stopping", "Parking",
    "Pedestrian Crossing", "Speed Limit -100-", "Speed Limit -30-",
    "Speed Limit -60-", "Stop Sign",
    "Traffic Light -Green-", "Traffic Light -Red-", "Traffic Light -Yellow-",
    "U-turn", "bike", "motobike", "person", "vehicle"
]

# ============== UTILS =====================
def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)

def nms(boxes, scores, iou_thresh):
    idxs = scores.argsort()[::-1]
    keep = []

    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)

        if idxs.size == 1:
            break

        ious = compute_iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_thresh]

    return keep

def compute_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    return inter / (area1 + area2 - inter + 1e-6)

# ============== DECODE =====================
def decode_nanodet(pred):
    boxes, scores, labels = [], [], []

    offset = 0
    for stride in STRIDES:
        feat_size = INPUT_SIZE // stride
        num_points = feat_size * feat_size

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

        grid_y, grid_x = np.divmod(np.where(keep)[0], feat_size)
        cx = (grid_x + 0.5) * stride
        cy = (grid_y + 0.5) * stride

        x1 = cx - dist[:, 0] * stride
        y1 = cy - dist[:, 1] * stride
        x2 = cx + dist[:, 2] * stride
        y2 = cy + dist[:, 3] * stride

        boxes.append(np.stack([x1, y1, x2, y2], axis=1))
        scores.append(cls_scores)
        labels.append(cls_labels)

    if not boxes:
        return [], [], []

    boxes = np.concatenate(boxes)
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)

    final_boxes, final_scores, final_labels = [], [], []

    for c in range(NUM_CLASSES):
        idx = labels == c
        if not np.any(idx):
            continue

        b = boxes[idx]
        s = scores[idx]

        keep = nms(b, s, NMS_THRESH)
        final_boxes.append(b[keep])
        final_scores.append(s[keep])
        final_labels.extend([c] * len(keep))

    if not final_boxes:
        return [], [], []

    return (
        np.concatenate(final_boxes),
        np.concatenate(final_scores),
        final_labels
    )

# ============== MAIN ======================
def main(image_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32)
    STD  = np.array([57.375, 57.12, 58.395], dtype=np.float32)

    inp = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32)
    inp = (inp - MEAN) / STD
    inp = inp.transpose(2, 0, 1)[None]


    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    pred = sess.run(None, {"data": inp})[0]

    boxes, scores, labels = decode_nanodet(pred)

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        x1 = int(x1 * w / INPUT_SIZE)
        y1 = int(y1 * h / INPUT_SIZE)
        x2 = int(x2 * w / INPUT_SIZE)
        y2 = int(y2 * h / INPUT_SIZE)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{CLASS_NAMES[label]} {score:.2f}"
        cv2.putText(img, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("NanoDet ONNX", img)
    cv2.waitKey(0)

if __name__ == "__main__":
    main("/home/newin/Projects/nanodet_sign/Nano_sign/yolodark_nano_dataset/test/images/000153.jpg")

'''
/home/newin/Projects/nanodet_sign/nanodet/tools/export_onnx.py

python /home/newin/Projects/nanodet_sign/nanodet/tools/export_onnx.py \
  --cfg /home/newin/Projects/nanodet_sign/workspace/traffic_sign/logs-2026-01-23-23-13-43/train_cfg.yml \
  --model /home/newin/Projects/nanodet_sign/workspace/traffic_sign/model_best/nanodet_model_best.pth \
  --out /home/newin/Projects/nanodet_sign/onnx_formats/nanodet_decoded.onnx \
  --input_shape 416,416

'''