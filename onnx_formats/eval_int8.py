import os
import json
import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ================= CONFIG =================
ONNX_PATH = "/home/newin/Projects/nanodet_sign/onnx_formats/nanodet_decoded_static_int8.onnx"

IMAGE_DIR = "/home/newin/Projects/nanodet_sign/Nano_sign/yolodark_nano_dataset/test/images"
GT_JSON  = "/home/newin/Projects/nanodet_sign/Nano_sign/yolodark_nano_dataset/test/instances_test.json"
PRED_JSON = "/home/newin/Projects/nanodet_sign/onnx_formats/int8_outs/predictions_int8.json"

INPUT_SIZE = 416
NUM_CLASSES = 21
REG_MAX = 7
STRIDES = [8, 16, 32, 64]

SCORE_THRESH = 0.001   # VERY LOW for mAP
NMS_THRESH = 0.6
TOP_K = 100
MAX_FINAL = 100

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
    inter = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
    area1 = (box[2]-box[0]) * (box[3]-box[1])
    area2 = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
    return inter / (area1 + area2 - inter + 1e-6)

def nms(boxes, scores, thr):
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        iou = compute_iou(boxes[i], boxes[order[1:]])
        order = order[1:][iou < thr]
    return keep

# ================= DECODE (IDENTICAL TO INFERENCE) =================
def decode_nanodet(pred):
    boxes_all, scores_all, labels_all = [], [], []
    offset = 0

    for stride in STRIDES:
        feat = INPUT_SIZE // stride
        num_points = feat * feat

        cls_logits = pred[:, offset:offset+num_points, :NUM_CLASSES][0]
        reg_pred   = pred[:, offset:offset+num_points, NUM_CLASSES:][0]
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

        # TOP-K
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

        boxes_all.append(np.stack([x1,y1,x2,y2], axis=1))
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

    # FINAL CAP
    if len(scores) > MAX_FINAL:
        top = np.argsort(scores)[::-1][:MAX_FINAL]
        boxes = boxes[top]
        scores = scores[top]
        labels = [labels[i] for i in top]

    return boxes, scores, labels

# ================= MAIN =================
def main():
    coco_gt = COCO(GT_JSON)
    img_ids = coco_gt.getImgIds()

    session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    results = []

    for img_id in tqdm(img_ids):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(IMAGE_DIR, img_info["file_name"])

        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        inp = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32)
        inp = (inp - MEAN) / STD
        inp = inp.transpose(2,0,1)[None]

        pred = session.run(None, {input_name: inp})[0]
        boxes, scores, labels = decode_nanodet(pred)

        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box

            x = x1 * w / INPUT_SIZE
            y = y1 * h / INPUT_SIZE
            bw = (x2 - x1) * w / INPUT_SIZE
            bh = (y2 - y1) * h / INPUT_SIZE

            results.append({
                "image_id": img_id,
                "category_id": int(label + 1),
                "bbox": [float(x), float(y), float(bw), float(bh)],
                "score": float(score)
            })

    os.makedirs(os.path.dirname(PRED_JSON), exist_ok=True)
    with open(PRED_JSON, "w") as f:
        json.dump(results, f)

    print("\n📊 Running COCO mAP evaluation (INT8)...")
    coco_dt = coco_gt.loadRes(PRED_JSON)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == "__main__":
    main()
