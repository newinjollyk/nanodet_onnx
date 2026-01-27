import onnxruntime as ort
import cv2
import numpy as np

# ===============================
# CONFIG
# ===============================
ONNX_PATH = "nanodet_traffic_sign_416.onnx"
IMAGE_PATH = "test_images/test.jpg"
INPUT_SIZE = (416, 416)

# NanoDet normalization (ImageNet)
MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32)
STD  = np.array([57.375, 57.12, 58.395], dtype=np.float32)

# ===============================
# Load ONNX model
# ===============================
session = ort.InferenceSession(
    ONNX_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
print("Input name:", input_name)

print("Model outputs:")
for o in session.get_outputs():
    print(" ", o.name, o.shape)

# ===============================
# Load & preprocess image
# ===============================
img = cv2.imread(IMAGE_PATH)
assert img is not None, "Image not found!"

img = cv2.resize(img, INPUT_SIZE)
img = img.astype(np.float32)

# BGR normalization
img = (img - MEAN) / STD

# HWC → CHW → NCHW
img = img.transpose(2, 0, 1)
img = np.expand_dims(img, axis=0)

# ===============================
# Run inference
# ===============================
outputs = session.run(None, {input_name: img})

print("\nInference results:")
print("Number of outputs:", len(outputs))
for i, out in enumerate(outputs):
    print(f"Output {i} shape:", out.shape)
