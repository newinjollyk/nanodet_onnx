

import os
import cv2
import numpy as np
import onnxruntime as ort

from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType
)

# ================= CONFIG =================
FP32_MODEL = "/home/newin/Projects/nanodet_sign/onnx_formats/nanodet_decoded.onnx"
INT8_MODEL = "/home/newin/Projects/nanodet_sign/onnx_formats/nanodet_decoded_static_int8.onnx"

CALIB_IMG_DIR = "/home/newin/Projects/nanodet_sign/onnx_formats/calib_images/image"


INPUT_SIZE = 416
MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32)
STD  = np.array([57.375, 57.12, 58.395], dtype=np.float32)

MAX_CALIB_IMAGES = 50


# ================= CALIBRATION READER =================
class NanoDetCalibrationReader(CalibrationDataReader):
    def __init__(self, image_dir, input_name, max_images=50):
        self.input_name = input_name
        self.image_paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ][:max_images]
        self.idx = 0

        print(f"📸 Using {len(self.image_paths)} calibration images")

    def get_next(self):
        if self.idx >= len(self.image_paths):
            return None

        img = cv2.imread(self.image_paths[self.idx])
        img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32)
        img = (img - MEAN) / STD
        img = img.transpose(2, 0, 1)[None]

        self.idx += 1
        return {self.input_name: img}


# ================= MAIN =================
def main():
    print("🔧 Static INT8 quantization started")
    print(f"FP32 model: {FP32_MODEL}")

    sess = ort.InferenceSession(FP32_MODEL, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    print(f"🔑 Model input name: {input_name}")

    calib_reader = NanoDetCalibrationReader(
        CALIB_IMG_DIR,
        input_name,
        MAX_CALIB_IMAGES
    )

    quantize_static(
        model_input=FP32_MODEL,
        model_output=INT8_MODEL,
        calibration_data_reader=calib_reader,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8
    )

    fp32_size = os.path.getsize(FP32_MODEL) / (1024 * 1024)
    int8_size = os.path.getsize(INT8_MODEL) / (1024 * 1024)

    print("✅ Static INT8 quantization complete!")
    print(f"FP32 size : {fp32_size:.2f} MB")
    print(f"INT8 size : {int8_size:.2f} MB")
    print(f"Compression: {fp32_size / int8_size:.2f}x")


if __name__ == "__main__":
    main()
