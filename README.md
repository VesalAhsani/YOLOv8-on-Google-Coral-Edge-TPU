# YOLOv8-on-Google-Coral-Edge-TPU
## YOLOv8-CLS Model Conversion for Coral Edge TPU Deployment

End-to-end pipeline for converting trained YOLOv8 classification models (`.pt`) from PyTorch to TFLite format compatible with **Google Coral Dev Board (Edge TPU)**. This guide includes model export, quantization, Edge TPU compilation, and final deployment steps.

---

## 🧠 Use Case

Deploy lightweight, real-time driver behavior classification models trained using YOLOv8-CLS on Coral Edge TPU for in-vehicle applications. The steps here are useful for anyone converting YOLOv8 classification models for embedded AI.

---

## 🖥️ 1. HPC System Used

Due to lack of native Linux support on local machine, conversion was performed on a High-Performance Computing (HPC) system with the following specs:

| Component      | Specification                 |
| -------------- | ----------------------------- |
| GPU            | 1x NVIDIA GTX 1080 Ti (11 GB) |
| CPU Cores      | 6 (heavy-duty node)           |
| RAM            | 32 GB                         |
| Storage        | 200 GB                        |
| OS             | Ubuntu 20.04.2 Server         |

---

## 🐍 2. Virtual Environment Setup

```bash
python3 -m venv edgetpu_env
source edgetpu_env/bin/activate
```

Install dependencies and freeze the environment:

```bash
pip install -r requirements.txt
pip freeze > requirements.txt
```

Make sure to version-control `requirements.txt` for reproducibility.

---

## 📝 3. Prepare Calibration YAML File

Create a file called `data_calib.yaml` for calibration metadata. Use a small subset of your dataset for calibration (\~200–500 images).

```yaml
nc: <number_of_classes>
names: [class1, class2, ..., classN]
train: /path/to/sample/images
val: /path/to/sample/images
```

Use either `train` or `val`, just make sure paths point to a directory structure with subfolders per class.

---

## 📦 4. Export YOLOv8 Model to ONNX

```bash
yolo export \
  model=/path/to/model.pt \
  format=onnx \
  data=data_calib.yaml \
  imgsz=512
```

Replace `imgsz` with the input resolution you trained on (e.g., 224, 320, 480, etc.).

---

## 🔁 5. Convert ONNX to TensorFlow SavedModel

```bash
onnx2tf -i /path/to/model.onnx -o saved_model/
```

This will create a `saved_model/` directory for use in TFLite conversion.

---

## 🔄 6. Convert to Float32 / Float16 TFLite

```python
import tensorflow as tf

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
SAVED_MODEL_DIR       = "saved_model"           # your onnx2tf export
FLOAT32_TFLITE_PATH   = "model_float32.tflite"
FLOAT16_TFLITE_PATH   = "model_float16.tflite"
IMG_SIZE              = 512                    # your input resolution
# ──────────────────────────────────────────────────────────────────────────────


# 1) Load the SavedModel
print("🔄 Loading SavedModel…")
loaded = tf.saved_model.load(SAVED_MODEL_DIR)


# 2) Make a tf.function wrapper with a proper input signature
@tf.function(
    input_signature=[tf.TensorSpec([1, IMG_SIZE, IMG_SIZE, 3], tf.float32, name="images")]
)
def inference(images):
    # The loaded object is callable
    return loaded(images)  


# 3) Grab the concrete function
print("🔄 Tracing concrete function…")
concrete_func = inference.get_concrete_function()


def convert_float32():
    print("➡️  Converting to float32 TFLite…")
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    tflite_model = converter.convert()
    with open(FLOAT32_TFLITE_PATH, "wb") as f:
        f.write(tflite_model)
    print(f"✅ Float32 model saved to {FLOAT32_TFLITE_PATH}")


def convert_float16():
    print("➡️  Converting to float16 TFLite…")
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_fp16 = converter.convert()
    with open(FLOAT16_TFLITE_PATH, "wb") as f:
        f.write(tflite_fp16)
    print(f"✅ Float16 model saved to {FLOAT16_TFLITE_PATH}")


if __name__ == "__main__":
    convert_float32()
    convert_float16()
```

---

## 📉 7. Full Integer Quantization (INT8)

Use this Python script to quantize to full INT8:

```python
import tensorflow as tf
import yaml
import pathlib
import numpy as np
import random
from PIL import Image

# ─── CONFIG ────────────────────────────────────────────────────────────────
SAVED_MODEL_DIR = "saved_model"
OUTPUT_TFLITE     = "model_int8.tflite"
CALIB_SAMPLES     = 500    # total images to use for calibration
IMG_SIZE          = (512, 512)
# ────────────────────────────────────────────────────────────────────────────

# 1) Load your SavedModel
loaded = tf.saved_model.load(SAVED_MODEL_DIR)

# 2) Wrap in a tf.function so we can get a ConcreteFunction
@tf.function(input_signature=[tf.TensorSpec([1, *IMG_SIZE, 3], tf.float32)])
def model_fn(x):
    return loaded(x)

concrete_func = model_fn.get_concrete_function()

# 3) Setup TFLiteConverter for full-int quant
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.uint8
converter.inference_output_type = tf.uint8

# 4) Build a balanced representative dataset
with open("data_calib.yaml") as f:
    cfg = yaml.safe_load(f)
calib_root = pathlib.Path(cfg["train"])
# get only subdirectories (one per class)
class_dirs = [d for d in calib_root.iterdir() if d.is_dir()]

if not class_dirs:
    raise RuntimeError(f"No class subfolders found under {calib_root}")

# how many per class
per_class = max(1, CALIB_SAMPLES // len(class_dirs))

# gather samples
img_paths = []
for d in class_dirs:
    imgs = [p for p in d.iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png")]
    if not imgs:
        continue
    chosen = random.sample(imgs, min(per_class, len(imgs)))
    img_paths += chosen

# if we came up short or long, trim or pad (optional)
img_paths = img_paths[:CALIB_SAMPLES]

def representative_dataset():
    for img_path in img_paths:
        img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
        arr = np.array(img, dtype=np.float32) / 255.0
        # tflite expects [1, H, W, C]
        yield [np.expand_dims(arr, axis=0)]

converter.representative_dataset = representative_dataset

# 5) Convert & save
tflite_quant_model = converter.convert()
with open(OUTPUT_TFLITE, "wb") as f:
    f.write(tflite_quant_model)

print(f"✅ {OUTPUT_TFLITE} generated successfully with {len(img_paths)} calibration images!")

```

---

## ⚙️ 8. Compile with Edge TPU Compiler

```bash
edgetpu_compiler model_int8.tflite
```

Output:

* `model_int8_edgetpu.tflite`
* `model_int8_edgetpu.log`

Check log file to confirm all ops were mapped:

```bash
cat model_int8_edgetpu.log
```

You should see:

```
All operations mapped to Edge TPU
```

---

## 🚀 9. Deploy to Coral Dev Board

1. Transfer `model_int8_edgetpu.tflite` to Coral Dev Board via SCP or USB.
2. Use a Python inference script to run predictions.
3. Ensure camera/video input and TFLite runtime is set up correctly.

---

## ✅ Final Notes

* Keep model paths and class names confidential in public repos.
* Always verify that operations are TPU-compatible before deployment.
* INT8 models will perform significantly faster on Coral vs. float32/16.

---

## 📂 License

MIT or your license here.

---

## 🙌 Acknowledgments

Thanks to [Ultralytics](https://github.com/ultralytics/ultralytics) and [Google Coral](https://coral.ai) teams for tools and documentation.
