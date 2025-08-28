# --- Task 2: Predict on Your Own Fashion Pieces -----------------------------
# Put your 10 square-ish photos (JPG/PNG) in:  my_fashion_pieces/
# Suggested: plain background, top-down photos, one item per image.

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

import tensorflow as tf

SAVED_MODEL_PATH = "models/fashion_cnn.keras"

# Check that the model file exists
import os
assert os.path.exists(SAVED_MODEL_PATH), f"Saved model not found: {SAVED_MODEL_PATH}"

# Load model
cnn = tf.keras.models.load_model(SAVED_MODEL_PATH)
cnn.summary()  # optional


# Class mapping used by Fashion-MNIST
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Where your raw pics live (change if needed)
IMG_DIR = Path("Lab2_photos")
IMG_DIR.mkdir(exist_ok=True)

# Optional: where a saved model lives if `cnn` is not in memory
SAVED_MODEL_DIR = Path("models/fashion_cnn")  # e.g., tf.keras.models.save_model(cnn, this_path)

# Sanity check
#assert IMG_DIR.exists(), f"Image folder not found: {IMG_DIR.resolve()}\nCreate it and add 10 images."

def make_square(img, target_size=28, pad_mode=cv2.BORDER_CONSTANT, pad_val=255):
    """
    Center-pad to make image square before resizing.
    pad_val=255 makes a white canvas (good for photos on light backgrounds).
    """
    h, w = img.shape[:2]
    side = max(h, w)
    top = (side - h) // 2
    bottom = side - h - top
    left = (side - w) // 2
    right = side - w - left
    squared = cv2.copyMakeBorder(img, top, bottom, left, right, pad_mode, value=pad_val)
    squared = cv2.resize(squared, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return squared

def preprocess_for_cnn(img_bgr, normalize=True, invert_if_bright_bg=True):
    """
    1) Convert to grayscale
    2) Make square + resize to 28x28
    3) Optionally invert if background is bright (to resemble Fashion-MNIST's dark background)
    4) Normalize to [0,1]
    5) Add channel dim: (28,28,1)
    """
    # to grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # square & resize
    gray = make_square(gray, target_size=28, pad_val=255)

    # Heuristic: if mean is very bright, invert so item becomes light on dark bg (FMNIST-like)
    if invert_if_bright_bg and gray.mean() > 127:
        gray = 255 - gray

    x = gray.astype("float32")
    if normalize:
        x /= 255.0

    # (1, 28, 28, 1) for prediction
    x = x.reshape(1, 28, 28, 1)
    return x, gray  # return both network input and the (possibly inverted) 28x28 grayscale for display

# Use the best model from Task 1 if it exists in memory; otherwise try loading
try:
    _ = cnn.summary()  # will raise if cnn isn't defined
    print("Using in-memory best model: cnn")
except Exception:
    import tensorflow as tf
    assert SAVED_MODEL_DIR.exists(), f"cnn not found in memory; saved model folder missing: {SAVED_MODEL_DIR}"
    cnn = tf.keras.models.load_model(SAVED_MODEL_DIR)
    print(f"Loaded model from: {SAVED_MODEL_DIR.resolve()}")

# Collect up to 10 images (sorted for reproducibility)
image_files = sorted([p for p in IMG_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])[:10]
assert len(image_files) == 10, f"Found {len(image_files)} files in {IMG_DIR}; please add exactly 10."

pred_labels = []
softmax_scores = []
display_tiles = []

for path in image_files:
    img_bgr = cv2.imread(str(path))
    assert img_bgr is not None, f"Could not read image: {path}"

    x_net, x_28x28 = preprocess_for_cnn(img_bgr)
    # Predict
    probs = cnn.predict(x_net, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_name = class_names[pred_idx]

    pred_labels.append(pred_name)
    softmax_scores.append(float(np.max(probs)))
    display_tiles.append((path.name, x_28x28, pred_name, float(np.max(probs))))

# Visualize all 10 side by side
cols = 10
fig, axes = plt.subplots(1, cols, figsize=(2.2*cols, 2.6))
for i, (fname, tile, pname, pscore) in enumerate(display_tiles):
    ax = axes[i]
    ax.imshow(tile, cmap="gray")
    ax.set_title(f"{pname}\n({pscore:.2f})", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(fname[:14] + ("…" if len(fname) > 14 else ""), fontsize=8)
plt.suptitle("Your 28×28 Grayscale Inputs • Predicted label (confidence)", fontsize=12, y=1.05)
plt.tight_layout()
plt.show()

print("Predicted labels (in file order):")
for f, pred, sc in zip(image_files, pred_labels, softmax_scores):
    print(f"- {f.name:20s}  →  {pred:12s}  (conf: {sc:.2f})")
