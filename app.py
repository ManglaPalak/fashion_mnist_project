# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

import streamlit as st
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_option('deprecation.showWarnings', False)

# ---------------------------
# Load the trained CNN model
# ---------------------------
MODEL_PATH = "models/fashion_cnn.keras"

try:
    cnn = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Class names for Fashion-MNIST
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# ---------------------------
# Image preprocessing
# ---------------------------
def preprocess_image(img: Image.Image):
    # Convert to grayscale
    img = img.convert("L")
    # Make square
    size = max(img.size)
    new_img = Image.new("L", (size, size), 255)
    new_img.paste(img, ((size - img.width)//2, (size - img.height)//2))
    # Resize to 28x28
    new_img = new_img.resize((28, 28))
    # Convert to array
    arr = np.array(new_img, dtype="float32")
    # Invert if background is bright
    if arr.mean() > 127:
        arr = 255 - arr
    # Normalize
    arr /= 255.0
    # Add batch and channel dimension
    arr = arr.reshape(1, 28, 28, 1)
    return arr

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("👕 Fashion-MNIST Classifier")
st.write("Upload one or more images of fashion items (JPG/PNG)")

uploaded_files = st.file_uploader(
    "Choose image files", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if uploaded_files:
    st.write("### Uploaded Images & Predictions")
    cols = st.columns(len(uploaded_files))  # display images side by side

    for i, uploaded_file in enumerate(uploaded_files):
        img = Image.open(uploaded_file)
        x = preprocess_image(img)
        pred = cnn.predict(x)
        label = class_names[int(np.argmax(pred))]
        confidence = float(np.max(pred))

        # Display in column
        with cols[i]:
            st.image(img, caption=f"{label} ({confidence:.2f})", use_column_width=True)

