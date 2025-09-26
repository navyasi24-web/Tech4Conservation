import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import gdown
import os
import matplotlib.pyplot as plt

# ====== Helper: download model from Google Drive ======
def download_model_from_drive(file_id, output_path, name):
    if not os.path.exists(output_path):
        with st.spinner(f"Downloading {name} model from Google Drive..."):
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, output_path, quiet=False)
            st.success(f"{name} model downloaded!")

# ====== Replace these with your actual Google Drive file IDs ======
cnn_file_id = "1YLRXJyc8jL7vDGIQ_HslUD6IeoRxUqoH"           # <-- your CNN .h5 file ID
mobilenet_file_id = "1YfhrMLJ0eqyyJPu1mezW7O4jUg7TuhSN"  # <-- your MobileNetV2 .h5 file ID

# Paths where the models will be saved locally
cnn_model_path = "seed_shape_model.h5"
mobilenet_model_path = "Mobilenet_tuned.h5"

# Download both models if not already downloaded
download_model_from_drive(cnn_file_id, cnn_model_path, "CNN")
download_model_from_drive(mobilenet_file_id, mobilenet_model_path, "MobileNetV2")

# Load models
with st.spinner("Loading models..."):
    cnn_model = load_model(cnn_model_path)
    mobilenet_model = load_model(mobilenet_model_path)
    st.success("Both models loaded successfully!")

# Class names
class_names = ['intact', 'broken', 'spotted', 'immature', 'skin-damaged']

# UI
st.title("🌱 Seed Shape Classification App")
st.write("Upload a seed image and choose a model to classify it.")

model_option = st.radio("Select Model", ['MobileNetV2', 'Custom CNN'])
uploaded_file = st.file_uploader("Upload a seed image", type=["jpg", "jpeg", "png"])

# Prediction
if uploaded_file is not None:
    image_data = Image.open(uploaded_file).convert("RGB")
    st.image(image_data, caption="Uploaded Image", use_column_width=True)

    # Resize according to model
    if model_option == 'MobileNetV2':
        img_resized = image_data.resize((224, 224))
    else:
        img_resized = image_data.resize((227, 227))

    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    if model_option == 'MobileNetV2':
        prediction = mobilenet_model.predict(img_array)[0]
    else:
        prediction = cnn_model.predict(img_array)[0]

    # Top prediction
    top_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.markdown(f"### 🧠 Prediction: `{top_class}` ({confidence:.2f}% confidence)")

    # Bar chart
    st.subheader("Class Probabilities")
    fig, ax = plt.subplots()
    ax.bar(class_names, prediction * 100)
    ax.set_ylabel("Confidence (%)")
    ax.set_ylim(0, 100)
    st.pyplot(fig)

    # Print all class probabilities
    st.subheader("Raw Scores")
    for i, prob in enumerate(prediction):
        st.write(f"**{class_names[i]}**: {prob*100:.2f}%")
