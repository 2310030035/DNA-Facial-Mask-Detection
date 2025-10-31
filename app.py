import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("mask_detector.h5")

st.title("😷 DNA Facial Mask Detection")
st.write("Upload an image to check whether the person is wearing a mask or not.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image for prediction
    img = img.resize((224, 224))  # Adjust this based on your training input size
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    prediction = model.predict(x)
    result = "Mask Detected" if prediction[0][0] < 0.5 else "No Mask Detected"

    st.subheader(f"Prediction: {result}")
