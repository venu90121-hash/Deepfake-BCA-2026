import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Load the trained brain
MODEL_PATH = 'models/deepfake_detector.h5'

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found! Please run train.py first.")
else:
    model = tf.keras.models.load_model(MODEL_PATH)

    st.set_page_config(page_title="Deepfake Detector", layout="centered")
    st.title("🛡️ Deepfake Detection System")
    st.info("BCA Project 2026 - Powered by MobileNetV2")

    uploaded_file = st.file_uploader("Upload a face image (JPG/PNG)...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Scanning this image...', use_container_width=True)
        
        # Preprocessing
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        with st.spinner('AI is analyzing...'):
            prediction = model.predict(img_array)[0][0]
        
        # LOGIC: Alphabetical (fake=0, real=1)
        # If score is closer to 0, it is Fake. If closer to 1, it is Real.
        if prediction < 0.5:
            confidence = (1 - prediction) * 100
            st.error(f"🚨 RESULT: This image is likely FAKE")
            st.warning(f"AI Confidence: {confidence:.2f}%")
        else:
            confidence = prediction * 100
            st.success(f"✅ RESULT: This image is likely REAL")
            st.info(f"AI Confidence: {confidence:.2f}%")