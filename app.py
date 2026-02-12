import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import requests
from streamlit_lottie import st_lottie
import time

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Deepfake Sentinel AI", page_icon="🛡️", layout="wide")

# Custom CSS for UI styling
st.markdown("""
    <style>
    .stApp { transition: background-color 0.4s ease; }
    .result-card {
        padding: 25px;
        border-radius: 20px;
        border: 2px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        margin-top: 20px;
        animation: slideUp 0.6s ease-out;
    }
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    div.stButton > button:first-child {
        width: 100%; border-radius: 12px; height: 3em;
        background-color: #7d33ff; color: white; font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. SAFE LOTTIE LOADER ---
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

# Load Animations with safety
lottie_main = load_lottieurl("https://lottie.host/9e0004f8-18e3-46c9-8356-027c62b2e85a/Osh08f4uU7.json")
lottie_scan = load_lottieurl("https://lottie.host/80a0302b-a633-4621-8868-b7161b96d911/lWvE5pXp7m.json")
lottie_success = load_lottieurl("https://lottie.host/362955f1-3316-430c-9975-9c9892183955/qQ6xX1q0XU.json")
lottie_warning = load_lottieurl("https://lottie.host/e660e737-2936-499b-9860-23429304387a/7f0iSNoA4m.json")

# --- 3. SIDEBAR ---
with st.sidebar:
    # SAFE CHECK: Only show if animation loaded
    if lottie_main:
        st_lottie(lottie_main, height=180, key="ai_icon")
    else:
        st.header("🛡️ Sentinel AI")
    
    st.title("Settings")
    is_dark = st.toggle("🌙 Dark Mode", value=True)
    if not is_dark:
        st.markdown("<style>.stApp {background-color: #f0f2f6; color: black;}</style>", unsafe_allow_html=True)
    
    st.divider()
    st.info("MobileNetV2 Engine Active")

# --- 4. MAIN UI ---
st.title("🛡️ Deepfake Sentinel AI")
st.write("Upload an image for forensic verification.")

col_upload, col_preview = st.columns([1, 1])

with col_upload:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if not uploaded_file and lottie_scan:
        st_lottie(lottie_scan, height=300, key="scanner")

if uploaded_file:
    image = Image.open(uploaded_file)
    with col_preview:
        st.image(image, caption="Uploaded Image", use_column_width=True)
        analyze_btn = st.button("🚀 START SCAN")

    if analyze_btn:
        progress_bar = st.progress(0)
        for p in range(0, 101, 10):
            time.sleep(0.05)
            progress_bar.progress(p)
            
        # --- PREDICTION ---
        try:
            # Load model (make sure path is correct)
            model = tf.keras.models.load_model('models/deepfake_detector.h5')
            
            # Preprocess
            img = image.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_reshape = img_array[np.newaxis, ...]
            
            prediction = model.predict(img_reshape)[0][0]
            
            # --- RESULTS ---
            st.divider()
            res_col, anim_col = st.columns([2, 1])
            
            with res_col:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                if prediction > 0.5:
                    st.header("✅ Result: AUTHENTIC")
                    st.write(f"Accuracy Score: {prediction*100:.2f}%")
                    st.balloons()
                else:
                    st.header("🚨 Result: FAKE")
                    st.write(f"Manipulation Prob: {(1-prediction)*100:.2f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with anim_col:
                # SAFE CHECK: Result animations
                if prediction > 0.5 and lottie_success:
                    st_lottie(lottie_success, height=150, key="success_anim")
                elif prediction <= 0.5 and lottie_warning:
                    st_lottie(lottie_warning, height=150, key="warn_anim")
                    
        except Exception as e:
            st.error(f"Prediction Error: {e}")
