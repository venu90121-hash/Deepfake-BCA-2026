import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import requests
from streamlit_lottie import st_lottie
import time

# --- 1. PAGE CONFIGURATION ---
# This changes the name in the BROWSER TAB
st.set_page_config(
    page_title="Deepfake AI image DEDECTION (VISTAS)", 
    page_icon="🛡️", 
    layout="wide"
)

# --- 2. ADVANCED CSS STYLING ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(-45deg, #0e1117, #161b22, #0d1117);
        background-size: 400% 400%;
        animation: gradientBG 10s ease infinite;
    }
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .result-card {
        padding: 30px;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        animation: fadeIn 1.2s ease-in-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. SAFE LOTTIE ASSET LOADER ---
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except:
        return None

# Fetching high-quality animations
lottie_main = load_lottieurl("https://lottie.host/9e0004f8-18e3-46c9-8356-027c62b2e85a/Osh08f4uU7.json")
lottie_scan_loop = load_lottieurl("https://lottie.host/880d8591-628a-4462-97b7-58a5e37340ec/9A45pW4vQ7.json")
lottie_processing = load_lottieurl("https://lottie.host/80a0302b-a633-4621-8868-b7161b96d911/lWvE5pXp7m.json")
lottie_success = load_lottieurl("https://lottie.host/362955f1-3316-430c-9975-9c9892183955/qQ6xX1q0XU.json")
lottie_fake = load_lottieurl("https://lottie.host/e660e737-2936-499b-9860-23429304387a/7f0iSNoA4m.json")

# --- 4. SIDEBAR ---
with st.sidebar:
    if lottie_main:
        st_lottie(lottie_main, height=200, key="sidebar_anim")
    st.title("🛡️ VISTAS Lab Settings")
    theme_choice = st.radio("UI Theme", ["Dark Stealth", "Light Crystal"])
    if theme_choice == "Light Crystal":
        st.markdown("<style>.stApp { background: white !important; color: black !important; }</style>", unsafe_allow_html=True)
    st.divider()
    st.caption("Deepfake Detection Engine v3.5")

# --- 5. MAIN UI ---
st.title("🛡️ Deepfake AI image DEDECTION (VISTAS)")
st.write("---")

col_left, col_right = st.columns([1, 1])

with col_left:
    uploaded_file = st.file_uploader("📂 Upload Image Specimen", type=["jpg", "png", "jpeg"])
    if not uploaded_file and lottie_scan_loop:
        st_lottie(lottie_scan_loop, height=350, key="idle_anim")

if uploaded_file:
    img = Image.open(uploaded_file)
    with col_right:
        st.image(img, caption="Target Image", use_column_width=True)
        if st.button("⚡ EXECUTE FORENSIC SCAN"):
            # Progress handling
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for p in range(0, 101, 20):
                time.sleep(0.15)
                # Casting to float to prevent "invalid type" errors
                progress_bar.progress(float(p / 100))
                status_text.text(f"Scanning Pixels... {p}%")
            
            if lottie_processing:
                st_lottie(lottie_processing, height=150, key="proc_anim")
            
            # --- AI PREDICTION LOGIC ---
            try:
                model = tf.keras.models.load_model('models/deepfake_detector.h5')
                
                # Image Preprocessing
                img_resized = img.resize((224, 224))
                img_array = np.array(img_resized) / 255.0
                img_batch = img_array[np.newaxis, ...]
                
                prediction = model.predict(img_batch)[0][0]
                
                # --- DISPLAY ANIMATED RESULTS ---
                st.write("### Analysis Conclusion")
                res_box, icon_box = st.columns([2, 1])
                
                with res_box:
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    if prediction > 0.5:
                        st.success("✅ **STATUS: AUTHENTIC**")
                        st.write(f"The AI is **{float(prediction*100):.2f}%** confident this is original.")
                        st.balloons()
                    else:
                        st.error("🚨 **STATUS: MANIPULATED (FAKE)**")
                        st.write(f"Alert: **{float((1-prediction)*100):.2f}%** probability of manipulation.")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with icon_box:
                    if prediction > 0.5 and lottie_success:
                        st_lottie(lottie_success, height=150)
                    elif prediction <= 0.5 and lottie_fake:
                        st_lottie(lottie_fake, height=150)
            
            except Exception as e:
                st.error(f"Prediction Error: {e}")
