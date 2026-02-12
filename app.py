import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import requests
from streamlit_lottie import st_lottie
import time

# --- 1. PAGE CONFIG & THEME ---
st.set_page_config(page_title="Deepfake Sentinel AI", page_icon="🛡️", layout="wide")

# Custom CSS for Glassmorphism and Smooth Animations
st.markdown("""
    <style>
    /* Main container transitions */
    .stApp {
        transition: background-color 0.4s ease;
    }
    
    /* Neon Glow for Result Cards */
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
    
    /* Button Hover Effects */
    div.stButton > button:first-child {
        width: 100%;
        border-radius: 12px;
        height: 3em;
        background-color: #7d33ff;
        color: white;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 10px 20px rgba(125, 51, 255, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# Function to fetch Lottie Animations
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except:
        return None

# --- 2. LOAD ANIMATION URLS ---
# AI Face Scan (Main Animation)
lottie_main = load_lottieurl("https://lottie.host/9e0004f8-18e3-46c9-8356-027c62b2e85a/Osh08f4uU7.json")
# Scanning Beam
lottie_scan = load_lottieurl("https://lottie.host/80a0302b-a633-4621-8868-b7161b96d911/lWvE5pXp7m.json")
# Success/Fake Icons
lottie_success = load_lottieurl("https://lottie.host/362955f1-3316-430c-9975-9c9892183955/qQ6xX1q0XU.json")
lottie_warning = load_lottieurl("https://lottie.host/e660e737-2936-499b-9860-23429304387a/7f0iSNoA4m.json")

# --- 3. SIDEBAR THEME TOGGLE ---
with st.sidebar:
    st_lottie(lottie_main, height=180, key="ai_icon")
    st.title("Sentinel Settings")
    
    # Toggle logic
    is_dark = st.toggle("🌙 Dark Mode", value=True)
    if not is_dark:
        st.markdown("<style>.stApp {background-color: #f0f2f6; color: black;}</style>", unsafe_allow_html=True)
    
    st.divider()
    st.info("Powered by MobileNetV2 Architecture & Streamlit 2026.")

# --- 4. MAIN USER INTERFACE ---
st.title("🛡️ Deepfake Sentinel AI")
st.subheader("High-Precision Facial Forensics Analysis")

col_upload, col_preview = st.columns([1, 1])

with col_upload:
    uploaded_file = st.file_uploader("Upload Image to Scan", type=["jpg", "png", "jpeg"])
    if not uploaded_file:
        st_lottie(lottie_scan, height=300)

if uploaded_file:
    with col_preview:
        image = Image.open(uploaded_file)
        st.image(image, caption="Source Image Uploaded", use_column_width=True)
        analyze_btn = st.button("🚀 PERFORM FORENSIC SCAN")

    if analyze_btn:
        # Progress Bar Simulation
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for percent in range(1, 101, 5):
            time.sleep(0.05)
            progress_bar.progress(percent)
            status_text.text(f"Extracting features: {percent}%")
        
        status_text.text("Applying MobileNetV2 Weights...")
        
        # --- MODEL PREDICTION LOGIC ---
        try:
            model = tf.keras.models.load_model('models/deepfake_detector.h5')
            
            # Preprocessing
            size = (224, 224)
            img_processed = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            img_array = np.asarray(img_processed) / 255.0
            img_reshape = img_array[np.newaxis, ...]
            
            prediction = model.predict(img_reshape)[0][0]
            
            # --- DISPLAY ANIMATED RESULTS ---
            st.divider()
            res_col, anim_col = st.columns([2, 1])
            
            with res_col:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                if prediction > 0.5:
                    st.header("✅ Result: AUTHENTIC")
                    st.write(f"Confidence Score: **{prediction*100:.2f}%**")
                    st.balloons()
                else:
                    st.header("🚨 Result: DEEPFAKE DETECTED")
                    st.write(f"Manipulation Probability: **{(1-prediction)*100:.2f}%**")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with anim_col:
                if prediction > 0.5:
                    st_lottie(lottie_success, height=150)
                else:
                    st_lottie(lottie_warning, height=150)
                    
        except Exception as e:
            st.error(f"Error Loading Model: {e}")
