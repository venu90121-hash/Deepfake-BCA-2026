import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import requests
from streamlit_lottie import st_lottie
import time

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Deepfake AI image DEDECTION (VISTAS)", 
    page_icon="🛡️", 
    layout="wide"
)

# --- 2. ADVANCED CSS (Laser Scan & Glassmorphism) ---
st.markdown("""
    <style>
    /* Laser Scan Animation */
    .scan-container {
        position: relative;
        overflow: hidden;
        border: 3px solid #00f2fe;
        border-radius: 15px;
    }
    .laser-line {
        position: absolute;
        top: 0; left: 0; width: 100%; height: 4px;
        background: #00f2fe;
        box-shadow: 0 0 15px 5px #00f2fe;
        z-index: 10;
        animation: scanMove 2s linear infinite;
    }
    @keyframes scanMove {
        0% { top: 0%; }
        50% { top: 100%; }
        100% { top: 0%; }
    }
    /* Smooth Result Card */
    .result-card {
        padding: 25px;
        border-radius: 15px;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.3s;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. SAFE ANIMATION LOADER ---
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200: return None
        return r.json()
    except: return None

# Fetching animations (Checks for NoneType safety)
lottie_main = load_lottieurl("https://lottie.host/9e0004f8-18e3-46c9-8356-027c62b2e85a/Osh08f4uU7.json")
lottie_processing = load_lottieurl("https://lottie.host/80a0302b-a633-4621-8868-b7161b96d911/lWvE5pXp7m.json")
lottie_success = load_lottieurl("https://lottie.host/362955f1-3316-430c-9975-9c9892183955/qQ6xX1q0XU.json")
lottie_fake = load_lottieurl("https://lottie.host/e660e737-2936-499b-9860-23429304387a/7f0iSNoA4m.json")
lottie_scanning_bg = load_lottieurl("https://lottie.host/880d8591-628a-4462-97b7-58a5e37340ec/9A45pW4vQ7.json")

# --- 4. SIDEBAR (Theme & Settings) ---
with st.sidebar:
    if lottie_main:
        st_lottie(lottie_main, height=180, key="side_ai")
    st.title("🛡️ VISTAS Lab Settings")
    
    # DARK/LIGHT MODE SWITCHER
    theme = st.toggle("🌙 Enable Dark Mode", value=True)
    if not theme:
        st.markdown("<style>.stApp { background: white !important; color: black !important; }</style>", unsafe_allow_html=True)
    
    st.divider()
    st.info("MobileNetV2 AI Engine Active")

# --- 5. MAIN INTERFACE ---
st.title("🛡️ Deepfake AI image DEDECTION (VISTAS)")
st.write("---")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("📂 Upload Evidence Image", type=["jpg", "png", "jpeg"])
    if not uploaded_file and lottie_scanning_bg:
        st_lottie(lottie_scanning_bg, height=350, key="bg_anim")

if uploaded_file:
    image = Image.open(uploaded_file)
    with col2:
        # LASER SCAN VISUAL
        st.markdown('<div class="scan-container"><div class="laser-line"></div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🔍 START FORENSIC ANALYSIS"):
            # Step-by-step Forensic Processing
            with st.status("🔬 Performing Forensic Layer Scan...") as status:
                st.write("Extracting noise patterns...")
                time.sleep(1)
                if lottie_processing: st_lottie(lottie_processing, height=100, key="proc")
                st.write("Verifying pixel consistency...")
                time.sleep(1)
                status.update(label="✅ Analysis Ready!", state="complete")

            # --- PREDICTION ---
            try:
                model = tf.keras.models.load_model('models/deepfake_detector.h5')
                img_resized = image.resize((224, 224))
                img_array = np.array(img_resized) / 255.0
                prediction = model.predict(img_array[np.newaxis, ...])[0][0]
                
                # --- ANIMATED RESULTS ---
                st.write("### Analysis Results")
                res_box, icon_box = st.columns([2, 1])
                
                with res_box:
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    if prediction > 0.5:
                        st.success(f"✅ **AUTHENTIC**")
                        st.write(f"Confidence Score: {prediction*100:.2f}%")
                        st.balloons()
                    else:
                        st.error(f"🚨 **DEEPFAKE DETECTED**")
                        st.write(f"Alert Level: {(1-prediction)*100:.2f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with icon_box:
                    # SAFETY CHECK FOR RESULTS ANIMATION
                    if prediction > 0.5 and lottie_success:
                        st_lottie(lottie_success, height=150, key="res_ok")
                    elif prediction <= 0.5 and lottie_fake:
                        st_lottie(lottie_fake, height=150, key="res_bad")
                        
            except Exception as e:
                st.error(f"Forensic Engine Error: {e}")
