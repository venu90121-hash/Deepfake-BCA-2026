import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import requests
from streamlit_lottie import st_lottie
import time

# --- 1. PAGE CONFIG ---
st.set_page_config(
    page_title="Deepfake AI image DEDECTION (VISTAS)", 
    page_icon="🛡️", 
    layout="wide"
)

# --- 2. CSS FOR SCANNER & RESULTS ---
st.markdown("""
    <style>
    .scan-container {
        position: relative;
        overflow: hidden;
        border: 3px solid #00f2fe;
        border-radius: 15px;
        box-shadow: 0 0 20px rgba(0, 242, 254, 0.5);
    }
    .laser {
        position: absolute;
        top: 0; left: 0; width: 100%; height: 5px;
        background: rgba(0, 242, 254, 0.8);
        box-shadow: 0 0 15px 5px #00f2fe;
        z-index: 10;
        animation: scanning 2s linear infinite;
    }
    @keyframes scanning {
        0% { top: 0%; }
        50% { top: 100%; }
        100% { top: 0%; }
    }
    .result-card {
        padding: 20px;
        border-radius: 15px;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. ANIMATION LOADER ---
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except: return None

# Specific Cyber/Forensic Animations
lottie_main = load_lottieurl("https://lottie.host/9e0004f8-18e3-46c9-8356-027c62b2e85a/Osh08f4uU7.json")
lottie_success = load_lottieurl("https://lottie.host/362955f1-3316-430c-9975-9c9892183955/qQ6xX1q0XU.json") # Smooth Checkmark
lottie_fake = load_lottieurl("https://lottie.host/e660e737-2936-499b-9860-23429304387a/7f0iSNoA4m.json") # Alert/Warning

# --- 4. SIDEBAR ---
with st.sidebar:
    if lottie_main: st_lottie(lottie_main, height=180, key="side_ai")
    st.title("🛡️ VISTAS Lab")
    st.write("Deep Learning Engine v4.0")
    st.divider()

# --- 5. MAIN UI ---
st.title("🛡️ Deepfake AI image DEDECTION (VISTAS)")
st.write("---")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("📂 Upload Evidence Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    with col2:
        # Visual Laser Scan Container
        st.markdown('<div class="scan-container"><div class="laser"></div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🔍 INITIATE FORENSIC SCAN"):
            with st.status("🔍 Analyzing pixels and noise patterns...") as status:
                time.sleep(2) # Simulated analysis time
                
                try:
                    # Model Inference
                    model = tf.keras.models.load_model('models/deepfake_detector.h5')
                    img_ready = np.array(image.resize((224, 224))) / 255.0
                    pred = model.predict(img_ready[np.newaxis, ...])[0][0]
                    status.update(label="✅ Analysis Complete", state="complete")

                    # Animated Results Section
                    st.write("### Forensic Conclusion")
                    res_col, anim_col = st.columns([2, 1])
                    
                    with res_col:
                        st.markdown('<div class="result-card">', unsafe_allow_html=True)
                        if pred > 0.5:
                            st.success(f"**AUTHENTIC**")
                            st.write(f"Confidence: {pred*100:.2f}%")
                            st.balloons()
                        else:
                            st.error(f"**FAKE / MANIPULATED**")
                            st.write(f"Alert Level: {(1-pred)*100:.2f}%")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with anim_col:
                        if pred > 0.5: st_lottie(lottie_success, height=150)
                        else: st_lottie(lottie_fake, height=150)
                        
                except Exception as e:
                    st.error(f"Error: {e}")
