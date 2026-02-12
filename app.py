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

# --- 2. THE BARCODE/LASER SCANNER CSS ---
# This adds a moving laser line over the image area
st.markdown("""
    <style>
    .scan-container {
        position: relative;
        overflow: hidden;
        border-radius: 15px;
        border: 2px solid #00f2fe;
    }
    .laser-line {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(to bottom, transparent, #00f2fe, transparent);
        box-shadow: 0 0 15px 5px rgba(0, 242, 254, 0.7);
        z-index: 10;
        animation: scan 2s linear infinite;
    }
    @keyframes scan {
        0% { top: 0%; }
        50% { top: 100%; }
        100% { top: 0%; }
    }
    .result-card {
        padding: 25px;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 242, 254, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. LOTTIE ASSETS ---
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except:
        return None

lottie_ai_scan = load_lottieurl("https://lottie.host/80a0302b-a633-4621-8868-b7161b96d911/lWvE5pXp7m.json") # QR/Barcode style
lottie_main = load_lottieurl("https://lottie.host/9e0004f8-18e3-46c9-8356-027c62b2e85a/Osh08f4uU7.json")

# --- 4. SIDEBAR ---
with st.sidebar:
    if lottie_main:
        st_lottie(lottie_main, height=180, key="side_anim")
    st.title("🛡️ VISTAS Lab")
    st.info("MobileNetV2 Neural Architecture")
    st.divider()
    st.write("2026 BCA Project")

# --- 5. MAIN UI ---
st.title("🛡️ Deepfake AI image DEDECTION (VISTAS)")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("📂 Drop image to scan...", type=["jpg", "png", "jpeg"])
    if not uploaded_file and lottie_ai_scan:
        st_lottie(lottie_ai_scan, height=350, key="idle_scan")

if uploaded_file:
    image = Image.open(uploaded_file)
    with col2:
        # We use a container to apply the "Laser Scan" CSS
        st.markdown('<div class="scan-container"><div class="laser-line"></div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🔍 SCAN FOR MANIPULATION"):
            with st.spinner("Decoding facial noise..."):
                time.sleep(2) # Visual pause for effect
                
                # --- PREDICTION ---
                try:
                    model = tf.keras.models.load_model('models/deepfake_detector.h5')
                    img = image.resize((224, 224))
                    img_array = np.array(img) / 255.0
                    img_batch = img_array[np.newaxis, ...]
                    
                    prediction = model.predict(img_batch)[0][0]
                    
                    # --- RESULTS ---
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    if prediction > 0.5:
                        st.balloons()
                        st.success(f"✅ **AUTHENTIC** ({prediction*100:.2f}%)")
                    else:
                        st.error(f"🚨 **FAKE DETECTED** ({(1-prediction)*100:.2f}%)")
                    st.markdown('</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error: {e}")
