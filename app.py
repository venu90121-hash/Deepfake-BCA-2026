import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
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

# --- 2. ADVANCED STYLING (CSS) ---
st.markdown("""
    <style>
    /* Animated Gradient Background */
    .stApp {
        background: linear-gradient(-45deg, #0e1117, #1d2129, #0e1117);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        transition: background-color 0.4s ease;
    }
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Result Card Animation */
    .result-card {
        padding: 30px;
        border-radius: 25px;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        animation: fadeInUp 1s ease-out;
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(50px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. LOTTIE ASSETS LOADER ---
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except:
        return None

# Fetching specific AI/Security animations
lottie_main = load_lottieurl("https://lottie.host/9e0004f8-18e3-46c9-8356-027c62b2e85a/Osh08f4uU7.json") # AI Core
lottie_scanning = load_lottieurl("https://lottie.host/80a0302b-a633-4621-8868-b7161b96d911/lWvE5pXp7m.json") # Search Beam
lottie_detect = load_lottieurl("https://lottie.host/880d8591-628a-4462-97b7-58a5e37340ec/9A45pW4vQ7.json") # Face Data
lottie_real = load_lottieurl("https://lottie.host/362955f1-3316-430c-9975-9c9892183955/qQ6xX1q0XU.json") # Success
lottie_fake = load_lottieurl("https://lottie.host/e660e737-2936-499b-9860-23429304387a/7f0iSNoA4m.json") # Warning

# --- 4. SIDEBAR NAVIGATION ---
with st.sidebar:
    if lottie_main:
        st_lottie(lottie_main, height=200, key="sidebar_ai")
    st.title("🛡️ VISTAS Forensic Lab")
    
    # Theme Control
    theme = st.select_slider("Select UI Theme", options=["Deep Dark", "Cyber Light"])
    if theme == "Cyber Light":
        st.markdown("<style>.stApp {background: #fdfdfd !important; color: black !important;}</style>", unsafe_allow_html=True)
    
    st.divider()
    st.markdown("**Version:** 3.1.0 (Animated)")
    st.markdown("**Project:** BCA Record 2026")

# --- 5. MAIN CONTENT ---
st.title("🛡️ Deepfake AI image DEDECTION (VISTAS)")
st.write("---")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload Image to Inspect", type=["jpg", "png", "jpeg"])
    if not uploaded_file and lottie_detect:
        st_lottie(lottie_detect, height=350, key="initial_anim")

if uploaded_file:
    image = Image.open(uploaded_file)
    with col2:
        st.image(image, caption="Analyzed Specimen", use_column_width=True)
        if st.button("🔍 START DEEP ANALYSIS"):
            # Animated Processing Phase
            with st.status("🔬 Performing Forensic Layer Scan...", expanded=True) as status:
                st.write("Extracting noise patterns...")
                if lottie_scanning: st_lottie(lottie_scanning, height=150, key="scan_beam")
                time.sleep(1)
                st.write("Running MobileNetV2 Neural Network...")
                time.sleep(1)
                status.update(label="✅ Scan Complete!", state="complete", expanded=False)

            # --- PREDICTION LOGIC ---
            try:
                model = tf.keras.models.load_model('models/deepfake_detector.h5')
                img = image.resize((224, 224))
                img_array = np.array(img) / 255.0
                img_reshape = img_array[np.newaxis, ...]
                
                prediction = model.predict(img_reshape)[0][0]
                
                # --- ANIMATED RESULTS ---
                st.write("### Analysis Results")
                res_box, icon_box = st.columns([3, 1])
                
                with res_box:
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    if prediction > 0.5:
                        st.subheader("✅ Status: AUTHENTIC")
                        st.progress(prediction)
                        st.write(f"Confidence: **{prediction*100:.2f}%**")
                        st.balloons()
                    else:
                        st.subheader("🚨 Status: MANIPULATED")
                        st.progress(1 - prediction)
                        st.write(f"Detection Score: **{(1-prediction)*100:.2f}%**")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with icon_box:
                    if prediction > 0.5 and lottie_real:
                        st_lottie(lottie_real, height=150)
                    elif prediction <= 0.5 and lottie_fake:
                        st_lottie(lottie_fake, height=150)
                        
            except Exception as e:
                st.error(f"System Error: {e}")
