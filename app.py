import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import tempfile

# Set page configuration
st.set_page_config(page_title="Watermark Logo Detector", page_icon="🔍", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f9f9f9;
        }
        .block-container {
            padding-top: 2rem;
        }
        h1 {
            color: #2c3e50;
        }
        .stButton>button {
            background-color: #2c3e50;
            color: white;
            border-radius: 8px;
            padding: 0.6em 1.5em;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #1a252f;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🔍 Watermark Logo Detector")
st.markdown("Detect watermark logos in uploaded images using your custom-trained **YOLOv8** model.")

# Load YOLOv8 model
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

# Upload image
uploaded_file = st.file_uploader("📤 Upload an image here", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Check file size
    file_size = uploaded_file.size / (1024 * 1024)
    if file_size > 5:
        st.warning("⚠️ File size is too large. Please upload an image smaller than 5MB.")
    else:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📷 Uploaded Image", use_container_width=True)

        # Confidence threshold slider
        confidence = st.slider("🎯 Confidence Threshold", 0.1, 1.0, 0.25, 0.01)

        col1, col2 = st.columns([1, 3])
        with col1:
            detect_btn = st.button("🚀 Run Detection")
        with col2:
            st.empty()

        if detect_btn:
            st.info("Processing image...")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                image.save(tmp.name)
                temp_img_path = tmp.name

            # Run YOLOv8 detection
            results = model.predict(source=temp_img_path, conf=confidence, save=False)

            boxes = results[0].boxes
            if len(boxes) == 0:
                st.warning("🚫 No watermark logos detected.")
            else:
                bgr_img = results[0].plot()
                rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                st.success(f"✅ Detected {len(boxes)} watermark logo(s).")
                st.image(rgb_img, caption="🎯 Detection Result", use_container_width=True)

# Footer
st.markdown("---")
st.markdown("<center><small>© 2025 Watermark Logo Detector | Built using YOLOv8 & Streamlit</small></center>", unsafe_allow_html=True)
