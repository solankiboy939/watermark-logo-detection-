import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import torch

# Set page title
st.set_page_config(page_title="Watermark Logo Detector")

st.title("🔍 Watermark Logo Detection App")
st.markdown("This app uses a custom-trained YOLOv8 model to detect watermark logos in images.")

# Load your trained YOLOv8 model
MODEL_PATH = "best.pt"  # Ensure best.pt is in the same folder as app.py

try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Failed to load YOLO model. Check if 'best.pt' exists and is a valid YOLOv8 model.\n\nError: {e}")
    st.stop()

# Upload image
uploaded_file = st.file_uploader("📁 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📸 Uploaded Image", use_container_width=True)

    if st.button("▶️ Run Detection"):
        st.info("🔎 Running watermark detection...")

        # Save the uploaded image to a temp file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            image.save(temp_file.name)
            temp_image_path = temp_file.name

        try:
            results = model.predict(source=temp_image_path, conf=0.25, save=False)
        except Exception as e:
            st.error(f"❌ Detection failed: {e}")
            st.stop()

        if not results or len(results[0].boxes) == 0:
            st.warning("🚫 No watermark logos detected.")
        else:
            # Visualize detections
            result_img = results[0].plot()
            result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            st.image(result_img_rgb, caption="✅ Detected Watermarks", use_container_width=True)
