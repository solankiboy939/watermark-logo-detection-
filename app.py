import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import tempfile

# Set page title
st.set_page_config(page_title="Watermark Logo Detector")

st.title("🔍 Watermark Logo Detection App")
st.markdown("This app uses your custom-trained YOLOv8 model to detect watermark logos in images.")

# Load your trained YOLOv8 model
MODEL_PATH = "best.pt"
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model: {e}")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display image
    image = Image.open(uploaded_file).convert("RGB")  # ensure it's RGB
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Add confidence threshold slider
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25)

    if st.button("Run Detection"):
        st.write("🧠 Detecting watermark logos...")

        # Save temporarily to run detection
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            temp_img_path = tmp.name

        with st.spinner('Detecting watermark logos...'):
            # Run detection with dynamic confidence
            results = model.predict(source=temp_img_path, conf=conf_threshold, save=False)

        if len(results[0].boxes) == 0:
            st.warning("🚫 No watermark logo detected.")
        else:
            # Convert BGR (from OpenCV) to RGB (for Streamlit)
            bgr_img = results[0].plot()
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            st.image(rgb_img, caption="Detected Watermarks", use_container_width=True)
