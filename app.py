import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import cv2
import numpy as np

# Set Streamlit page config
st.set_page_config(page_title="Watermark Logo Detector")

# Title and description
st.title("🔍 Watermark Logo Detection App")
st.markdown("Upload an image and detect watermark logos using your custom-trained YOLOv8 model.")

# Load YOLO model
model = YOLO("best.pt")  # Ensure best.pt is uploaded to Streamlit Cloud

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Open and display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Run Detection"):
        st.write("🧠 Detecting watermark logos...")

        # Save image temporarily
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image.save(tmp.name)
            temp_image_path = tmp.name

        # Run detection on saved image
        results = model.predict(source=temp_image_path, conf=0.25, save=False)

        if len(results[0].boxes) == 0:
            st.warning("🚫 No watermark logo detected.")
        else:
            # Plot results (BGR), convert to RGB for Streamlit
            bgr_img = results[0].plot()
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            st.image(rgb_img, caption="Detected Watermarks", use_container_width=True)
