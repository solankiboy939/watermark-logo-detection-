import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import uuid
import cv2
import numpy as np

# Set page title
st.set_page_config(page_title="Watermark Logo Detector")

st.title("🔍 Watermark Logo Detection App")
st.markdown("This app uses your custom-trained YOLOv8 model to detect watermark logos in images.")

# Load your trained YOLOv8 model
MODEL_PATH = r"C:\Users\shiva\OneDrive\Desktop\solanki\runs\detect\train\weights\best.pt"
model = YOLO(MODEL_PATH)

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display image
    image = Image.open(uploaded_file).convert("RGB")  # ensure it's RGB
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save uploaded image
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    unique_filename = f"{uuid.uuid4().hex}.jpg"
    img_path = os.path.join(upload_dir, unique_filename)
    image.save(img_path)  # Now it's always RGB-safe

    if st.button("Run Detection"):
        st.write("🧠 Detecting watermark logos...")

        # Run detection with fixed confidence
        results = model.predict(source=img_path, conf=0.25, save=False)

        if len(results[0].boxes) == 0:
            st.warning("🚫 No watermark logo detected.")
        else:
            # Convert BGR (from OpenCV) to RGB (for Streamlit)
            bgr_img = results[0].plot()
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            st.image(rgb_img, caption="Detected Watermarks", use_column_width=True)       dont save the uplods in the folder rest all is good
