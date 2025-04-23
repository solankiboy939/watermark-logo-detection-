import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# Set Streamlit page config
st.set_page_config(page_title="Watermark Logo Detector")

# Title and description
st.title("🔍 Watermark Logo Detection App")
st.markdown("Upload an image and detect watermark logos using your custom-trained YOLOv8 model.")

# Load YOLO model (make sure best.pt is in the same folder or adjust the path)
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Run detection
    if st.button("Run Detection"):
        st.write("🧠 Detecting watermark logos...")

        # Convert PIL image to numpy array (RGB format)
        img_array = np.array(image)

        # Run prediction directly on numpy array
        results = model.predict(source=img_array, conf=0.25, save=False)

        # Show results
        if len(results[0].boxes) == 0:
            st.warning("🚫 No watermark logo detected.")
        else:
            # Plot detection and convert to RGB for Streamlit display
            detected_img_bgr = results[0].plot()
            detected_img_rgb = cv2.cvtColor(detected_img_bgr, cv2.COLOR_BGR2RGB)
            st.image(detected_img_rgb, caption="Detected Watermarks", use_container_width=True)
