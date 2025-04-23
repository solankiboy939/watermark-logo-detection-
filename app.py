import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import cv2
import numpy as np

# Set Streamlit page configuration
st.set_page_config(page_title="Watermark Logo Detection", layout="centered")

st.title("🔍 Watermark Logo Detection App")
st.markdown("This app detects watermark logos in uploaded images using a custom-trained YOLOv8 model.")

# Load your trained YOLOv8 model
try:
    model = YOLO("best.pt")
except Exception as e:
    st.error(f"❌ Failed to load YOLOv8 model: {e}")
    st.stop()

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Run Detection"):
        st.write("🧠 Detecting watermark logos...")

        try:
            # Save image temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                image.save(tmp.name)
                temp_image_path = tmp.name

            # Run prediction
            results = model.predict(source=temp_image_path, conf=0.25, save=False)

            # Check if any detections were made
            if len(results[0].boxes) == 0:
                st.warning("🚫 No watermark logo detected.")
            else:
                # Draw detections on image
                img_with_boxes = results[0].plot()
                img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, caption="Detected Watermarks", use_column_width=True)

        except Exception as e:
            st.error(f"❌ Error during detection: {e}")
