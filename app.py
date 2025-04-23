import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# Set page title
st.set_page_config(page_title="Watermark Logo Detector")

st.title("🔍 Watermark Logo Detection App")
st.markdown("This app uses your custom-trained YOLOv8 model to detect watermark logos in images.")

# Load your trained YOLOv8 model
model = YOLO("best.pt")  # Ensure 'best.pt' is in your working directory (or correct relative path)

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display image
    image = Image.open(uploaded_file).convert("RGB")  # Ensure RGB
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Run Detection"):
        st.write("🧠 Detecting watermark logos...")

        # Convert PIL image to NumPy array (RGB format)
        img_array = np.array(image)

        # Run detection directly on NumPy array
        results = model.predict(source=img_array, conf=0.25, save=False)

        if len(results[0].boxes) == 0:
            st.warning("🚫 No watermark logo detected.")
        else:
            # Convert BGR to RGB for display
            bgr_img = results[0].plot()
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            st.image(rgb_img, caption="Detected Watermarks", use_column_width=True)
