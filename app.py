import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import tempfile

# Set page configuration
st.set_page_config(
    page_title="Watermark Detector Pro",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# High-Contrast UI CSS
st.markdown("""
    <style>
        :root {
            --primary: #1E40AF;
            --secondary: #1E3A8A;
            --text: #1F2937;
            --background: #F3F4F6;
        }
        
        body {
            color: var(--text) !important;
            background-color: var(--background) !important;
        }
        
        .main {
            background: var(--background);
        }
        
        .block-container {
            background: linear-gradient(to right, rgb(248, 229, 173), rgb(194, 211, 238));
            border-radius: 12px;
            padding: 2rem;
            margin: 2rem auto;
            max-width: 800px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            border: 1px solid #E5E7EB;
        }
        
        h1, h2, h3 {
            color: var(--text) !important;
        }
        
        .upload-section {
            border: 2px dashed #E5E7EB;
            border-radius: 8px;
            padding: 2rem;
            text-align: center;
            margin: 1.5rem 0;
            background: #F9FAFB;
        }
        
        .stButton>button {
            background: var(--primary) !important;
            color: white !important;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 6px;
            font-weight: 600;
        }
        
        .footer {
            text-align: center;
            padding: 1.5rem;
            color: #6B7280;
            font-size: 0.875rem;
            margin-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1>🔍 Watermark Detector Pro</h1>
        <p style="color: #4B5563;">
            Professional Watermark Detection Solution
        </p>
    </div>
""", unsafe_allow_html=True)

# Load YOLOv8 model
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

# Upload Section
with st.container():
    uploaded_file = st.file_uploader(
        "📤 Upload an image (JPG, PNG, max 5MB)",
        type=["jpg", "jpeg", "png"],
        label_visibility="visible"
    )

if uploaded_file:
    # File validation
    file_size = uploaded_file.size / (1024 * 1024)
    if file_size > 5:
        st.error("⚠️ File exceeds 5MB limit")
    else:
        # Image preview
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Detection controls
        confidence = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.25,
            step=0.01,
            help="Adjust detection sensitivity level"
        )
        
        if st.button("🔍 Detect Watermarks", type="primary"):
            with st.spinner("Analyzing image..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    image.save(tmp.name)
                    temp_img_path = tmp.name

                # Run detection
                results = model.predict(source=temp_img_path, conf=confidence)
                boxes = results[0].boxes
                
                # Display results
                if len(boxes) == 0:
                    st.warning("No watermarks detected")
                else:
                    st.success(f"Detected {len(boxes)} watermark(s)")
                    st.image(
                        cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB),
                        caption="Detection Results",
                        use_column_width=True
                    )

# Footer
st.markdown("""
    <div class="footer">
        © 2024 Watermark Detector Pro | v2.0
    </div>
""", unsafe_allow_html=True)
