import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import tempfile

# Set page configuration
st.set_page_config(
    page_title="AquaMark Pro - Watermark Detection Suite",
    page_icon=":mag:",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Gradient Background CSS
st.markdown("""
    <style>
        :root {
            --primary: #2563eb;
            --secondary: #4f46e5;
            --accent: #f59e0b;
            --text: #ffffff;
        }
        
        .main {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            min-height: 100vh;
        }
        
        .block-container {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 20px;
            padding: 2.5rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
            margin: 2rem auto;
            max-width: 800px;
        }
        
        .header {
            text-align: center;
            padding: 1.5rem 0;
            margin-bottom: 2rem;
        }
        
        h1 {
            color: var(--text);
            font-size: 2.8rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(45deg, #fff, #e0e7ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .upload-container {
            border: 2px dashed rgba(255, 255, 255, 0.3);
            border-radius: 15px;
            padding: 2.5rem;
            text-align: center;
            margin: 2rem 0;
            background: rgba(255, 255, 255, 0.1);
            transition: all 0.3s ease;
        }
        
        .upload-container:hover {
            border-color: var(--accent);
            background: rgba(255, 255, 255, 0.15);
        }
        
        .stButton>button {
            background: rgba(255, 255, 255, 0.9);
            color: var(--primary);
            border: none;
            padding: 12px 30px;
            border-radius: 10px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
            background: white !important;
        }
        
        .result-card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 2rem;
            margin: 2rem 0;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        }
        
        .confidence-slider .stSlider {
            margin: 1.5rem 0;
        }
        
        .footer {
            text-align: center;
            padding: 1.5rem;
            color: rgba(255, 255, 255, 0.8);
            font-size: 0.9rem;
            margin-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
    <div class="header">
        <h1>🌐 AquaMark Pro</h1>
        <p style="color: rgba(255, 255, 255, 0.9); font-size: 1.2rem;">
            Enterprise-Grade Watermark Recognition System
        </p>
    </div>
""", unsafe_allow_html=True)

# Load YOLOv8 model
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

# Upload Section
with st.container():
    st.markdown("""
        <div class="upload-container">
            <h3 style="color: white; margin-bottom: 1rem;">📤 Drag & Drop Media</h3>
            <p style="color: rgba(255, 255, 255, 0.8);">Supported formats: JPG, PNG | Max size: 5MB</p>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file:
    # File validation
    file_size = uploaded_file.size / (1024 * 1024)
    if file_size > 5:
        st.error("⚠️ File size exceeds maximum limit (5MB)")
    else:
        # Image preview
        image = Image.open(uploaded_file).convert("RGB")
        with st.expander("🖼️ UPLOAD PREVIEW", expanded=True):
            st.image(image, use_column_width=True)

        # Detection controls
        st.markdown("---")
        col1, col2 = st.columns([1, 2])
        with col1:
            confidence = st.slider(
                "Detection Sensitivity",
                min_value=0.1,
                max_value=1.0,
                value=0.25,
                step=0.01,
                help="Adjust model confidence threshold"
            )
        with col2:
            if st.button("🚀 Start Analysis", use_container_width=True):
                with st.spinner("Analyzing content for watermarks..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                        image.save(tmp.name)
                        temp_img_path = tmp.name

                    # Run detection
                    results = model.predict(source=temp_img_path, conf=confidence)
                    boxes = results[0].boxes
                    
                    # Display results
                    with st.container():
                        st.markdown("## 📊 Detection Report")
                        if len(boxes) == 0:
                            st.warning("No watermarks detected")
                        else:
                            st.success(f"**{len(boxes)}** watermarks identified")
                            bgr_img = results[0].plot()
                            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                            
                            # Result card styling
                            with st.container():
                                st.image(rgb_img, caption="Processed Analysis", use_column_width=True)
                                with st.expander("🔍 Technical Insights", expanded=False):
                                    st.json({
                                        "detections": len(boxes),
                                        "confidence_range": f"{min(boxes.conf.numpy()):.2%} - {max(boxes.conf.numpy()):.2%}",
                                        "image_resolution": results[0].orig_shape,
                                        "processing_time": f"{sum(results[0].speed.values()):.1f}ms"
                                    })

# Footer
st.markdown("""
    <div class="footer">
        © 2024 AquaMark Pro | Enterprise Watermark Detection Solution<br>
        v2.1.0 | ISO 27001 Certified System
    </div>
""", unsafe_allow_html=True)
