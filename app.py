import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import tempfile

# Set page configuration
st.set_page_config(
    page_title="AquaMark Analyzer",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# High-Contrast UI CSS
st.markdown("""
    <style>
        :root {
            --primary: #1A56DB;
            --secondary: #1E429F;
            --accent: #3B82F6;
            --text: #111827;
            --background: #F9FAFB;
        }
        
        .main {
            background: var(--background);
            min-height: 100vh;
        }
        
        .block-container {
            background: white;
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            margin: 2rem auto;
            max-width: 800px;
            border: 1px solid #E5E7EB;
        }
        
        .header {
            text-align: center;
            padding: 1.5rem 0;
            margin-bottom: 1rem;
        }
        
        h1 {
            color: var(--text);
            font-size: 2.25rem;
            margin-bottom: 0.5rem;
            font-weight: 700;
        }
        
        .upload-section {
            border: 2px dashed #E5E7EB;
            border-radius: 8px;
            padding: 2rem;
            text-align: center;
            margin: 1.5rem 0;
            background: #F9FAFB;
            transition: all 0.2s ease;
        }
        
        .upload-section:hover {
            border-color: var(--accent);
            background: #F3F4F6;
        }
        
        .stButton>button {
            background: var(--primary);
            color: white !important;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 6px;
            font-weight: 600;
            transition: all 0.2s ease;
        }
        
        .stButton>button:hover {
            background: var(--secondary);
            transform: translateY(-1px);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        .result-card {
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            margin: 1.5rem 0;
            border: 1px solid #E5E7EB;
        }
        
        .stSlider {
            margin: 1rem 0;
        }
        
        .footer {
            text-align: center;
            padding: 1.5rem;
            color: #6B7280;
            font-size: 0.875rem;
            margin-top: 2rem;
        }
        
        .metric-box {
            background: #F3F4F6;
            border-radius: 6px;
            padding: 1rem;
            margin: 1rem 0;
            border: 1px solid #E5E7EB;
        }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
    <div class="header">
        <h1>🔍 AquaMark Analyzer</h1>
        <p style="color: #6B7280; margin-top: 0.5rem;">
            Professional Watermark Detection Solution
        </p>
    </div>
""", unsafe_allow_html=True)

# Load YOLOv8 model
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

# Upload Section
with st.container():
    st.markdown("""
        <div class="upload-section">
            <h3 style="color: var(--text); margin-bottom: 1rem;">📤 Upload Image</h3>
            <p style="color: #6B7280;">Supported formats: JPG, PNG | Max size: 5MB</p>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file:
    # File validation
    file_size = uploaded_file.size / (1024 * 1024)
    if file_size > 5:
        st.error("⚠️ File exceeds 5MB limit")
    else:
        # Image preview
        image = Image.open(uploaded_file).convert("RGB")
        with st.expander("🖼️ IMAGE PREVIEW", expanded=True):
            st.image(image, use_column_width=True)

        # Detection controls
        st.markdown("---")
        col1, col2 = st.columns([1, 2])
        with col1:
            confidence = st.slider(
                "Confidence Threshold",
                min_value=0.1,
                max_value=1.0,
                value=0.25,
                step=0.01,
                help="Adjust detection sensitivity level"
            )
        with col2:
            if st.button("🔍 Detect Watermarks", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                        image.save(tmp.name)
                        temp_img_path = tmp.name

                    # Run detection
                    results = model.predict(source=temp_img_path, conf=confidence)
                    boxes = results[0].boxes
                    
                    # Display results
                    with st.container():
                        st.markdown("## 📄 Results Summary")
                        if len(boxes) == 0:
                            st.warning("No watermarks detected")
                        else:
                            with st.container():
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.metric("Total Detections", len(boxes))
                                with col_b:
                                    st.metric("Average Confidence", f"{np.mean(boxes.conf.numpy()):.1%}")
                                
                                st.image(
                                    cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB),
                                    caption="Detection Visualization",
                                    use_column_width=True
                                )
                                
                                with st.expander("Technical Details", expanded=False):
                                    st.json({
                                        "image_dimensions": results[0].orig_shape,
                                        "processing_time": f"{sum(results[0].speed.values()):.1f}ms",
                                        "model_version": "YOLOv8n",
                                        "detection_areas": [box.xyxy.tolist() for box in boxes]
                                    })

# Footer
st.markdown("""
    <div class="footer">
        © 2024 AquaMark Analyzer | Enterprise-grade Detection System<br>
        v4.0.0 | WCAG 2.1 AA Compliant
    </div>
""", unsafe_allow_html=True)
