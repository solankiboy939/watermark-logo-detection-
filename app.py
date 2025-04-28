import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import tempfile

# Set page configuration
st.set_page_config(
    page_title="AquaVision Pro",
    page_icon="🌊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Modern Gradient UI CSS
st.markdown("""
    <style>
        :root {
            --primary: #1a73e8;
            --secondary: #0d47a1;
            --accent: #00c853;
            --text: #2d3436;
        }
        
        .main {
            background: linear-gradient(145deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
        }
        
        .block-container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 2rem 3rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.05);
            margin: 2rem auto;
            max-width: 800px;
            border: 1px solid rgba(255, 255, 255, 0.3);
        }
        
        .header {
            text-align: center;
            padding: 2rem 0;
            margin-bottom: 2rem;
        }
        
        h1 {
            color: var(--text);
            font-size: 2.8rem;
            margin-bottom: 0.5rem;
            font-weight: 700;
            letter-spacing: -0.5px;
        }
        
        .upload-section {
            border: 2px dashed #1a73e8;
            border-radius: 15px;
            padding: 2.5rem;
            text-align: center;
            margin: 2rem 0;
            background: rgba(26, 115, 232, 0.03);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .upload-section:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(26, 115, 232, 0.1);
        }
        
        .stButton>button {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white !important;
            border: none;
            padding: 12px 30px;
            border-radius: 10px;
            font-weight: 600;
            transition: all 0.3s ease;
            width: 100%;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(26, 115, 232, 0.2);
        }
        
        .result-card {
            background: white;
            border-radius: 15px;
            padding: 2rem;
            margin: 2rem 0;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
            border: 1px solid rgba(0, 0, 0, 0.05);
        }
        
        .confidence-slider .stSlider {
            margin: 1.5rem 0;
        }
        
        .footer {
            text-align: center;
            padding: 1.5rem;
            color: #636e72;
            font-size: 0.9rem;
            margin-top: 2rem;
        }
        
        .metric-box {
            background: rgba(26, 115, 232, 0.08);
            border-radius: 12px;
            padding: 1rem;
            margin: 1rem 0;
        }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
    <div class="header">
        <h1>🌊 AquaVision Pro</h1>
        <p style="color: #636e72; font-size: 1.1rem; margin-top: 0.5rem;">
            Advanced Watermark Detection Platform
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
            <p style="color: #636e72;">Supported formats: JPG, PNG | Max size: 5MB</p>
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
        © 2024 AquaVision Pro | Enterprise-grade Detection System<br>
        v3.0.1 | ISO 27001 Certified | GDPR Compliant
    </div>
""", unsafe_allow_html=True)
