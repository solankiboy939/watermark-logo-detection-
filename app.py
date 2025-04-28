import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import tempfile

# Set page configuration
st.set_page_config(
    page_title="AquaMark Vision - Watermark Detection",
    page_icon=":magnifying_glass_tilted_left:",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
        :root {
            --primary: #2563eb;
            --secondary: #1d4ed8;
            --accent: #f59e0b;
            --background: #f8fafc;
            --text: #1e293b;
        }
        
        .main {
            background-color: var(--background);
        }
        
        .block-container {
            padding: 2rem 1rem;
            max-width: 800px;
        }
        
        .header {
            text-align: center;
            padding: 2rem 0;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            border-radius: 12px;
            margin-bottom: 2rem;
        }
        
        h1 {
            color: white;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        
        .subheader {
            color: rgba(255, 255, 255, 0.9);
            font-size: 1.1rem;
        }
        
        .upload-container {
            border: 2px dashed var(--primary);
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
            margin: 1.5rem 0;
            background: rgba(37, 99, 235, 0.05);
            transition: all 0.3s ease;
        }
        
        .upload-container:hover {
            border-color: var(--secondary);
            background: rgba(37, 99, 235, 0.1);
        }
        
        .stButton>button {
            width: 100%;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
        }
        
        .result-card {
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1.5rem 0;
            background: white;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        }
        
        .confidence-slider .stSlider {
            margin: 1rem 0;
        }
        
        .footer {
            text-align: center;
            padding: 1.5rem 0;
            color: #64748b;
            font-size: 0.9rem;
            margin-top: 2rem;
            border-top: 1px solid #e2e8f0;
        }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
    <div class="header">
        <h1>🌊 AquaMark Vision</h1>
        <div class="subheader">
            Advanced Watermark Detection powered by YOLOv8
        </div>
    </div>
""", unsafe_allow_html=True)

# Load YOLOv8 model
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

# Upload Section
with st.container():
    st.markdown("""
        <div class="upload-container">
            <h3>📤 Upload Media for Analysis</h3>
            <p>Supported formats: JPG, PNG | Max size: 5MB</p>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file:
    # File size validation
    file_size = uploaded_file.size / (1024 * 1024)
    if file_size > 5:
        st.error("⚠️ File size exceeds 5MB limit. Please upload a smaller file.")
    else:
        # Image preview
        image = Image.open(uploaded_file).convert("RGB")
        with st.expander("📷 Uploaded Image Preview", expanded=True):
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
                help="Adjust the sensitivity of the detection model"
            )
        with col2:
            if st.button("🔍 Start Detection", use_container_width=True):
                # Processing
                with st.spinner("Analyzing image for watermarks..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                        image.save(tmp.name)
                        temp_img_path = tmp.name

                    # Run detection
                    results = model.predict(source=temp_img_path, conf=confidence)
                    boxes = results[0].boxes
                    
                    # Display results
                    with st.container():
                        st.markdown("## 📝 Detection Results")
                        if len(boxes) == 0:
                            st.warning("No watermarks detected. Try adjusting the confidence threshold.")
                        else:
                            st.success(f"**{len(boxes)}** watermarks detected successfully!")
                            bgr_img = results[0].plot()
                            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                            st.image(rgb_img, caption="Processed Image with Detections", use_column_width=True)
                            
                            # Results summary
                            with st.expander("📊 Technical Details", expanded=False):
                                st.json({
                                    "detections": len(boxes),
                                    "average_confidence": f"{np.mean(boxes.conf.numpy()):.2%}",
                                    "image_dimensions": results[0].orig_shape
                                })

# Footer
st.markdown("""
    <div class="footer">
        © 2024 AquaMark Vision | Powered by YOLOv8 & Streamlit<br>
        Made with ❤️ by Computer Vision Experts
    </div>
""", unsafe_allow_html=True)
