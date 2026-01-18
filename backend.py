from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
from PIL import Image
import io
import base64
from ultralytics import YOLO
import tempfile
import os
from typing import Optional
import json
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Watermark Detector Pro API", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None

def load_model():
    """Load the YOLO model"""
    global model
    try:
        model_path = os.getenv("MODEL_PATH", "best.pt")
        logger.info(f"Attempting to load model from: {model_path}")
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at: {model_path}")
            return False
            
        model = YOLO(model_path)
        logger.info("✅ Model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return False

def image_to_base64(image_array):
    """Convert numpy array to base64 string"""
    try:
        # Convert BGR to RGB if needed
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_array)
        
        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        logger.error(f"Error converting image to base64: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("🚀 Starting Watermark Detector Pro API...")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Files in directory: {os.listdir('.')}")
    
    # Create necessary directories
    os.makedirs("uploads", exist_ok=True)
    logger.info("📁 Created uploads directory")
    
    # Load model
    model_loaded = load_model()
    if not model_loaded:
        logger.warning("⚠️ Model not loaded - some endpoints will not work")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    try:
        frontend_path = "frontend/index.html"
        if os.path.exists(frontend_path):
            with open(frontend_path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        else:
            logger.warning(f"Frontend file not found: {frontend_path}")
            return HTMLResponse(content="""
            <html>
                <head><title>Watermark Detector Pro</title></head>
                <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
                    <h1>🔍 Watermark Detector Pro</h1>
                    <p>API is running! Frontend files not found.</p>
                    <p><a href="/docs">View API Documentation</a></p>
                    <p><a href="/health">Check Health Status</a></p>
                </body>
            </html>
            """)
    except Exception as e:
        logger.error(f"Error serving root page: {e}")
        return HTMLResponse(content=f"""
        <html>
            <head><title>Watermark Detector Pro</title></head>
            <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
                <h1>🔍 Watermark Detector Pro</h1>
                <p>API is running!</p>
                <p>Error loading frontend: {str(e)}</p>
                <p><a href="/docs">View API Documentation</a></p>
            </body>
        </html>
        """)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        health_status = {
            "status": "healthy",
            "model_loaded": model is not None,
            "message": "Watermark Detector Pro API is running",
            "python_version": sys.version,
            "working_directory": os.getcwd(),
            "model_file_exists": os.path.exists("best.pt"),
            "frontend_exists": os.path.exists("frontend/index.html")
        }
        
        logger.info(f"Health check: {health_status}")
        return health_status
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "error",
            "model_loaded": False,
            "message": f"Health check failed: {str(e)}"
        }

@app.post("/detect")
async def detect_watermarks(
    file: UploadFile = File(...),
    confidence: float = 0.25
):
    """Detect watermarks in uploaded image"""
    
    if not model:
        logger.error("Model not loaded")
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Validate confidence
    if not 0.1 <= confidence <= 1.0:
        raise HTTPException(status_code=400, detail="Confidence must be between 0.1 and 1.0")
    
    try:
        logger.info(f"Processing image: {file.filename}, confidence: {confidence}")
        
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Run detection
        results = model.predict(
            source=img_array,
            conf=confidence,
            verbose=False
        )
        
        # Process results
        boxes = results[0].boxes
        num_detections = len(boxes) if boxes is not None else 0
        
        logger.info(f"Detection complete: {num_detections} detections found")
        
        # Create result image with detections
        result_img = results[0].plot()
        
        # Convert images to base64
        original_b64 = image_to_base64(img_array)
        result_b64 = image_to_base64(result_img)
        
        # Extract detection details
        detections = []
        if boxes is not None and len(boxes) > 0:
            for i, box in enumerate(boxes):
                detection = {
                    "id": i,
                    "confidence": float(box.conf[0]),
                    "class_id": int(box.cls[0]) if box.cls is not None else 0,
                    "bbox": box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                }
                detections.append(detection)
        
        return JSONResponse(content={
            "success": True,
            "num_detections": num_detections,
            "detections": detections,
            "original_image": original_b64,
            "result_image": result_b64,
            "confidence_used": confidence,
            "image_size": {
                "width": image.width,
                "height": image.height
            }
        })
        
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    if not model:
        return {"model_loaded": False, "error": "Model not loaded"}
    
    try:
        return {
            "model_loaded": True,
            "model_type": "YOLOv8",
            "model_file": "best.pt",
            "classes": getattr(model, 'names', {}),
            "input_size": getattr(model, 'imgsz', 640)
        }
    except Exception as e:
        logger.error(f"Model info error: {e}")
        return {"model_loaded": False, "error": str(e)}

# Mount static files (CSS, JS, images) if they exist
if os.path.exists("frontend/static"):
    app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
    logger.info("📁 Mounted static files")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("frontend/static/css", exist_ok=True)
    os.makedirs("frontend/static/js", exist_ok=True)
    os.makedirs("frontend/static/images", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)
    
    # Get port from environment (for cloud deployment)
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info("🚀 Starting Watermark Detector Pro API...")
    logger.info(f"🌐 Server will be available at: http://{host}:{port}")
    logger.info("📁 Make sure your model file 'best.pt' exists!")
    
    uvicorn.run(
        "backend:app",
        host=host,
        port=port,
        reload=False,  # Disable reload in production
        log_level="info",
        timeout_keep_alive=120,
        limit_concurrency=10  # Limit concurrent requests
    )