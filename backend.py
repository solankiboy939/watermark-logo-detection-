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
        model = YOLO("best.pt")
        print("✅ Model loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

def image_to_base64(image_array):
    """Convert numpy array to base64 string"""
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

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    try:
        with open("frontend/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html>
            <body>
                <h1>Frontend not found</h1>
                <p>Please make sure the frontend files are in the 'frontend' directory.</p>
                <p>Run the application setup first.</p>
            </body>
        </html>
        """)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "message": "Watermark Detector Pro API is running"
    }

@app.post("/detect")
async def detect_watermarks(
    file: UploadFile = File(...),
    confidence: float = 0.25
):
    """Detect watermarks in uploaded image"""
    
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Validate confidence
    if not 0.1 <= confidence <= 1.0:
        raise HTTPException(status_code=400, detail="Confidence must be between 0.1 and 1.0")
    
    try:
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
        return {"model_loaded": False, "error": str(e)}

# Mount static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

if __name__ == "__main__":
    # Create frontend directory if it doesn't exist
    os.makedirs("frontend/static/css", exist_ok=True)
    os.makedirs("frontend/static/js", exist_ok=True)
    os.makedirs("frontend/static/images", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)
    
    # Get port from environment (for cloud deployment)
    port = int(os.getenv("PORT", 8000))
    
    print("🚀 Starting Watermark Detector Pro API...")
    print(f"🌐 Server will be available at: http://0.0.0.0:{port}")
    print("📁 Make sure your model file 'best.pt' exists!")
    
    uvicorn.run(
        "backend:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Disable reload in production
        log_level="info",
        timeout_keep_alive=120,
        limit_concurrency=10  # Limit concurrent requests
    )