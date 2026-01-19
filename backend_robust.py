import os
import sys
import threading
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

print("Starting Watermark Detector Pro - Robust Version")
print(f"Working directory: {os.getcwd()}")
print(f"Files: {os.listdir('.')}")

app = FastAPI(title="Watermark Detector Pro", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
model_loading = False
model_error = None

def load_model_background():
    """Load model in background with better error handling"""
    global model, model_loading, model_error
    
    print("Starting model loading in background...")
    model_loading = True
    
    try:
        # Set environment variables to avoid display issues
        os.environ['DISPLAY'] = ':99'
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
        
        print("Importing required libraries...")
        import cv2
        print("OpenCV imported successfully")
        
        from ultralytics import YOLO
        print("YOLO imported successfully")
        
        if not os.path.exists("best.pt"):
            model_error = "Model file best.pt not found"
            print(f"Error: {model_error}")
            return
            
        print("Loading YOLO model...")
        
        # Try to load model with error handling
        try:
            model = YOLO("best.pt")
            print("Model loaded successfully!")
            model_error = None
            
            # Test the model with a dummy prediction to ensure it works
            print("Testing model with dummy data...")
            import numpy as np
            dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
            results = model.predict(dummy_image, verbose=False)
            print("Model test successful!")
            
        except Exception as model_load_error:
            model_error = f"Model loading failed: {str(model_load_error)}"
            print(f"Model loading error: {model_error}")
            model = None
        
    except ImportError as import_error:
        model_error = f"Import error: {str(import_error)}"
        print(f"Import error: {model_error}")
    except Exception as e:
        model_error = f"Unexpected error: {str(e)}"
        print(f"Unexpected error: {model_error}")
    finally:
        model_loading = False
        print(f"Model loading finished. Success: {model is not None}")

@app.on_event("startup")
async def startup():
    """Startup - start model loading in background"""
    print("FastAPI startup event triggered")
    
    # Start model loading in background thread
    thread = threading.Thread(target=load_model_background, daemon=True)
    thread.start()
    print("Model loading started in background thread")

@app.get("/")
async def root():
    """Root endpoint with detailed status"""
    if model:
        status = "Ready"
        status_class = "ready"
        status_message = "Model is ready for watermark detection!"
    elif model_loading:
        status = "Loading..."
        status_class = "loading"
        status_message = "Model is loading in the background..."
    else:
        status = "Error"
        status_class = "error"
        status_message = f"Error: {model_error}"
    
    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Watermark Detector Pro</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{ font-family: Arial, sans-serif; text-align: center; padding: 50px; background: #f5f5f5; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .status {{ padding: 20px; border-radius: 5px; margin: 20px 0; }}
            .ready {{ background: #d4edda; color: #155724; }}
            .loading {{ background: #fff3cd; color: #856404; }}
            .error {{ background: #f8d7da; color: #721c24; }}
            a {{ color: #007bff; text-decoration: none; margin: 0 10px; padding: 10px 15px; border: 1px solid #007bff; border-radius: 5px; display: inline-block; margin: 5px; }}
            a:hover {{ background: #007bff; color: white; text-decoration: none; }}
            .error-details {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 15px 0; text-align: left; font-family: monospace; font-size: 12px; }}
            .refresh-btn {{ background: #28a745; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 10px; }}
            .refresh-btn:hover {{ background: #218838; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 Watermark Detector Pro</h1>
            <div class="status {status_class}">
                <h2>Status: {status}</h2>
                <p>{status_message}</p>
                {f'<div class="error-details"><strong>Technical Details:</strong><br>{model_error}</div>' if model_error else ''}
            </div>
            
            <div>
                <a href="/docs">📚 API Documentation</a>
                <a href="/health">❤️ Health Check</a>
                <a href="/model-status">🧠 Model Status</a>
            </div>
            
            {f'<button class="refresh-btn" onclick="location.reload()">🔄 Refresh Status</button>' if model_loading else ''}
            
            <div style="margin-top: 30px; padding: 20px; background: #e9ecef; border-radius: 5px;">
                <h3>🛠️ Troubleshooting</h3>
                <p>If you see an error, this usually means:</p>
                <ul style="text-align: left; display: inline-block;">
                    <li>Missing system libraries (libxcb.so.1 error)</li>
                    <li>Model file issues</li>
                    <li>OpenCV display problems</li>
                </ul>
                <p>The app is still functional for API testing!</p>
            </div>
        </div>
        
        <script>
            // Auto-refresh if model is loading
            if ({str(model_loading).lower()}) {{
                setTimeout(() => location.reload(), 10000);
            }}
        </script>
    </body>
    </html>
    """)

@app.get("/health")
async def health():
    """Health check - ALWAYS returns healthy"""
    print("Health check called - responding immediately")
    
    return {
        "status": "healthy",
        "message": "Server is running",
        "model_status": "ready" if model else "loading" if model_loading else "error",
        "model_error": model_error,
        "server_functional": True
    }

@app.get("/model-status")
async def model_status():
    """Detailed model status"""
    return {
        "model_loaded": model is not None,
        "model_loading": model_loading,
        "model_error": model_error,
        "can_detect": model is not None,
        "model_file_exists": os.path.exists("best.pt"),
        "model_file_size": os.path.getsize("best.pt") if os.path.exists("best.pt") else 0,
        "environment": {
            "DISPLAY": os.environ.get("DISPLAY", "not set"),
            "QT_QPA_PLATFORM": os.environ.get("QT_QPA_PLATFORM", "not set")
        }
    }

@app.post("/detect")
async def detect(file: UploadFile = File(...), confidence: float = 0.25):
    """Detect watermarks"""
    
    if model_loading:
        raise HTTPException(503, "Model is still loading, please wait...")
    
    if not model:
        raise HTTPException(503, f"Model not available: {model_error}")
    
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(400, "Please upload an image file")
    
    try:
        print(f"Processing detection for {file.filename}")
        
        # Import here to avoid startup delays
        import numpy as np
        from PIL import Image
        import io
        import base64
        import cv2
        
        # Process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        img_array = np.array(image)
        
        # Run detection
        results = model.predict(source=img_array, conf=confidence, verbose=False)
        boxes = results[0].boxes
        num_detections = len(boxes) if boxes is not None else 0
        
        print(f"Detection complete: {num_detections} detections found")
        
        # Create result image
        result_img = results[0].plot()
        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        
        # Convert to base64
        def img_to_base64(img_array):
            pil_img = Image.fromarray(img_array)
            buffer = io.BytesIO()
            pil_img.save(buffer, format='PNG')
            return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"
        
        # Extract detection details
        detections = []
        if boxes is not None and len(boxes) > 0:
            for i, box in enumerate(boxes):
                detections.append({
                    "id": i,
                    "confidence": float(box.conf[0]),
                    "class_id": int(box.cls[0]) if box.cls is not None else 0,
                    "bbox": box.xyxy[0].tolist()
                })
        
        return {
            "success": True,
            "num_detections": num_detections,
            "detections": detections,
            "original_image": img_to_base64(img_array),
            "result_image": img_to_base64(result_img_rgb),
            "confidence_used": confidence,
            "image_size": {"width": image.width, "height": image.height}
        }
        
    except Exception as e:
        print(f"Detection error: {e}")
        raise HTTPException(500, f"Detection failed: {str(e)}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = "0.0.0.0"
    
    print(f"Starting server on {host}:{port}")
    print("Health check will respond immediately")
    print("Model will load in background with better error handling")
    
    uvicorn.run(
        "backend_robust:app",
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )