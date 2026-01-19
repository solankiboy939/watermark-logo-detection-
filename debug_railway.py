#!/usr/bin/env python3
"""
Debug Railway Deployment Issues
"""

import os
import sys
import subprocess
import time

def create_minimal_backend():
    """Create a minimal backend that starts quickly for debugging"""
    minimal_backend = '''
import os
import sys
import logging
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Watermark Detector Debug", version="1.0.0")

@app.get("/")
async def root():
    return {"message": "Watermark Detector Pro - Debug Mode", "status": "running"}

@app.get("/health")
async def health_check():
    """Simple health check that always works"""
    logger.info("Health check called")
    
    health_info = {
        "status": "healthy",
        "message": "Debug mode - basic functionality working",
        "python_version": sys.version,
        "working_directory": os.getcwd(),
        "files_in_directory": os.listdir('.'),
        "environment_variables": {
            "PORT": os.getenv("PORT", "not set"),
            "HOST": os.getenv("HOST", "not set"),
            "PYTHONPATH": os.getenv("PYTHONPATH", "not set")
        }
    }
    
    logger.info(f"Health check response: {health_info}")
    return health_info

@app.get("/test-model")
async def test_model():
    """Test if model file exists and can be loaded"""
    try:
        model_path = "best.pt"
        model_exists = os.path.exists(model_path)
        
        if model_exists:
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            logger.info(f"Model file found: {file_size:.1f} MB")
            
            # Try to load the model
            try:
                from ultralytics import YOLO
                logger.info("Attempting to load YOLO model...")
                model = YOLO(model_path)
                logger.info("Model loaded successfully!")
                
                return {
                    "model_file_exists": True,
                    "model_size_mb": file_size,
                    "model_loaded": True,
                    "model_type": "YOLOv8",
                    "message": "Model is working correctly"
                }
            except Exception as e:
                logger.error(f"Model loading failed: {e}")
                return {
                    "model_file_exists": True,
                    "model_size_mb": file_size,
                    "model_loaded": False,
                    "error": str(e),
                    "message": "Model file exists but failed to load"
                }
        else:
            logger.error("Model file not found")
            return {
                "model_file_exists": False,
                "model_loaded": False,
                "message": "Model file 'best.pt' not found"
            }
            
    except Exception as e:
        logger.error(f"Model test failed: {e}")
        return {
            "error": str(e),
            "message": "Failed to test model"
        }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info("🚀 Starting Watermark Detector Pro - DEBUG MODE")
    logger.info(f"🌐 Server starting on: http://{host}:{port}")
    logger.info(f"📁 Working directory: {os.getcwd()}")
    logger.info(f"📂 Files in directory: {os.listdir('.')}")
    
    uvicorn.run(
        "backend_debug:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
'''.strip()
    
    with open("backend_debug.py", "w") as f:
        f.write(minimal_backend)
    
    print("✅ Created minimal debug backend")

def create_debug_dockerfile():
    """Create a debug Dockerfile that starts faster"""
    debug_dockerfile = '''
# Debug Dockerfile for Watermark Detector Pro
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1 \\
    PIP_ROOT_USER_ACTION=ignore

# Install curl for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy debug backend and model
COPY backend_debug.py .
COPY best.pt .

# Install minimal dependencies for debug
RUN pip install --upgrade pip --root-user-action=ignore
RUN pip install fastapi uvicorn --root-user-action=ignore

# Only install ML dependencies if we need to test model loading
RUN pip install ultralytics --root-user-action=ignore

# Expose port
EXPOSE 8000

# Simple health check with very long timeout
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run the debug application
CMD ["python", "backend_debug.py"]
'''.strip()
    
    with open("Dockerfile.debug", "w") as f:
        f.write(debug_dockerfile)
    
    print("✅ Created debug Dockerfile")

def update_railway_config():
    """Update Railway config to use debug dockerfile"""
    railway_config = '''{
  "build": {
    "builder": "DOCKERFILE",
    "dockerfilePath": "Dockerfile.debug"
  },
  "deploy": {
    "startCommand": "python backend_debug.py",
    "healthcheckPath": "/health",
    "healthcheckTimeout": 120,
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 3
  }
}'''.strip()
    
    with open("railway.json", "w") as f:
        f.write(railway_config)
    
    print("✅ Updated Railway config for debug mode")

def commit_debug_version():
    """Commit the debug version"""
    try:
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", "Add debug version to troubleshoot Railway deployment"], check=True)
        subprocess.run(["git", "push"], check=True)
        print("✅ Pushed debug version to Railway")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Git operation failed: {e}")
        return False

def main():
    print("🐛 Railway Debug Mode Setup")
    print("=" * 50)
    
    print("The health check is still failing. Let's create a debug version to see what's happening.")
    print()
    
    print("🔧 Creating debug files...")
    create_minimal_backend()
    create_debug_dockerfile()
    update_railway_config()
    
    print()
    print("🚀 Debug version features:")
    print("✅ Minimal FastAPI app that starts quickly")
    print("✅ Detailed logging and environment info")
    print("✅ Separate endpoint to test model loading")
    print("✅ Very long health check timeout (180s)")
    print("✅ Simple health check that always works")
    
    print()
    choice = input("Deploy debug version to Railway? (y/n): ").lower().strip()
    
    if choice == 'y':
        if commit_debug_version():
            print()
            print("🎯 Debug version deployed! Here's what to do:")
            print("1. Wait 3-5 minutes for Railway to deploy")
            print("2. Check Railway logs for detailed startup info")
            print("3. Visit your app URL to see if basic functionality works")
            print("4. Visit /health endpoint to see detailed system info")
            print("5. Visit /test-model endpoint to test model loading")
            print()
            print("💡 This will help us identify the exact issue!")
        else:
            print("❌ Failed to deploy debug version")
    else:
        print("Debug files created but not deployed.")
        print("You can manually deploy with: git add . && git commit -m 'debug' && git push")

if __name__ == "__main__":
    main()