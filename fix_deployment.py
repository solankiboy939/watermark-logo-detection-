#!/usr/bin/env python3
"""
Deployment Fix Script for Watermark Detector Pro

This script helps fix common deployment issues.
"""

import os
import sys
import subprocess
from pathlib import Path

def fix_dockerfile():
    """Fix Dockerfile for different platforms"""
    print("🔧 Fixing Dockerfile for better compatibility...")
    
    # Check if lightweight version exists
    if Path("Dockerfile.lightweight").exists():
        print("✅ Using lightweight Dockerfile")
        if Path("Dockerfile").exists():
            os.rename("Dockerfile", "Dockerfile.backup")
        os.rename("Dockerfile.lightweight", "Dockerfile")
        print("✅ Dockerfile updated")
    else:
        print("❌ Lightweight Dockerfile not found")

def create_minimal_requirements():
    """Create minimal requirements for deployment"""
    minimal_reqs = """
# Core dependencies only
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
ultralytics==8.0.196
opencv-python-headless==4.8.1.78
Pillow==10.1.0
numpy==1.24.3
torch==2.1.1
""".strip()
    
    with open("requirements.minimal.txt", "w") as f:
        f.write(minimal_reqs)
    
    print("✅ Created minimal requirements file")
    print("💡 To use: cp requirements.minimal.txt requirements.txt")

def test_docker_build():
    """Test Docker build locally"""
    print("🐳 Testing Docker build...")
    
    try:
        # Build with lightweight dockerfile
        result = subprocess.run([
            "docker", "build", 
            "-f", "Dockerfile.lightweight",
            "-t", "watermark-detector-test", 
            "."
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Docker build successful!")
            return True
        else:
            print("❌ Docker build failed:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Docker build timed out")
        return False
    except FileNotFoundError:
        print("❌ Docker not installed")
        return False

def check_model_file():
    """Check if model file exists and is valid"""
    model_path = Path("best.pt")
    
    if not model_path.exists():
        print("❌ Model file 'best.pt' not found!")
        print("💡 Make sure your trained model is in the project directory")
        return False
    
    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"✅ Model file found ({size_mb:.1f} MB)")
    
    if size_mb > 500:
        print("⚠️  Warning: Very large model file may cause deployment issues")
        print("💡 Consider model compression or using cloud storage")
    
    return True

def create_railway_config():
    """Create optimized Railway configuration"""
    railway_config = {
        "build": {
            "builder": "DOCKERFILE",
            "dockerfilePath": "Dockerfile.lightweight"
        },
        "deploy": {
            "startCommand": "python backend.py",
            "healthcheckPath": "/health",
            "healthcheckTimeout": 60,
            "restartPolicyType": "ON_FAILURE",
            "restartPolicyMaxRetries": 3
        }
    }
    
    import json
    with open("railway.json", "w") as f:
        json.dump(railway_config, f, indent=2)
    
    print("✅ Created optimized Railway configuration")

def create_render_config():
    """Create optimized Render configuration"""
    render_config = """
services:
  - type: web
    name: watermark-detector-pro
    env: docker
    plan: starter
    dockerfilePath: ./Dockerfile.lightweight
    envVars:
      - key: PYTHONPATH
        value: /app
      - key: MODEL_PATH
        value: /app/best.pt
    healthCheckPath: /health
    autoDeploy: false
    buildFilter:
      paths:
      - backend.py
      - frontend/**
      - requirements.txt
      - Dockerfile.lightweight
      - best.pt
""".strip()
    
    with open("render.yaml", "w") as f:
        f.write(render_config)
    
    print("✅ Created optimized Render configuration")

def main():
    print("🔧 Watermark Detector Pro - Deployment Fix")
    print("=" * 50)
    
    # Check model file first
    if not check_model_file():
        print("\n❌ Cannot proceed without model file")
        sys.exit(1)
    
    print("\n🛠️  Available fixes:")
    print("1. 🐳 Fix Dockerfile compatibility issues")
    print("2. 📦 Create minimal requirements file")
    print("3. 🧪 Test Docker build locally")
    print("4. 🚂 Create optimized Railway config")
    print("5. 🎨 Create optimized Render config")
    print("6. 🔄 Apply all fixes")
    print("7. ❌ Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-7): ").strip()
            
            if choice == "1":
                fix_dockerfile()
            elif choice == "2":
                create_minimal_requirements()
            elif choice == "3":
                test_docker_build()
            elif choice == "4":
                create_railway_config()
            elif choice == "5":
                create_render_config()
            elif choice == "6":
                print("🔄 Applying all fixes...")
                fix_dockerfile()
                create_minimal_requirements()
                create_railway_config()
                create_render_config()
                print("✅ All fixes applied!")
                print("\n💡 Next steps:")
                print("1. Test locally: docker build -f Dockerfile.lightweight -t test .")
                print("2. Push to GitHub: git add . && git commit -m 'Fix deployment' && git push")
                print("3. Deploy to your chosen platform")
            elif choice == "7":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-7.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

if __name__ == "__main__":
    main()