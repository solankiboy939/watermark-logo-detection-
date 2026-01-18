#!/usr/bin/env python3
"""
Watermark Detector Pro - Deployment Helper

This script helps you deploy your watermark detection model to various cloud platforms.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_requirements():
    """Check if all required files exist"""
    required_files = [
        "best.pt",
        "backend.py",
        "frontend/index.html",
        "requirements.txt",
        "Dockerfile"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ All required files found!")
    return True

def check_model_size():
    """Check model file size"""
    model_path = Path("best.pt")
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"📊 Model size: {size_mb:.1f} MB")
        
        if size_mb > 100:
            print("⚠️  Warning: Large model file may cause deployment issues on some platforms")
            print("   Consider using model compression or cloud storage")
        
        return size_mb
    return 0

def test_local_deployment():
    """Test the application locally"""
    print("🧪 Testing local deployment...")
    
    try:
        # Test if dependencies are installed
        import fastapi
        import uvicorn
        import ultralytics
        print("✅ Core dependencies installed")
        
        # Test model loading
        from ultralytics import YOLO
        model = YOLO("best.pt")
        print("✅ Model loads successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Run: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False

def create_git_repo():
    """Initialize git repository if needed"""
    if not Path(".git").exists():
        print("📁 Initializing Git repository...")
        subprocess.run(["git", "init"], check=True)
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit - Watermark Detector Pro"], check=True)
        print("✅ Git repository created")
    else:
        print("✅ Git repository exists")

def deploy_railway():
    """Deploy to Railway"""
    print("\n🚂 Deploying to Railway...")
    print("1. Go to https://railway.app")
    print("2. Sign up/Login with GitHub")
    print("3. Click 'New Project' → 'Deploy from GitHub repo'")
    print("4. Select your repository")
    print("5. Railway will automatically detect and deploy!")
    print("\n✨ Your app will be live in a few minutes!")

def deploy_render():
    """Deploy to Render"""
    print("\n🎨 Deploying to Render...")
    print("1. Go to https://render.com")
    print("2. Sign up/Login with GitHub")
    print("3. Click 'New' → 'Web Service'")
    print("4. Connect your repository")
    print("5. Settings:")
    print("   - Environment: Docker")
    print("   - Build Command: docker build -t app .")
    print("   - Start Command: python backend.py")
    print("6. Click 'Create Web Service'")
    print("\n✨ Your app will be live in a few minutes!")

def deploy_flyio():
    """Deploy to Fly.io"""
    print("\n🪂 Deploying to Fly.io...")
    
    # Check if flyctl is installed
    try:
        subprocess.run(["flyctl", "version"], check=True, capture_output=True)
        print("✅ Fly CLI installed")
        
        print("Running deployment commands...")
        subprocess.run(["flyctl", "launch"], check=True)
        subprocess.run(["flyctl", "deploy"], check=True)
        print("✅ Deployed to Fly.io!")
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Fly CLI not installed")
        print("Install it from: https://fly.io/docs/hands-on/install-flyctl/")
        print("Then run: flyctl launch && flyctl deploy")

def deploy_docker():
    """Deploy with Docker locally"""
    print("\n🐳 Deploying with Docker...")
    
    try:
        # Build image
        print("Building Docker image...")
        subprocess.run(["docker", "build", "-t", "watermark-detector", "."], check=True)
        
        # Run container
        print("Starting container...")
        subprocess.run([
            "docker", "run", "-d", 
            "-p", "8000:8000", 
            "--name", "watermark-detector-app",
            "watermark-detector"
        ], check=True)
        
        print("✅ Docker container running!")
        print("🌐 Access your app at: http://localhost:8000")
        
    except subprocess.CalledProcessError:
        print("❌ Docker deployment failed")
        print("Make sure Docker is installed and running")

def main():
    print("🚀 Watermark Detector Pro - Deployment Helper")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check model size
    model_size = check_model_size()
    
    # Test local deployment
    if not test_local_deployment():
        print("\n❌ Local testing failed. Fix issues before deploying.")
        sys.exit(1)
    
    # Create git repo if needed
    create_git_repo()
    
    print("\n🌟 Choose your deployment platform:")
    print("1. 🚂 Railway (Recommended - Easy & Free)")
    print("2. 🎨 Render (Great Free Tier)")
    print("3. 🪂 Fly.io (Global Edge Deployment)")
    print("4. 🐳 Docker (Local/Self-hosted)")
    print("5. 📖 Show all deployment options")
    print("6. ❌ Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == "1":
                deploy_railway()
                break
            elif choice == "2":
                deploy_render()
                break
            elif choice == "3":
                deploy_flyio()
                break
            elif choice == "4":
                deploy_docker()
                break
            elif choice == "5":
                print("\n📖 All deployment options:")
                print("- Railway: Easiest, great free tier")
                print("- Render: 750 hours/month free")
                print("- Fly.io: Global edge, pay-as-you-go")
                print("- Heroku: Classic platform ($7/month)")
                print("- Vercel: Serverless (may have limitations)")
                print("- AWS/GCP/Azure: Enterprise options")
                print("\nSee DEPLOYMENT.md for detailed instructions!")
            elif choice == "6":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-6.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

if __name__ == "__main__":
    main()