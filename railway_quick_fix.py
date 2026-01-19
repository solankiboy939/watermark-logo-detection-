#!/usr/bin/env python3
"""
Railway Quick Fix - Multiple Solutions
"""

import os
import subprocess
import sys
import shutil

def fix_option_1_disable_healthcheck():
    """Option 1: Disable health check temporarily"""
    print("🔧 Option 1: Disabling health check...")
    
    # Copy the no-healthcheck config
    shutil.copy("railway_no_healthcheck.json", "railway.json")
    print("✅ Railway config updated to disable health check")
    
    return "Disabled health check - Railway will deploy without waiting for /health endpoint"

def fix_option_2_fast_start():
    """Option 2: Use fast-start backend"""
    print("🔧 Option 2: Using fast-start backend...")
    
    # Create Dockerfile for fast start
    dockerfile_content = '''
# Fast Start Dockerfile for Watermark Detector Pro
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1 \\
    PIP_ROOT_USER_ACTION=ignore

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend_fast_start.py .
COPY frontend/ ./frontend/
COPY best.pt .
RUN mkdir -p uploads

RUN pip install --upgrade pip --root-user-action=ignore
RUN pip install fastapi uvicorn python-multipart --root-user-action=ignore
RUN pip install Pillow numpy --root-user-action=ignore
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --root-user-action=ignore
RUN pip install opencv-python-headless --root-user-action=ignore
RUN pip install ultralytics --root-user-action=ignore

EXPOSE 8000

# No health check - server starts immediately
CMD ["python", "backend_fast_start.py"]
'''.strip()
    
    with open("Dockerfile.faststart", "w") as f:
        f.write(dockerfile_content)
    
    # Update railway config
    railway_config = '''{
  "build": {
    "builder": "DOCKERFILE",
    "dockerfilePath": "Dockerfile.faststart"
  },
  "deploy": {
    "startCommand": "python backend_fast_start.py",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 3
  }
}'''.strip()
    
    with open("railway.json", "w") as f:
        f.write(railway_config)
    
    print("✅ Created fast-start backend and Dockerfile")
    
    return "Fast-start mode - server starts immediately, model loads in background"

def fix_option_3_debug_mode():
    """Option 3: Use debug mode"""
    print("🔧 Option 3: Using debug mode...")
    
    # Run the debug setup
    try:
        subprocess.run([sys.executable, "debug_railway.py"], input="y\n", text=True, check=True)
        return "Debug mode activated - minimal app with detailed logging"
    except subprocess.CalledProcessError:
        return "Failed to activate debug mode"

def commit_and_push(message):
    """Commit and push changes"""
    try:
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", message], check=True)
        subprocess.run(["git", "push"], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Git operation failed: {e}")
        return False

def main():
    print("🚀 Railway Quick Fix - Health Check Issues")
    print("=" * 60)
    
    print("Your Railway deployment is failing health checks. Here are 3 quick fixes:")
    print()
    
    print("1. 🚫 Disable Health Check (Fastest)")
    print("   • Removes health check requirement")
    print("   • App deploys immediately when server starts")
    print("   • Good for testing if the app actually works")
    print()
    
    print("2. ⚡ Fast Start Mode (Recommended)")
    print("   • Server starts immediately")
    print("   • Model loads in background")
    print("   • Health check passes while model loads")
    print("   • Users see loading status")
    print()
    
    print("3. 🐛 Debug Mode")
    print("   • Minimal app with detailed logging")
    print("   • Helps identify the exact issue")
    print("   • Good for troubleshooting")
    print()
    
    while True:
        try:
            choice = input("Choose fix (1, 2, 3, or q to quit): ").strip().lower()
            
            if choice == 'q':
                print("👋 Goodbye!")
                break
            elif choice == '1':
                result = fix_option_1_disable_healthcheck()
                message = "Fix Railway health check - disable health check"
                break
            elif choice == '2':
                result = fix_option_2_fast_start()
                message = "Fix Railway health check - fast start mode"
                break
            elif choice == '3':
                result = fix_option_3_debug_mode()
                message = "Fix Railway health check - debug mode"
                break
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or q.")
                continue
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)
    
    if choice != 'q':
        print(f"\n✅ Applied fix: {result}")
        
        deploy = input("\nDeploy to Railway now? (y/n): ").strip().lower()
        
        if deploy == 'y':
            if commit_and_push(message):
                print("\n🚀 Fix deployed to Railway!")
                print("⏱️  Wait 3-5 minutes for deployment to complete")
                print("🔍 Check Railway dashboard for logs")
                
                if choice == '2':
                    print("\n💡 With fast-start mode:")
                    print("   • App should deploy successfully")
                    print("   • Visit /model-status to check model loading")
                    print("   • Model will be ready in 1-2 minutes after deployment")
                elif choice == '1':
                    print("\n💡 With health check disabled:")
                    print("   • App should deploy immediately")
                    print("   • Visit your app URL to test functionality")
                    print("   • Check /health endpoint for detailed status")
            else:
                print("❌ Failed to deploy. You can manually push with: git push")
        else:
            print("Fix applied but not deployed. Push manually when ready: git push")

if __name__ == "__main__":
    main()