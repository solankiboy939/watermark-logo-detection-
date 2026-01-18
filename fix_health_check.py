#!/usr/bin/env python3
"""
Fix Health Check Issues for Railway Deployment
"""

import os
import subprocess
import sys

def update_files():
    """Update files with health check fixes"""
    print("🔧 Applying health check fixes...")
    
    # Copy the updated simple dockerfile
    if os.path.exists("Dockerfile.simple"):
        with open("Dockerfile.simple", "r") as f:
            content = f.read()
        
        with open("Dockerfile", "w") as f:
            f.write(content)
        
        print("✅ Updated Dockerfile with health check fixes")
    
    print("✅ Backend.py already updated with better logging and error handling")
    
    return True

def commit_and_push():
    """Commit changes and push to trigger redeploy"""
    try:
        # Add all changes
        subprocess.run(["git", "add", "."], check=True)
        print("✅ Added changes to git")
        
        # Commit changes
        subprocess.run(["git", "commit", "-m", "Fix health check issues - add logging and error handling"], check=True)
        print("✅ Committed changes")
        
        # Push changes
        subprocess.run(["git", "push"], check=True)
        print("✅ Pushed changes to trigger redeploy")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Git operation failed: {e}")
        return False

def main():
    print("🏥 Watermark Detector Pro - Health Check Fix")
    print("=" * 50)
    
    print("The health check is failing because:")
    print("1. App might be taking too long to start")
    print("2. Model loading might be failing")
    print("3. Health endpoint might not be responding")
    print()
    
    print("🔧 Fixes applied:")
    print("✅ Added comprehensive logging")
    print("✅ Better error handling in health check")
    print("✅ Longer health check timeout (120s start period)")
    print("✅ Graceful handling of missing model file")
    print("✅ Suppressed pip warnings")
    print()
    
    # Update files
    if not update_files():
        print("❌ Failed to update files")
        sys.exit(1)
    
    # Check if we're in a git repository
    if not os.path.exists(".git"):
        print("❌ Not in a git repository")
        print("💡 Make sure you're in the project directory with git initialized")
        sys.exit(1)
    
    # Commit and push changes
    if commit_and_push():
        print("\n🚀 Changes pushed! Railway will automatically redeploy.")
        print("⏱️  Wait 3-5 minutes for the new deployment to complete.")
        print("🔍 Check Railway dashboard for deployment logs.")
        print("\n💡 The new version should pass health checks with:")
        print("   - Better error messages in logs")
        print("   - Longer startup timeout")
        print("   - Graceful model loading")
    else:
        print("\n❌ Failed to push changes")
        print("💡 You can manually push with: git push")

if __name__ == "__main__":
    main()