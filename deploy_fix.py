#!/usr/bin/env python3
"""
Deploy the comprehensive fix for libxcb error
"""

import subprocess
import sys
import shutil

def apply_comprehensive_fix():
    """Apply the comprehensive fix"""
    print("🔧 Applying comprehensive fix...")
    
    # Option 1: Use the robust backend
    print("Option 1: Use robust backend with better error handling")
    
    # Option 2: Keep current backend but update Dockerfile
    print("Option 2: Update Dockerfile with all required libraries")
    
    choice = input("Choose option (1 for robust backend, 2 for Dockerfile fix): ").strip()
    
    if choice == "1":
        # Use robust backend
        shutil.copy("backend_robust.py", "backend.py")
        print("✅ Switched to robust backend")
        return "Use robust backend with comprehensive error handling"
    else:
        # Keep current backend, Dockerfile is already updated
        print("✅ Using updated Dockerfile with all system libraries")
        return "Updated Dockerfile with comprehensive system libraries"

def deploy():
    """Deploy the fix"""
    try:
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", "Comprehensive fix for libxcb error - add all required libraries"], check=True)
        subprocess.run(["git", "push"], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Git failed: {e}")
        return False

def main():
    print("🔧 Comprehensive Fix for libxcb.so.1 Error")
    print("=" * 50)
    
    print("Current issue: Model loading fails with libxcb.so.1 error")
    print("Server works fine, but YOLO model can't load")
    print()
    
    fix_description = apply_comprehensive_fix()
    
    print(f"\n✅ Fix applied: {fix_description}")
    print()
    print("🚀 Deploy now?")
    
    deploy_choice = input("Deploy to Railway? (y/n): ").strip().lower()
    
    if deploy_choice == 'y':
        if deploy():
            print("\n🎉 Fix deployed!")
            print()
            print("⏱️ What happens next:")
            print("1. Railway rebuilds with new Dockerfile (3-4 minutes)")
            print("2. Container includes all required system libraries")
            print("3. Model loading should work correctly")
            print("4. Status changes from 'Error' to 'Loading...' to 'Ready'")
            print()
            print("🔍 Check your app in 5 minutes!")
        else:
            print("❌ Deployment failed")
    else:
        print("Fix ready but not deployed.")

if __name__ == "__main__":
    main()