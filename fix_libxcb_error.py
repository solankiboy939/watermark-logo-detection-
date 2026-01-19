#!/usr/bin/env python3
"""
Fix libxcb.so.1 error - Missing system libraries
"""

import subprocess
import sys

def fix_dockerfile():
    """Apply the fix to Dockerfile"""
    print("🔧 Fixing Dockerfile to include missing system libraries...")
    
    # The Dockerfile has already been updated above
    print("✅ Dockerfile updated with required system libraries:")
    print("   - libxcb1 (the missing library)")
    print("   - libxrender1")
    print("   - libxext6") 
    print("   - libgl1-mesa-glx")
    print("   - libglib2.0-0")
    print("   - libgomp1")

def deploy_fix():
    """Deploy the fix"""
    try:
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", "Fix libxcb.so.1 error - add missing system libraries"], check=True)
        subprocess.run(["git", "push"], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Git failed: {e}")
        return False

def main():
    print("🔧 Fix libxcb.so.1 Error")
    print("=" * 30)
    
    print("The error you're seeing is:")
    print("❌ libxcb.so.1: cannot open shared object file")
    print()
    print("This happens because OpenCV needs certain system libraries")
    print("that weren't installed in the Docker container.")
    print()
    
    print("🎯 Fix applied:")
    print("✅ Added missing system libraries to Dockerfile")
    print("✅ Includes libxcb1 and other required libraries")
    print()
    
    fix_dockerfile()
    
    deploy = input("Deploy the fix now? (y/n): ").strip().lower()
    
    if deploy == 'y':
        print("\n🚀 Deploying fix...")
        if deploy_fix():
            print("✅ Fix deployed!")
            print()
            print("⏱️ What happens next:")
            print("1. Railway rebuilds your app (2-3 minutes)")
            print("2. New container includes required libraries")
            print("3. Model loading should work correctly")
            print("4. Status should change from 'Error' to 'Loading...' then 'Ready'")
            print()
            print("🔍 Refresh your app page in a few minutes!")
        else:
            print("❌ Deployment failed")
    else:
        print("Fix ready but not deployed. Run 'git push' when ready.")

if __name__ == "__main__":
    main()