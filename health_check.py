#!/usr/bin/env python3
"""
Health Check Script for Watermark Detector Pro

This script checks if your deployment is working correctly.
"""

import requests
import sys
import time
from pathlib import Path

def check_local_health():
    """Check local deployment health"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ Local server is healthy!")
            print(f"   Status: {data.get('status')}")
            print(f"   Model loaded: {data.get('model_loaded')}")
            return True
        else:
            print(f"❌ Local server returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to local server: {e}")
        return False

def check_remote_health(url):
    """Check remote deployment health"""
    try:
        health_url = f"{url.rstrip('/')}/health"
        response = requests.get(health_url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Remote server ({url}) is healthy!")
            print(f"   Status: {data.get('status')}")
            print(f"   Model loaded: {data.get('model_loaded')}")
            return True
        else:
            print(f"❌ Remote server returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to remote server: {e}")
        return False

def test_detection_endpoint(base_url):
    """Test the detection endpoint with a sample image"""
    print(f"\n🧪 Testing detection endpoint at {base_url}...")
    
    # Create a simple test image (1x1 pixel PNG)
    test_image_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\nIDATx\x9cc\xf8\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00IEND\xaeB`\x82'
    
    try:
        files = {'file': ('test.png', test_image_data, 'image/png')}
        data = {'confidence': '0.5'}
        
        detect_url = f"{base_url.rstrip('/')}/detect"
        response = requests.post(detect_url, files=files, data=data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Detection endpoint working!")
            print(f"   Detections found: {result.get('num_detections', 0)}")
            print(f"   Confidence used: {result.get('confidence_used', 0)}")
            return True
        else:
            print(f"❌ Detection endpoint returned status {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('detail', 'Unknown error')}")
            except:
                print(f"   Response: {response.text[:200]}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Detection endpoint test failed: {e}")
        return False

def main():
    print("🏥 Watermark Detector Pro - Health Check")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        # Check remote URL
        url = sys.argv[1]
        print(f"🌐 Checking remote deployment: {url}")
        
        if check_remote_health(url):
            test_detection_endpoint(url)
        
    else:
        # Check local deployment
        print("🏠 Checking local deployment...")
        
        if check_local_health():
            test_detection_endpoint("http://localhost:8000")
        else:
            print("\n💡 To start local server, run:")
            print("   python backend.py")
    
    print("\n" + "=" * 50)
    print("Health check complete!")

if __name__ == "__main__":
    main()