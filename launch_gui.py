#!/usr/bin/env python3
"""
Watermark Detector Pro - GUI Launcher

This script allows you to choose between different GUI implementations:
1. Modern Web App (FastAPI + HTML/CSS/JS) - Recommended
2. CustomTkinter (Modern, lightweight)
3. PyQt6 (Professional, feature-rich)
4. Streamlit (Web-based, original)
"""

import sys
import subprocess
import importlib.util
import webbrowser
import time
import threading

def check_package(package_name):
    """Check if a package is installed"""
    spec = importlib.util.find_spec(package_name)
    return spec is not None

def install_package(package_name):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        return False

def launch_web_app():
    """Launch Modern Web App (FastAPI)"""
    if not check_package("fastapi"):
        print("FastAPI not found. Installing...")
        if not install_package("fastapi"):
            print("Failed to install FastAPI. Please install manually: pip install fastapi uvicorn[standard]")
            return False
    
    if not check_package("uvicorn"):
        print("Uvicorn not found. Installing...")
        if not install_package("uvicorn[standard]"):
            print("Failed to install Uvicorn. Please install manually: pip install uvicorn[standard]")
            return False
    
    try:
        print("🚀 Starting Modern Web Application...")
        print("📱 The app will open in your browser automatically")
        print("🌐 URL: http://localhost:8000")
        print("⏹️  Press Ctrl+C to stop the server")
        print()
        
        # Open browser after a short delay
        def open_browser():
            time.sleep(2)
            webbrowser.open("http://localhost:8000")
        
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Start the FastAPI server
        subprocess.run([sys.executable, "backend.py"])
        return True
    except Exception as e:
        print(f"Error launching Web App: {e}")
        return False

def launch_customtkinter():
    """Launch CustomTkinter GUI"""
    if not check_package("customtkinter"):
        print("CustomTkinter not found. Installing...")
        if not install_package("customtkinter"):
            print("Failed to install CustomTkinter. Please install manually: pip install customtkinter")
            return False
    
    try:
        from watermark_gui import WatermarkDetectorGUI
        app = WatermarkDetectorGUI()
        app.run()
        return True
    except Exception as e:
        print(f"Error launching CustomTkinter GUI: {e}")
        return False

def launch_pyqt():
    """Launch PyQt6 GUI"""
    if not check_package("PyQt6"):
        print("PyQt6 not found. Installing...")
        if not install_package("PyQt6"):
            print("Failed to install PyQt6. Please install manually: pip install PyQt6")
            return False
    
    try:
        from watermark_gui_qt import main
        main()
        return True
    except Exception as e:
        print(f"Error launching PyQt6 GUI: {e}")
        return False

def launch_streamlit():
    """Launch Streamlit web app"""
    if not check_package("streamlit"):
        print("Streamlit not found. Installing...")
        if not install_package("streamlit"):
            print("Failed to install Streamlit. Please install manually: pip install streamlit")
            return False
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
        return True
    except Exception as e:
        print(f"Error launching Streamlit: {e}")
        return False

def main():
    print("🔍 Watermark Detector Pro - GUI Launcher")
    print("=" * 60)
    print()
    print("Choose your preferred interface:")
    print("1. 🌐 Modern Web App (FastAPI + HTML/CSS/JS) - RECOMMENDED")
    print("   ✨ Professional design, responsive, works on any device")
    print("   🚀 Fast, modern, and feature-rich")
    print()
    print("2. 🖥️  CustomTkinter GUI (Desktop - Modern)")
    print("   🎨 Modern dark theme, lightweight, cross-platform")
    print()
    print("3. 🏢 PyQt6 GUI (Desktop - Professional)")
    print("   💼 Enterprise-grade, native look, advanced features")
    print()
    print("4. 📊 Streamlit Web App (Original)")
    print("   🌍 Browser-based, simple, easy to share")
    print()
    print("5. ❌ Exit")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (1-5): ").strip()
            
            if choice == "1":
                print("\n🌐 Launching Modern Web Application...")
                print("This is the recommended option with the best user experience!")
                if launch_web_app():
                    break
                else:
                    print("Failed to launch Web App. Try another option.")
                    
            elif choice == "2":
                print("\n🖥️  Launching CustomTkinter GUI...")
                if launch_customtkinter():
                    break
                else:
                    print("Failed to launch CustomTkinter GUI. Try another option.")
                    
            elif choice == "3":
                print("\n🏢 Launching PyQt6 GUI...")
                if launch_pyqt():
                    break
                else:
                    print("Failed to launch PyQt6 GUI. Try another option.")
                    
            elif choice == "4":
                print("\n📊 Launching Streamlit Web App...")
                print("Your browser should open automatically. If not, go to: http://localhost:8501")
                if launch_streamlit():
                    break
                else:
                    print("Failed to launch Streamlit. Try another option.")
                    
            elif choice == "5":
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, 4, or 5.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()