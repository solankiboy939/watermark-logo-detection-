# 🔍 Watermark Detector Pro

A professional watermark detection application powered by YOLOv8, featuring multiple modern interfaces including a cutting-edge web application for the best user experience.

## ✨ Features

- **🤖 Advanced AI Detection**: YOLOv8-powered watermark detection with adjustable confidence thresholds
- **🌐 Multiple Interface Options**: Choose from Modern Web App, CustomTkinter, PyQt6, or Streamlit
- **🎨 Professional Design**: Modern, responsive interfaces with dark themes and smooth animations
- **⚡ Real-time Processing**: Threaded detection to prevent UI freezing
- **📱 Cross-Platform**: Works on desktop, mobile, and any device with a browser
- **🖼️ Image Preview**: Side-by-side comparison of original and processed images
- **📊 Detailed Results**: Comprehensive detection statistics and visual feedback

## 🚀 Interface Options

### 1. 🌐 Modern Web App (RECOMMENDED)
- **🎯 Best User Experience**: Professional, responsive design that works on any device
- **⚡ Lightning Fast**: Modern HTML5/CSS3/JavaScript with FastAPI backend
- **📱 Mobile Friendly**: Responsive design that looks great on phones and tablets
- **🔄 Real-time Updates**: Live progress indicators and smooth animations
- **🌍 Universal Access**: Works in any modern web browser
- **🎨 Beautiful UI**: Gradient backgrounds, smooth transitions, and modern typography

### 2. 🖥️ CustomTkinter GUI
- **🌙 Modern Dark Theme**: Clean, professional appearance
- **🪶 Lightweight**: Fast startup and low resource usage
- **🔄 Cross-Platform**: Works on Windows, macOS, and Linux
- **📑 Tabbed Interface**: Organized layout for better workflow

### 3. 🏢 PyQt6 GUI
- **💼 Enterprise-Grade**: Professional native OS appearance
- **🔧 Feature-Rich**: Advanced UI components and layouts
- **🎨 Highly Customizable**: Extensive theming and styling options
- **🏗️ Robust Architecture**: Scalable and maintainable codebase

### 4. 📊 Streamlit Web App (Original)
- **🌐 Browser-Based**: Access from any device with a web browser
- **🔗 Easy Sharing**: Share detection results via web links
- **📱 Responsive**: Works on desktop and mobile devices

## 📦 Installation

1. **Clone or download this repository**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure your model file exists**:
   - Make sure `best.pt` (your trained YOLOv8 model) is in the project directory

4. **Optional: TypeScript Development** (for web interface customization):
   ```bash
   cd frontend
   npm install
   npm run build
   ```

## 🎮 Usage

### 🚀 Quick Start (Recommended)
Run the launcher script to choose your preferred interface:
```bash
python launch_gui.py
```

### 🎯 Direct Launch Options

#### 🌐 Modern Web App (Best Experience)
```bash
python backend.py
```
Then open http://localhost:8000 in your browser

#### 🖥️ CustomTkinter GUI
```bash
python watermark_gui.py
```

#### 🏢 PyQt6 GUI
```bash
python watermark_gui_qt.py
```

#### 📊 Streamlit Web App
```bash
streamlit run app.py
```

## 📖 How to Use

1. **🚀 Launch** your preferred interface
2. **📤 Upload** an image using drag-and-drop or the "Choose Image" button
3. **⚙️ Adjust** the confidence threshold (0.1 - 1.0) as needed
4. **🔍 Click** "Detect Watermarks" to analyze the image
5. **👀 View** results with highlighted detections and detailed statistics

## 🎯 Model Requirements

- **📁 Model File**: `best.pt` (YOLOv8 trained model)
- **📋 Format**: PyTorch (.pt) format
- **🏷️ Classes**: Should be trained to detect watermarks/logos

## 📚 Dependencies

### Core Dependencies
- `ultralytics` - YOLOv8 implementation
- `opencv-python-headless` - Image processing
- `Pillow` - Image handling
- `numpy` - Numerical operations
- `torch` - PyTorch backend

### GUI Dependencies
- `customtkinter` - Modern Tkinter interface
- `PyQt6` - Professional Qt interface
- `streamlit` - Web-based interface

### Web Framework Dependencies
- `fastapi` - Modern web API framework
- `uvicorn` - ASGI server
- `python-multipart` - File upload support

## 💻 System Requirements

- **🐍 Python**: 3.8 or higher
- **💾 RAM**: 4GB minimum, 8GB recommended
- **💿 Storage**: 2GB free space for dependencies
- **🎮 GPU**: Optional (CUDA-compatible for faster inference)
- **🌐 Browser**: Any modern browser (Chrome, Firefox, Safari, Edge)

## 🔧 Development

### TypeScript Development (Web Interface)
```bash
cd frontend
npm install
npm run dev  # Watch mode for development
npm run build  # Production build
```

### API Documentation
When running the web app, visit http://localhost:8000/docs for interactive API documentation.

## 🛠️ Troubleshooting

### Model Loading Issues
- Ensure `best.pt` exists in the project directory
- Check that the model was trained with compatible YOLOv8 version
- Verify model file is not corrupted

### Web Interface Issues
- **Port Conflicts**: Change port in `backend.py` if 8000 is occupied
- **CORS Issues**: Check browser console for cross-origin errors
- **File Upload**: Ensure file size is under 10MB limit

### Desktop GUI Issues
- **CustomTkinter**: Update to latest version: `pip install --upgrade customtkinter`
- **PyQt6**: Install system dependencies if needed
- **Streamlit**: Check port 8501 is available

### Performance Issues
- Reduce image size for faster processing
- Lower confidence threshold for more detections
- Use GPU acceleration if available

## 📁 File Structure

```
watermark-detector/
├── backend.py                 # FastAPI web server
├── frontend/                  # Web interface files
│   ├── index.html            # Main HTML page
│   ├── static/
│   │   ├── css/styles.css    # Modern CSS styles
│   │   └── js/
│   │       ├── app.js        # JavaScript application
│   │       └── app.ts        # TypeScript version
│   ├── package.json          # Node.js dependencies
│   └── tsconfig.json         # TypeScript configuration
├── app.py                    # Streamlit web interface
├── watermark_gui.py          # CustomTkinter GUI
├── watermark_gui_qt.py       # PyQt6 GUI
├── launch_gui.py             # Universal launcher script
├── best.pt                   # Your trained YOLOv8 model
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🎨 Screenshots & Features

### Modern Web Interface
- **🎨 Beautiful Design**: Gradient backgrounds and modern typography
- **📱 Responsive Layout**: Perfect on desktop, tablet, and mobile
- **🔄 Real-time Progress**: Animated loading indicators
- **📊 Rich Results**: Detailed statistics and visual feedback
- **🌙 Dark Theme**: Easy on the eyes with professional appearance

### Desktop Applications
- **🖥️ Native Feel**: Platform-specific styling and behavior
- **⚡ Fast Performance**: Direct system integration
- **🔧 Advanced Controls**: Detailed configuration options

## 📄 License

This project is provided as-is for educational and research purposes.

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section above
2. Ensure all dependencies are properly installed
3. Verify your model file is compatible with YOLOv8
4. For web interface issues, check browser console for errors

---

**🌟 Enjoy detecting watermarks with style! The Modern Web App provides the best experience with professional design and cross-platform compatibility.**