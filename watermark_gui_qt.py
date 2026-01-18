import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QSlider, QTextEdit,
                            QFileDialog, QMessageBox, QProgressBar, QTabWidget,
                            QFrame, QGridLayout, QSplitter, QGroupBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QPixmap, QFont, QPalette, QColor, QIcon
from PIL import Image, ImageQt
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile

class DetectionWorker(QThread):
    """Worker thread for watermark detection"""
    finished = pyqtSignal(object, int)  # results, num_detections
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    
    def __init__(self, model, image, confidence):
        super().__init__()
        self.model = model
        self.image = image
        self.confidence = confidence
    
    def run(self):
        try:
            self.progress.emit(30)
            
            # Convert PIL image to numpy array
            img_array = np.array(self.image)
            
            self.progress.emit(60)
            
            # Run detection
            results = self.model.predict(
                source=img_array, 
                conf=self.confidence,
                verbose=False
            )
            
            self.progress.emit(90)
            
            # Process results
            boxes = results[0].boxes
            result_img = results[0].plot()
            result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            
            self.progress.emit(100)
            self.finished.emit(result_img_rgb, len(boxes))
            
        except Exception as e:
            self.error.emit(str(e))

class WatermarkDetectorQt(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.current_image = None
        self.processed_image = None
        self.detection_worker = None
        
        self.init_ui()
        self.load_model()
        self.apply_styles()
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("🔍 Watermark Detector Pro")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(1000, 700)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Header
        self.create_header(main_layout)
        
        # Content area
        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(content_splitter)
        
        # Left panel - Controls
        left_panel = self.create_left_panel()
        content_splitter.addWidget(left_panel)
        
        # Right panel - Image display
        right_panel = self.create_right_panel()
        content_splitter.addWidget(right_panel)
        
        # Set splitter proportions
        content_splitter.setSizes([400, 800])
        
        # Status bar
        self.create_status_bar(main_layout)
    
    def create_header(self, layout):
        """Create the header section"""
        header_frame = QFrame()
        header_frame.setObjectName("headerFrame")
        header_layout = QVBoxLayout(header_frame)
        
        # Title
        title_label = QLabel("🔍 Watermark Detector Pro")
        title_label.setObjectName("titleLabel")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(title_label)
        
        # Subtitle
        subtitle_label = QLabel("Professional Watermark Detection Solution")
        subtitle_label.setObjectName("subtitleLabel")
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(subtitle_label)
        
        layout.addWidget(header_frame)
    
    def create_left_panel(self):
        """Create the left control panel"""
        left_widget = QWidget()
        left_widget.setMaximumWidth(400)
        left_layout = QVBoxLayout(left_widget)
        
        # Upload section
        upload_group = QGroupBox("📤 Upload Image")
        upload_layout = QVBoxLayout(upload_group)
        
        self.upload_btn = QPushButton("Choose Image")
        self.upload_btn.setObjectName("primaryButton")
        self.upload_btn.clicked.connect(self.upload_image)
        upload_layout.addWidget(self.upload_btn)
        
        self.file_info_label = QLabel("No file selected")
        self.file_info_label.setObjectName("infoLabel")
        self.file_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        upload_layout.addWidget(self.file_info_label)
        
        left_layout.addWidget(upload_group)
        
        # Settings section
        settings_group = QGroupBox("⚙️ Detection Settings")
        settings_layout = QVBoxLayout(settings_group)
        
        # Confidence slider
        conf_label = QLabel("Confidence Threshold:")
        settings_layout.addWidget(conf_label)
        
        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setMinimum(10)
        self.confidence_slider.setMaximum(100)
        self.confidence_slider.setValue(25)
        self.confidence_slider.valueChanged.connect(self.update_confidence_display)
        settings_layout.addWidget(self.confidence_slider)
        
        self.conf_value_label = QLabel("0.25")
        self.conf_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        settings_layout.addWidget(self.conf_value_label)
        
        # Detect button
        self.detect_btn = QPushButton("🔍 Detect Watermarks")
        self.detect_btn.setObjectName("detectButton")
        self.detect_btn.clicked.connect(self.detect_watermarks)
        self.detect_btn.setEnabled(False)
        settings_layout.addWidget(self.detect_btn)
        
        left_layout.addWidget(settings_group)
        
        # Results section
        results_group = QGroupBox("📊 Results")
        results_layout = QVBoxLayout(results_group)
        
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(150)
        self.results_text.setPlainText("No detection performed yet.")
        results_layout.addWidget(self.results_text)
        
        left_layout.addWidget(results_group)
        
        left_layout.addStretch()
        return left_widget
    
    def create_right_panel(self):
        """Create the right image display panel"""
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Image tabs
        self.image_tabs = QTabWidget()
        
        # Original image tab
        self.original_tab = QWidget()
        original_layout = QVBoxLayout(self.original_tab)
        
        self.original_image_label = QLabel()
        self.original_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_image_label.setMinimumHeight(400)
        self.original_image_label.setStyleSheet("border: 2px dashed #ccc; background-color: #f9f9f9;")
        self.original_image_label.setText("No image loaded\n\nClick 'Choose Image' to get started")
        original_layout.addWidget(self.original_image_label)
        
        self.image_tabs.addTab(self.original_tab, "Original Image")
        
        # Detection results tab
        self.results_tab = QWidget()
        results_layout = QVBoxLayout(self.results_tab)
        
        self.results_image_label = QLabel()
        self.results_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.results_image_label.setMinimumHeight(400)
        self.results_image_label.setStyleSheet("border: 2px dashed #ccc; background-color: #f9f9f9;")
        self.results_image_label.setText("No detection performed\n\nUpload an image and click 'Detect Watermarks'")
        results_layout.addWidget(self.results_image_label)
        
        self.image_tabs.addTab(self.results_tab, "Detection Results")
        
        right_layout.addWidget(self.image_tabs)
        return right_widget
    
    def create_status_bar(self, layout):
        """Create the status bar"""
        status_frame = QFrame()
        status_layout = QVBoxLayout(status_frame)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        status_layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready - Upload an image to begin detection")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_layout.addWidget(self.status_label)
        
        layout.addWidget(status_frame)
    
    def apply_styles(self):
        """Apply custom styles to the application"""
        style = """
        QMainWindow {
            background-color: #f5f5f5;
        }
        
        #headerFrame {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                      stop:0 #667eea, stop:1 #764ba2);
            border-radius: 10px;
            padding: 20px;
            margin: 10px;
        }
        
        #titleLabel {
            color: white;
            font-size: 28px;
            font-weight: bold;
            margin: 10px;
        }
        
        #subtitleLabel {
            color: #e0e0e0;
            font-size: 16px;
            margin: 5px;
        }
        
        QGroupBox {
            font-weight: bold;
            font-size: 14px;
            border: 2px solid #ddd;
            border-radius: 8px;
            margin: 10px 0px;
            padding-top: 10px;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
        
        #primaryButton {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 12px;
            font-size: 14px;
            font-weight: bold;
            border-radius: 6px;
        }
        
        #primaryButton:hover {
            background-color: #45a049;
        }
        
        #detectButton {
            background-color: #2196F3;
            color: white;
            border: none;
            padding: 15px;
            font-size: 16px;
            font-weight: bold;
            border-radius: 6px;
        }
        
        #detectButton:hover {
            background-color: #1976D2;
        }
        
        #detectButton:disabled {
            background-color: #cccccc;
            color: #666666;
        }
        
        #infoLabel {
            color: #666;
            font-size: 12px;
            padding: 5px;
        }
        
        #statusLabel {
            color: #666;
            font-size: 12px;
            padding: 10px;
        }
        
        QSlider::groove:horizontal {
            border: 1px solid #bbb;
            background: white;
            height: 10px;
            border-radius: 4px;
        }
        
        QSlider::sub-page:horizontal {
            background: qlineargradient(x1: 0, y1: 0,    x2: 0, y2: 1,
                stop: 0 #66e, stop: 1 #bbf);
            background: qlineargradient(x1: 0, y1: 0.2, x2: 1, y2: 1,
                stop: 0 #bbf, stop: 1 #55f);
            border: 1px solid #777;
            height: 10px;
            border-radius: 4px;
        }
        
        QSlider::add-page:horizontal {
            background: #fff;
            border: 1px solid #777;
            height: 10px;
            border-radius: 4px;
        }
        
        QSlider::handle:horizontal {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #eee, stop:1 #ccc);
            border: 1px solid #777;
            width: 18px;
            margin-top: -2px;
            margin-bottom: -2px;
            border-radius: 3px;
        }
        
        QTabWidget::pane {
            border: 1px solid #ddd;
            background-color: white;
        }
        
        QTabBar::tab {
            background-color: #f0f0f0;
            padding: 10px 20px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        
        QTabBar::tab:selected {
            background-color: white;
            border-bottom: 2px solid #2196F3;
        }
        """
        self.setStyleSheet(style)
    
    def load_model(self):
        """Load the YOLO model"""
        try:
            self.model = YOLO("best.pt")
            self.status_label.setText("Model loaded successfully - Ready for detection")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
            self.status_label.setText("Failed to load model")
    
    def update_confidence_display(self):
        """Update confidence value display"""
        value = self.confidence_slider.value() / 100.0
        self.conf_value_label.setText(f"{value:.2f}")
    
    def upload_image(self):
        """Handle image upload"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select an image",
            "",
            "Image files (*.jpg *.jpeg *.png *.bmp *.tiff);;All files (*.*)"
        )
        
        if file_path:
            try:
                # Load image
                self.current_image = Image.open(file_path)
                self.display_original_image()
                
                # Update UI
                filename = os.path.basename(file_path)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                self.file_info_label.setText(f"{filename}\n{file_size:.1f} MB")
                
                self.detect_btn.setEnabled(True)
                self.status_label.setText("Image loaded successfully - Ready for detection")
                
                # Clear previous results
                self.results_text.setPlainText("Image loaded. Click 'Detect Watermarks' to analyze.")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")
    
    def display_original_image(self):
        """Display the original image"""
        if self.current_image:
            # Calculate display size
            label_size = self.original_image_label.size()
            display_width = min(600, label_size.width() - 20)
            display_height = min(400, label_size.height() - 20)
            
            # Resize image for display
            img_copy = self.current_image.copy()
            img_copy.thumbnail((display_width, display_height), Image.Resampling.LANCZOS)
            
            # Convert to QPixmap
            qimage = ImageQt.ImageQt(img_copy)
            pixmap = QPixmap.fromImage(qimage)
            
            # Update label
            self.original_image_label.setPixmap(pixmap)
            self.original_image_label.setText("")
    
    def detect_watermarks(self):
        """Start watermark detection"""
        if not self.current_image or not self.model:
            return
        
        # Disable button and show progress
        self.detect_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Analyzing image...")
        
        # Start detection worker
        confidence = self.confidence_slider.value() / 100.0
        self.detection_worker = DetectionWorker(self.model, self.current_image, confidence)
        self.detection_worker.finished.connect(self.on_detection_finished)
        self.detection_worker.error.connect(self.on_detection_error)
        self.detection_worker.progress.connect(self.progress_bar.setValue)
        self.detection_worker.start()
    
    def on_detection_finished(self, result_img_array, num_detections):
        """Handle detection completion"""
        # Convert result to PIL Image
        self.processed_image = Image.fromarray(result_img_array)
        self.display_processed_image()
        
        # Update results
        if num_detections == 0:
            self.results_text.setPlainText("✅ Analysis Complete\n\n❌ No watermarks detected\n\nThe image appears to be clean.")
            self.status_label.setText("Detection complete - No watermarks found")
        else:
            self.results_text.setPlainText(f"✅ Analysis Complete\n\n🎯 Found {num_detections} watermark(s)\n\nCheck the 'Detection Results' tab to see highlighted areas.")
            self.status_label.setText(f"Detection complete - {num_detections} watermark(s) found")
        
        # Switch to results tab
        self.image_tabs.setCurrentIndex(1)
        
        # Reset UI
        self.progress_bar.setVisible(False)
        self.detect_btn.setEnabled(True)
    
    def display_processed_image(self):
        """Display the processed image with detections"""
        if self.processed_image:
            # Calculate display size
            label_size = self.results_image_label.size()
            display_width = min(600, label_size.width() - 20)
            display_height = min(400, label_size.height() - 20)
            
            # Resize image for display
            img_copy = self.processed_image.copy()
            img_copy.thumbnail((display_width, display_height), Image.Resampling.LANCZOS)
            
            # Convert to QPixmap
            qimage = ImageQt.ImageQt(img_copy)
            pixmap = QPixmap.fromImage(qimage)
            
            # Update label
            self.results_image_label.setPixmap(pixmap)
            self.results_image_label.setText("")
    
    def on_detection_error(self, error_msg):
        """Handle detection errors"""
        self.results_text.setPlainText(f"❌ Detection Failed\n\nError: {error_msg}")
        self.status_label.setText("Detection failed - Check error message")
        self.progress_bar.setVisible(False)
        self.detect_btn.setEnabled(True)
        QMessageBox.critical(self, "Detection Error", f"Detection failed: {error_msg}")

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Watermark Detector Pro")
    
    # Set application icon (if you have one)
    # app.setWindowIcon(QIcon("icon.png"))
    
    window = WatermarkDetectorQt()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()