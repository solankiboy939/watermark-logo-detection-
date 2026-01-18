import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import os

class WatermarkDetectorGUI:
    def __init__(self):
        # Set appearance mode and color theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Create main window
        self.root = ctk.CTk()
        self.root.title("🔍 Watermark Detector Pro")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # Load model
        self.model = None
        self.load_model()
        
        # Variables
        self.current_image = None
        self.processed_image = None
        self.confidence_var = ctk.DoubleVar(value=0.25)
        
        self.setup_ui()
        
    def load_model(self):
        """Load the YOLO model"""
        try:
            self.model = YOLO("best.pt")
            print("Model loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main container
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        title_label = ctk.CTkLabel(
            main_frame, 
            text="🔍 Watermark Detector Pro", 
            font=ctk.CTkFont(size=32, weight="bold")
        )
        title_label.pack(pady=(20, 10))
        
        subtitle_label = ctk.CTkLabel(
            main_frame, 
            text="Professional Watermark Detection Solution", 
            font=ctk.CTkFont(size=16),
            text_color="gray"
        )
        subtitle_label.pack(pady=(0, 30))
        
        # Content frame
        content_frame = ctk.CTkFrame(main_frame)
        content_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Left panel - Controls
        left_panel = ctk.CTkFrame(content_frame)
        left_panel.pack(side="left", fill="y", padx=(20, 10), pady=20)
        left_panel.configure(width=300)
        
        # Upload section
        upload_frame = ctk.CTkFrame(left_panel)
        upload_frame.pack(fill="x", padx=20, pady=20)
        
        upload_label = ctk.CTkLabel(
            upload_frame, 
            text="📤 Upload Image", 
            font=ctk.CTkFont(size=18, weight="bold")
        )
        upload_label.pack(pady=(20, 10))
        
        self.upload_btn = ctk.CTkButton(
            upload_frame,
            text="Choose Image",
            command=self.upload_image,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.upload_btn.pack(pady=10, padx=20, fill="x")
        
        # File info
        self.file_info_label = ctk.CTkLabel(
            upload_frame,
            text="No file selected",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.file_info_label.pack(pady=(0, 20))
        
        # Settings section
        settings_frame = ctk.CTkFrame(left_panel)
        settings_frame.pack(fill="x", padx=20, pady=20)
        
        settings_label = ctk.CTkLabel(
            settings_frame, 
            text="⚙️ Detection Settings", 
            font=ctk.CTkFont(size=18, weight="bold")
        )
        settings_label.pack(pady=(20, 10))
        
        # Confidence slider
        conf_label = ctk.CTkLabel(
            settings_frame, 
            text="Confidence Threshold:", 
            font=ctk.CTkFont(size=14)
        )
        conf_label.pack(pady=(10, 5))
        
        self.confidence_slider = ctk.CTkSlider(
            settings_frame,
            from_=0.1,
            to=1.0,
            variable=self.confidence_var,
            number_of_steps=90
        )
        self.confidence_slider.pack(pady=5, padx=20, fill="x")
        
        self.conf_value_label = ctk.CTkLabel(
            settings_frame,
            text=f"{self.confidence_var.get():.2f}",
            font=ctk.CTkFont(size=12)
        )
        self.conf_value_label.pack(pady=(0, 20))
        
        # Update confidence display
        self.confidence_var.trace("w", self.update_confidence_display)
        
        # Detect button
        self.detect_btn = ctk.CTkButton(
            settings_frame,
            text="🔍 Detect Watermarks",
            command=self.detect_watermarks,
            height=50,
            font=ctk.CTkFont(size=16, weight="bold"),
            state="disabled"
        )
        self.detect_btn.pack(pady=20, padx=20, fill="x")
        
        # Results section
        results_frame = ctk.CTkFrame(left_panel)
        results_frame.pack(fill="x", padx=20, pady=20)
        
        results_label = ctk.CTkLabel(
            results_frame, 
            text="📊 Results", 
            font=ctk.CTkFont(size=18, weight="bold")
        )
        results_label.pack(pady=(20, 10))
        
        self.results_text = ctk.CTkTextbox(
            results_frame,
            height=100,
            font=ctk.CTkFont(size=12)
        )
        self.results_text.pack(pady=(0, 20), padx=20, fill="x")
        self.results_text.insert("0.0", "No detection performed yet.")
        
        # Right panel - Image display
        right_panel = ctk.CTkFrame(content_frame)
        right_panel.pack(side="right", fill="both", expand=True, padx=(10, 20), pady=20)
        
        # Image tabs
        self.image_tabview = ctk.CTkTabview(right_panel)
        self.image_tabview.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Original image tab
        self.image_tabview.add("Original Image")
        self.original_image_label = ctk.CTkLabel(
            self.image_tabview.tab("Original Image"),
            text="No image loaded\n\nClick 'Choose Image' to get started",
            font=ctk.CTkFont(size=16),
            text_color="gray"
        )
        self.original_image_label.pack(expand=True)
        
        # Detection results tab
        self.image_tabview.add("Detection Results")
        self.results_image_label = ctk.CTkLabel(
            self.image_tabview.tab("Detection Results"),
            text="No detection performed\n\nUpload an image and click 'Detect Watermarks'",
            font=ctk.CTkFont(size=16),
            text_color="gray"
        )
        self.results_image_label.pack(expand=True)
        
        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(main_frame)
        self.progress_bar.pack(fill="x", padx=40, pady=(0, 20))
        self.progress_bar.set(0)
        
        # Status bar
        self.status_label = ctk.CTkLabel(
            main_frame,
            text="Ready - Upload an image to begin detection",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.status_label.pack(pady=(0, 10))
    
    def update_confidence_display(self, *args):
        """Update confidence value display"""
        self.conf_value_label.configure(text=f"{self.confidence_var.get():.2f}")
    
    def upload_image(self):
        """Handle image upload"""
        file_types = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=file_types
        )
        
        if file_path:
            try:
                # Load and display image
                self.current_image = Image.open(file_path)
                self.display_original_image()
                
                # Update UI
                filename = os.path.basename(file_path)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                self.file_info_label.configure(
                    text=f"{filename}\n{file_size:.1f} MB"
                )
                
                self.detect_btn.configure(state="normal")
                self.status_label.configure(text="Image loaded successfully - Ready for detection")
                
                # Clear previous results
                self.results_text.delete("0.0", "end")
                self.results_text.insert("0.0", "Image loaded. Click 'Detect Watermarks' to analyze.")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def display_original_image(self):
        """Display the original image"""
        if self.current_image:
            # Calculate display size
            display_width = 600
            display_height = 400
            
            # Resize image for display
            img_copy = self.current_image.copy()
            img_copy.thumbnail((display_width, display_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img_copy)
            
            # Update label
            self.original_image_label.configure(image=photo, text="")
            self.original_image_label.image = photo  # Keep a reference
    
    def detect_watermarks(self):
        """Perform watermark detection in a separate thread"""
        if not self.current_image or not self.model:
            return
        
        # Disable button and show progress
        self.detect_btn.configure(state="disabled")
        self.progress_bar.set(0)
        self.status_label.configure(text="Analyzing image...")
        
        # Run detection in thread to prevent UI freezing
        thread = threading.Thread(target=self._perform_detection)
        thread.daemon = True
        thread.start()
    
    def _perform_detection(self):
        """Perform the actual detection"""
        try:
            # Update progress
            self.root.after(0, lambda: self.progress_bar.set(0.3))
            
            # Convert PIL image to numpy array
            img_array = np.array(self.current_image)
            
            # Run detection
            self.root.after(0, lambda: self.progress_bar.set(0.6))
            results = self.model.predict(
                source=img_array, 
                conf=self.confidence_var.get(),
                verbose=False
            )
            
            # Process results
            self.root.after(0, lambda: self.progress_bar.set(0.9))
            boxes = results[0].boxes
            
            # Create result image
            result_img = results[0].plot()
            result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            self.processed_image = Image.fromarray(result_img_rgb)
            
            # Update UI in main thread
            self.root.after(0, lambda: self._update_results(len(boxes)))
            
        except Exception as e:
            self.root.after(0, lambda: self._handle_detection_error(str(e)))
    
    def _update_results(self, num_detections):
        """Update UI with detection results"""
        # Display processed image
        self.display_processed_image()
        
        # Update results text
        self.results_text.delete("0.0", "end")
        if num_detections == 0:
            self.results_text.insert("0.0", "✅ Analysis Complete\n\n❌ No watermarks detected\n\nThe image appears to be clean.")
            self.status_label.configure(text="Detection complete - No watermarks found")
        else:
            self.results_text.insert("0.0", f"✅ Analysis Complete\n\n🎯 Found {num_detections} watermark(s)\n\nCheck the 'Detection Results' tab to see highlighted areas.")
            self.status_label.configure(text=f"Detection complete - {num_detections} watermark(s) found")
        
        # Switch to results tab
        self.image_tabview.set("Detection Results")
        
        # Reset UI
        self.progress_bar.set(1.0)
        self.detect_btn.configure(state="normal")
    
    def display_processed_image(self):
        """Display the processed image with detections"""
        if self.processed_image:
            # Calculate display size
            display_width = 600
            display_height = 400
            
            # Resize image for display
            img_copy = self.processed_image.copy()
            img_copy.thumbnail((display_width, display_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img_copy)
            
            # Update label
            self.results_image_label.configure(image=photo, text="")
            self.results_image_label.image = photo  # Keep a reference
    
    def _handle_detection_error(self, error_msg):
        """Handle detection errors"""
        self.results_text.delete("0.0", "end")
        self.results_text.insert("0.0", f"❌ Detection Failed\n\nError: {error_msg}")
        self.status_label.configure(text="Detection failed - Check error message")
        self.progress_bar.set(0)
        self.detect_btn.configure(state="normal")
        messagebox.showerror("Detection Error", f"Detection failed: {error_msg}")
    
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = WatermarkDetectorGUI()
    app.run()