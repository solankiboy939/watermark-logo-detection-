# Watermark Logo Detector

## Overview
The **Watermark Logo Detector** is a machine learning-powered application designed to detect watermark logos in images using the state-of-the-art **YOLOv8** object detection model. This web-based application built with **Streamlit** allows users to upload images, adjust detection settings, and visualize the results in real-time.

### Key Features:
- **YOLOv8-based Object Detection**: Uses the YOLOv8 model to detect watermark logos in images.
- **Custom Confidence Threshold**: Adjust the confidence level for detections.
- **User-friendly Interface**: Simple drag-and-drop functionality for uploading images.
- **Real-time Results**: Instant processing and display of detected logos with bounding boxes.

---

## Requirements

- **Python 3.10+**
- **YOLOv8**: The model is pre-trained and customized for watermark logo detection.
- **Streamlit**: For building the web interface.
- **OpenCV**: For image processing.
- **PIL (Pillow)**: For image handling.
- **NumPy**: For numerical operations.

### Install Dependencies
To install the necessary dependencies, run the following command:

```bash
pip install -r requirements.txt

```
## How to Use

1. **Visit the Web App**: You can access the watermark logo detection tool by visiting the following link:  
   [Watermark Logo Detector](https://watermarklogodetecti0n.streamlit.app/)
2. **Upload an Image**: Click on the "Upload Image" button to upload your image in JPG, JPEG, or PNG format.
3. **Adjust Confidence Threshold**: Use the slider to set the minimum confidence for detection. The higher the threshold, the fewer but more confident detections will be made.
4. **Run Detection**: Click the "Run Detection" button to initiate the process. The application will display the results with bounding boxes around the detected watermark logos.
5. **View Results**: After detection, the image with marked watermark logos will be shown.
