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
