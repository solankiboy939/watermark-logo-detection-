# Super Simple Dockerfile for Watermark Detector Pro
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_ROOT_USER_ACTION=ignore

# Install curl for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy application files
COPY backend.py .
COPY frontend/ ./frontend/
COPY best.pt .
RUN mkdir -p uploads

# Install dependencies step by step
RUN pip install --upgrade pip --root-user-action=ignore

# Web framework
RUN pip install fastapi uvicorn python-multipart --root-user-action=ignore

# Basic dependencies
RUN pip install Pillow numpy --root-user-action=ignore

# PyTorch CPU (smaller and faster)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --root-user-action=ignore

# Computer vision
RUN pip install opencv-python-headless --root-user-action=ignore

# YOLO
RUN pip install ultralytics --root-user-action=ignore

# Expose port
EXPOSE 8000

# Health check with longer timeout
HEALTHCHECK --interval=60s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "backend.py"]