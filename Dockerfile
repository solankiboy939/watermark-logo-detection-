# Optimized Dockerfile for Watermark Detector Pro
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

# Install system dependencies (minimal set)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set work directory
WORKDIR /app

# Install Python dependencies in stages to avoid timeout
# Stage 1: Core dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Stage 2: Install dependencies one by one to avoid timeout
RUN pip install --no-cache-dir fastapi==0.104.1
RUN pip install --no-cache-dir "uvicorn[standard]==0.24.0"
RUN pip install --no-cache-dir python-multipart==0.0.6
RUN pip install --no-cache-dir Pillow==10.1.0
RUN pip install --no-cache-dir numpy==1.24.3

# Stage 3: Install PyTorch (CPU only for smaller size)
RUN pip install --no-cache-dir torch==2.1.1+cpu torchvision==0.16.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Stage 4: Install OpenCV and Ultralytics
RUN pip install --no-cache-dir opencv-python-headless==4.8.1.78
RUN pip install --no-cache-dir ultralytics==8.0.196

# Copy only necessary files
COPY backend.py .
COPY frontend/ ./frontend/
COPY best.pt .

# Create uploads directory
RUN mkdir -p uploads

# Create non-root user
RUN useradd --create-home --shell /bin/bash --uid 1000 app && \
    chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "backend.py"]