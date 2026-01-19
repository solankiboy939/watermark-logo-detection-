FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
ENV DEBIAN_FRONTEND=noninteractive

# Install ALL required system dependencies for OpenCV and YOLO
RUN apt-get update && apt-get install -y \
    curl \
    libxcb1 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxrender1 \
    libfontconfig1 \
    libice6 \
    libxrandr2 \
    libxss1 \
    libxtst6 \
    libxi6 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxfixes3 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Create directories
RUN mkdir -p uploads

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip

# Install web framework
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    python-multipart==0.0.6

# Install basic ML dependencies
RUN pip install --no-cache-dir \
    Pillow==10.1.0 \
    numpy==1.24.3

# Install PyTorch CPU version
RUN pip install --no-cache-dir \
    torch==2.1.1+cpu \
    torchvision==0.16.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install OpenCV headless (should work without display)
RUN pip install --no-cache-dir opencv-python-headless==4.8.1.78

# Install YOLO
RUN pip install --no-cache-dir ultralytics==8.0.196

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "backend.py"]