FROM python:3.10-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore

# Install system dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Create uploads directory
RUN mkdir -p uploads

# Install Python dependencies
RUN pip install --no-cache-dir fastapi uvicorn python-multipart Pillow numpy
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir opencv-python-headless ultralytics

# Expose port
EXPOSE 8000

# Start the application
CMD ["python", "backend.py"]