FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
ENV DEBIAN_FRONTEND=noninteractive

# Install EVERYTHING that could possibly be needed
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    libxcb1 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libice6 \
    libxrandr2 \
    libxss1 \
    libxtst6 \
    libxi6 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxfixes3 \
    libfontconfig1 \
    libasound2 \
    libgtk-3-0 \
    libgdk-pixbuf2.0-0 \
    libcairo-gobject2 \
    libgtk2.0-0 \
    libgconf-2-4 \
    libxss1 \
    libappindicator1 \
    libnss3 \
    lsb-release \
    xdg-utils \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app
COPY . .
RUN mkdir -p uploads

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir fastapi uvicorn python-multipart Pillow numpy
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir opencv-python-headless ultralytics

EXPOSE 8000
CMD ["python", "backend.py"]