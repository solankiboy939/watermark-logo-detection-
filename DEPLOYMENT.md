# 🚀 Deployment Guide - Watermark Detector Pro

This guide covers multiple deployment options for hosting your watermark detection model online.

## 📋 Prerequisites

Before deploying, ensure you have:
- ✅ Your trained model file (`best.pt`) in the project directory
- ✅ All dependencies listed in `requirements.txt`
- ✅ Docker installed (for containerized deployments)
- ✅ Git repository with your code

## 🌟 Recommended Deployment Options

### 1. 🚂 Railway (Easiest & Recommended)

**Why Railway?**
- ✅ **Free tier available** with generous limits
- ✅ **Automatic HTTPS** and custom domains
- ✅ **Zero configuration** deployment
- ✅ **Built-in monitoring** and logs
- ✅ **Supports large model files**

**Steps:**
1. **Sign up** at [railway.app](https://railway.app)
2. **Connect your GitHub** repository
3. **Deploy with one click:**
   ```bash
   # Push your code to GitHub first
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```
4. **Railway will automatically:**
   - Detect the `Dockerfile`
   - Build and deploy your app
   - Provide a public URL
   - Handle SSL certificates

**Configuration:** Already included in `railway.json`

---

### 2. 🎨 Render (Great Free Option)

**Why Render?**
- ✅ **Generous free tier** (750 hours/month)
- ✅ **Automatic SSL** and CDN
- ✅ **Easy custom domains**
- ✅ **Built-in monitoring**

**Steps:**
1. **Sign up** at [render.com](https://render.com)
2. **Connect GitHub** repository
3. **Create new Web Service:**
   - Environment: `Docker`
   - Build Command: `docker build -t app .`
   - Start Command: `python backend.py`
4. **Deploy automatically**

**Configuration:** Already included in `render.yaml`

---

### 3. 🪂 Fly.io (Developer Friendly)

**Why Fly.io?**
- ✅ **Global edge deployment**
- ✅ **Excellent performance**
- ✅ **Free allowance** included
- ✅ **Advanced scaling options**

**Steps:**
1. **Install Fly CLI:**
   ```bash
   # macOS
   brew install flyctl
   
   # Windows
   iwr https://fly.io/install.ps1 -useb | iex
   
   # Linux
   curl -L https://fly.io/install.sh | sh
   ```

2. **Login and deploy:**
   ```bash
   fly auth login
   fly launch
   fly deploy
   ```

**Configuration:** Already included in `fly.toml`

---

### 4. 🟣 Heroku (Classic Choice)

**Why Heroku?**
- ✅ **Well-established platform**
- ✅ **Easy scaling**
- ✅ **Add-ons ecosystem**
- ⚠️ **No free tier** (paid plans start at $7/month)

**Steps:**
1. **Install Heroku CLI**
2. **Login and create app:**
   ```bash
   heroku login
   heroku create your-app-name
   heroku stack:set container
   git push heroku main
   ```

**Configuration:** Already included in `heroku.yml`

---

## 🐳 Docker Deployment (Self-Hosted)

### Local Docker Testing
```bash
# Build the image
docker build -t watermark-detector .

# Run locally
docker run -p 8000:8000 watermark-detector

# Or use docker-compose
docker-compose up
```

### Production Docker Deployment
```bash
# With Nginx reverse proxy
docker-compose --profile production up -d
```

---

## ☁️ Cloud Platform Specific Guides

### AWS (Amazon Web Services)

**Option 1: AWS App Runner**
```bash
# Create apprunner.yaml
version: 1.0
runtime: docker
build:
  commands:
    build:
      - echo "Building Docker image"
run:
  runtime-version: latest
  command: python backend.py
  network:
    port: 8000
    env: PORT
  env:
    - name: PYTHONPATH
      value: /app
```

**Option 2: AWS ECS with Fargate**
1. Push Docker image to ECR
2. Create ECS task definition
3. Deploy with Fargate

### Google Cloud Platform

**Cloud Run Deployment:**
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT-ID/watermark-detector
gcloud run deploy --image gcr.io/PROJECT-ID/watermark-detector --platform managed
```

### Microsoft Azure

**Container Instances:**
```bash
az container create \
  --resource-group myResourceGroup \
  --name watermark-detector \
  --image your-registry/watermark-detector \
  --ports 8000
```

---

## 🔧 Environment Configuration

### Required Environment Variables
```bash
PYTHONPATH=/app
MODEL_PATH=/app/best.pt
PORT=8000  # Some platforms require this
```

### Optional Environment Variables
```bash
# Performance tuning
WORKERS=1
MAX_REQUESTS=1000
TIMEOUT=120

# Security
ALLOWED_HOSTS=your-domain.com,www.your-domain.com
CORS_ORIGINS=https://your-domain.com
```

---

## 📊 Performance Optimization

### 1. Model Optimization
```python
# Add to backend.py for faster loading
import torch
model = YOLO("best.pt")
model.to('cpu')  # Ensure CPU usage for cloud deployment
```

### 2. Memory Management
```python
# Add memory cleanup
import gc
import torch

def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

### 3. Caching Strategy
```python
# Add Redis caching for repeated requests
import redis
import hashlib

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def get_cache_key(image_data, confidence):
    return hashlib.md5(f"{image_data}{confidence}".encode()).hexdigest()
```

---

## 🔒 Security Best Practices

### 1. Rate Limiting
```python
# Add to backend.py
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/detect")
@limiter.limit("5/minute")
async def detect_watermarks(request: Request, ...):
    # Your detection code
```

### 2. Input Validation
```python
# Enhanced file validation
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}

def validate_file(file: UploadFile):
    if file.size > MAX_FILE_SIZE:
        raise HTTPException(400, "File too large")
    
    if not any(file.filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise HTTPException(400, "Invalid file type")
```

### 3. HTTPS Configuration
```python
# Force HTTPS in production
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware

if os.getenv("ENVIRONMENT") == "production":
    app.add_middleware(HTTPSRedirectMiddleware)
```

---

## 📈 Monitoring & Logging

### 1. Health Checks
```python
# Enhanced health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "memory_usage": psutil.virtual_memory().percent,
        "timestamp": datetime.utcnow().isoformat()
    }
```

### 2. Logging Configuration
```python
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
```

---

## 💰 Cost Optimization

### Free Tier Recommendations:
1. **Railway**: 500 hours/month free
2. **Render**: 750 hours/month free
3. **Fly.io**: $5 credit monthly
4. **Vercel**: Generous free tier for serverless

### Paid Recommendations:
1. **Railway Pro**: $20/month for production apps
2. **Render**: $7/month for always-on services
3. **Fly.io**: Pay-as-you-go pricing
4. **Heroku**: $7/month basic dyno

---

## 🚨 Troubleshooting

### Common Issues:

**1. Model Loading Errors**
```bash
# Check model file exists
ls -la best.pt

# Verify model compatibility
python -c "from ultralytics import YOLO; model = YOLO('best.pt'); print('Model loaded successfully')"
```

**2. Memory Issues**
```bash
# Reduce model precision
model.half()  # Use FP16 instead of FP32

# Limit concurrent requests
uvicorn backend:app --workers 1 --limit-concurrency 10
```

**3. Timeout Issues**
```python
# Increase timeout in backend.py
if __name__ == "__main__":
    uvicorn.run(
        "backend:app",
        host="0.0.0.0",
        port=8000,
        timeout_keep_alive=120,
        timeout_graceful_shutdown=30
    )
```

---

## 🎯 Quick Start Commands

### Railway (Recommended)
```bash
git add . && git commit -m "Deploy to Railway" && git push
# Then connect repository on Railway dashboard
```

### Render
```bash
git add . && git commit -m "Deploy to Render" && git push
# Then create web service on Render dashboard
```

### Fly.io
```bash
fly launch
fly deploy
```

### Docker (Self-hosted)
```bash
docker-compose up -d
```

---

## 🌐 Custom Domain Setup

### 1. Railway
- Go to Settings → Domains
- Add your custom domain
- Update DNS records as shown

### 2. Render
- Go to Settings → Custom Domains
- Add domain and verify DNS

### 3. Fly.io
```bash
fly certs add your-domain.com
```

---

## 📞 Support

If you encounter issues:
1. Check the platform-specific documentation
2. Review logs in the platform dashboard
3. Test locally with Docker first
4. Ensure your model file is included in deployment

**Your watermark detection model will be live and accessible worldwide! 🌍**