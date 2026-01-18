@echo off
echo 🛠️  Watermark Detector Pro - Windows Deployment Setup
echo ============================================================

REM Check if git is available
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Git is not installed or not in PATH
    echo 💡 Please install Git from: https://git-scm.com/download/win
    echo 💡 Or use GitHub Desktop: https://desktop.github.com/
    pause
    exit /b 1
)

REM Check if required files exist
if not exist "best.pt" (
    echo ❌ Missing required file: best.pt
    echo 💡 Make sure your trained model file is in this directory
    pause
    exit /b 1
)

if not exist "backend.py" (
    echo ❌ Missing required file: backend.py
    pause
    exit /b 1
)

if not exist "frontend\index.html" (
    echo ❌ Missing required file: frontend\index.html
    pause
    exit /b 1
)

echo ✅ All required files found!

REM Setup Dockerfile
if exist "Dockerfile.simple" (
    copy "Dockerfile.simple" "Dockerfile" >nul
    echo ✅ Dockerfile.simple copied to Dockerfile
) else (
    echo ❌ Dockerfile.simple not found
    pause
    exit /b 1
)

REM Initialize git repository if needed
if not exist ".git" (
    echo 📁 Initializing Git repository...
    git init
    echo ✅ Git repository initialized
    
    REM Create .gitignore
    echo # Python > .gitignore
    echo __pycache__/ >> .gitignore
    echo *.pyc >> .gitignore
    echo venv/ >> .gitignore
    echo .env >> .gitignore
    echo uploads/ >> .gitignore
    echo *.tmp >> .gitignore
    echo ✅ .gitignore created
    
    REM Add all files
    git add .
    echo ✅ Files added to git
    
    REM Initial commit
    git commit -m "Initial commit - Watermark Detector Pro"
    echo ✅ Initial commit created
) else (
    echo ✅ Git repository already exists
    
    REM Add and commit current changes
    git add .
    git commit -m "Update deployment configuration" 2>nul
    if %errorlevel% equ 0 (
        echo ✅ Changes committed to git
    ) else (
        echo ℹ️  No changes to commit
    )
)

echo.
echo 🚀 Deployment Options (No Docker Required):
echo ============================================================
echo.
echo 1. 🚂 Railway (Recommended - Easiest)
echo    • Go to: https://railway.app
echo    • Sign up with GitHub
echo    • Click 'New Project' → 'Deploy from GitHub repo'
echo    • Select your repository
echo    • Railway builds and deploys automatically!
echo.
echo 2. 🎨 Render (Great Free Option)
echo    • Go to: https://render.com
echo    • Sign up with GitHub
echo    • Click 'New' → 'Web Service'
echo    • Connect your GitHub repository
echo    • Environment: Docker
echo    • Click 'Create Web Service'
echo.
echo 🐙 GitHub Repository Setup:
echo ============================================================
echo 1. Create repository at: https://github.com/new
echo 2. Repository name: watermark-detector-pro
echo 3. Make it Public (required for free deployments)
echo 4. Don't initialize with README
echo 5. Click 'Create repository'
echo.
echo 6. Connect your local repository:
echo    git remote add origin https://github.com/YOUR_USERNAME/watermark-detector-pro.git
echo    git branch -M main
echo    git push -u origin main
echo.
echo 🎯 Next Steps:
echo 1. Create GitHub repository (see instructions above)
echo 2. Push your code to GitHub
echo 3. Deploy using Railway or Render
echo 4. Your watermark detector will be live online! 🚀
echo.
echo 💡 No Docker installation needed - cloud platforms handle the building!
echo.
pause