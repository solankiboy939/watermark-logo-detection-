#!/usr/bin/env python3
"""
Setup Deployment Script for Watermark Detector Pro

This script helps you set up deployment without needing Docker locally.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_git():
    """Check if git is available"""
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def init_git_repo():
    """Initialize git repository"""
    try:
        # Initialize git repo
        subprocess.run(["git", "init"], check=True)
        print("✅ Git repository initialized")
        
        # Create .gitignore if it doesn't exist
        gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log

# Local uploads
uploads/
*.tmp
""".strip()
        
        if not Path(".gitignore").exists():
            with open(".gitignore", "w") as f:
                f.write(gitignore_content)
            print("✅ .gitignore created")
        
        # Add all files
        subprocess.run(["git", "add", "."], check=True)
        print("✅ Files added to git")
        
        # Initial commit
        subprocess.run(["git", "commit", "-m", "Initial commit - Watermark Detector Pro"], check=True)
        print("✅ Initial commit created")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Git setup failed: {e}")
        return False

def check_required_files():
    """Check if all required files exist"""
    required_files = [
        "best.pt",
        "backend.py", 
        "frontend/index.html",
        "Dockerfile.simple"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ All required files found!")
    return True

def setup_dockerfile():
    """Setup the correct Dockerfile"""
    if Path("Dockerfile.simple").exists():
        # Copy simple dockerfile to main dockerfile
        with open("Dockerfile.simple", "r") as f:
            content = f.read()
        
        with open("Dockerfile", "w") as f:
            f.write(content)
        
        print("✅ Dockerfile.simple copied to Dockerfile")
        return True
    else:
        print("❌ Dockerfile.simple not found")
        return False

def show_deployment_options():
    """Show deployment options that don't require local Docker"""
    print("\n🚀 Deployment Options (No Docker Required):")
    print("=" * 60)
    
    print("\n1. 🚂 Railway (Recommended - Easiest)")
    print("   • Go to: https://railway.app")
    print("   • Sign up with GitHub")
    print("   • Click 'New Project' → 'Deploy from GitHub repo'")
    print("   • Select your repository")
    print("   • Railway builds and deploys automatically!")
    
    print("\n2. 🎨 Render (Great Free Option)")
    print("   • Go to: https://render.com")
    print("   • Sign up with GitHub")
    print("   • Click 'New' → 'Web Service'")
    print("   • Connect your GitHub repository")
    print("   • Environment: Docker")
    print("   • Click 'Create Web Service'")
    
    print("\n3. 🪂 Fly.io (Global Deployment)")
    print("   • Install Fly CLI: https://fly.io/docs/hands-on/install-flyctl/")
    print("   • Run: flyctl auth login")
    print("   • Run: flyctl launch")
    print("   • Run: flyctl deploy")
    
    print("\n4. 🌐 GitHub Codespaces (Test Online)")
    print("   • Go to your GitHub repository")
    print("   • Click 'Code' → 'Codespaces' → 'Create codespace'")
    print("   • Test your app in the cloud environment")

def create_github_repo_instructions():
    """Create instructions for GitHub repository setup"""
    instructions = """
🐙 GitHub Repository Setup Instructions:

1. **Create GitHub Repository:**
   • Go to: https://github.com/new
   • Repository name: watermark-detector-pro
   • Make it Public (required for free deployments)
   • Don't initialize with README (we have files already)
   • Click 'Create repository'

2. **Connect Local Repository to GitHub:**
   Run these commands in your terminal:
   
   git remote add origin https://github.com/YOUR_USERNAME/watermark-detector-pro.git
   git branch -M main
   git push -u origin main

   Replace YOUR_USERNAME with your actual GitHub username.

3. **Verify Upload:**
   • Refresh your GitHub repository page
   • You should see all your files including:
     - backend.py
     - frontend/ folder
     - best.pt (your model file)
     - Dockerfile
     - All configuration files

4. **Deploy:**
   • Once files are on GitHub, use Railway or Render
   • They will automatically build and deploy your app
   • No local Docker needed!
""".strip()
    
    with open("GITHUB_SETUP.md", "w") as f:
        f.write(instructions)
    
    print("✅ GitHub setup instructions saved to GITHUB_SETUP.md")

def main():
    print("🛠️  Watermark Detector Pro - Deployment Setup")
    print("=" * 60)
    
    # Check if git is available
    if not check_git():
        print("❌ Git is not installed or not in PATH")
        print("💡 Please install Git from: https://git-scm.com/download/win")
        print("💡 Or use GitHub Desktop: https://desktop.github.com/")
        return
    
    # Check required files
    if not check_required_files():
        print("\n❌ Cannot proceed without required files")
        return
    
    # Setup Dockerfile
    if not setup_dockerfile():
        print("\n❌ Cannot setup Dockerfile")
        return
    
    # Initialize git repository if needed
    if not Path(".git").exists():
        print("\n📁 Initializing Git repository...")
        if not init_git_repo():
            print("❌ Failed to initialize Git repository")
            return
    else:
        print("✅ Git repository already exists")
        
        # Add and commit current changes
        try:
            subprocess.run(["git", "add", "."], check=True)
            subprocess.run(["git", "commit", "-m", "Update deployment configuration"], check=True)
            print("✅ Changes committed to git")
        except subprocess.CalledProcessError:
            print("ℹ️  No changes to commit")
    
    # Create GitHub instructions
    create_github_repo_instructions()
    
    # Show deployment options
    show_deployment_options()
    
    print("\n" + "=" * 60)
    print("🎯 Next Steps:")
    print("1. Read GITHUB_SETUP.md for GitHub repository setup")
    print("2. Push your code to GitHub")
    print("3. Deploy using Railway (recommended) or Render")
    print("4. Your watermark detector will be live online! 🚀")
    
    print("\n💡 No Docker installation needed - cloud platforms handle the building!")

if __name__ == "__main__":
    main()