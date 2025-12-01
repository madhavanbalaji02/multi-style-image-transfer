#!/bin/bash

# Quick Start Deployment Script for Render.com
# This script helps you prepare and deploy your style transfer app

set -e  # Exit on error

echo "🚀 Style Transfer App - Render.com Deployment Helper"
echo "======================================================"
echo ""

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install git first."
    exit 1
fi

echo "📋 Step 1: Git Repository Setup"
echo "--------------------------------"

# Check if already a git repository
if [ -d ".git" ]; then
    echo "✅ Git repository already initialized"
else
    echo "Initializing git repository..."
    git init
    echo "✅ Git repository initialized"
fi

# Add .gitignore if not already added
if git ls-files --error-unmatch .gitignore > /dev/null 2>&1; then
    echo "✅ .gitignore already tracked"
else
    git add .gitignore
    echo "✅ Added .gitignore"
fi

echo ""
echo "📝 Step 2: Review Files to Commit"
echo "-----------------------------------"
git status

echo ""
read -p "Do you want to commit all changes? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    git add .
    git commit -m "Prepare for Render.com deployment"
    echo "✅ Changes committed"
else
    echo "⏭️  Skipping commit"
fi

echo ""
echo "🔗 Step 3: GitHub Repository"
echo "-----------------------------"
echo "You need to create a GitHub repository and push your code."
echo ""
echo "1. Go to https://github.com/new"
echo "2. Create a new repository (public or private)"
echo "3. DO NOT initialize with README"
echo "4. Copy the repository URL"
echo ""
read -p "Enter your GitHub repository URL (e.g., https://github.com/username/repo.git): " REPO_URL

if [ -z "$REPO_URL" ]; then
    echo "⏭️  No URL provided. You can add it manually later with:"
    echo "   git remote add origin YOUR_REPO_URL"
    echo "   git push -u origin main"
else
    # Check if remote already exists
    if git remote | grep -q "^origin$"; then
        echo "Remote 'origin' already exists. Updating URL..."
        git remote set-url origin "$REPO_URL"
    else
        git remote add origin "$REPO_URL"
    fi
    
    echo ""
    read -p "Push to GitHub now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git branch -M main
        git push -u origin main
        echo "✅ Code pushed to GitHub"
    else
        echo "⏭️  Skipping push. You can push later with: git push -u origin main"
    fi
fi

echo ""
echo "🌐 Step 4: Deploy to Render.com"
echo "--------------------------------"
echo ""
echo "Now you need to deploy your services on Render.com:"
echo ""
echo "📦 BACKEND DEPLOYMENT:"
echo "  1. Go to https://dashboard.render.com/"
echo "  2. Click 'New +' → 'Web Service'"
echo "  3. Connect your GitHub repository"
echo "  4. Configure:"
echo "     - Name: style-transfer-backend"
echo "     - Root Directory: backend"
echo "     - Build Command: pip install -r requirements.txt"
echo "     - Start Command: gunicorn -w 1 -k uvicorn.workers.UvicornWorker main:app"
echo "  5. Add Disk:"
echo "     - Name: style-transfer-disk"
echo "     - Mount Path: /opt/render/project/src/backend/uploads"
echo "     - Size: 1 GB"
echo "  6. Click 'Create Web Service'"
echo "  7. Wait for deployment (~10-15 minutes)"
echo "  8. Note your backend URL: https://YOUR-SERVICE.onrender.com"
echo ""
read -p "Enter your deployed backend URL (or press Enter to skip): " BACKEND_URL

if [ -n "$BACKEND_URL" ]; then
    echo ""
    echo "📝 Updating frontend configuration..."
    
    # Update script.js with the backend URL
    sed -i.bak "s|const API_BASE_URL = \"http://localhost:8000\";|const API_BASE_URL = \"$BACKEND_URL\";|g" frontend/script.js
    rm frontend/script.js.bak 2>/dev/null || true
    
    echo "✅ Updated frontend/script.js with backend URL"
    
    # Commit the change
    git add frontend/script.js
    git commit -m "Update API URL for production deployment"
    git push
    
    echo "✅ Changes pushed to GitHub"
fi

echo ""
echo "🎨 FRONTEND DEPLOYMENT:"
echo "  1. Go to https://dashboard.render.com/"
echo "  2. Click 'New +' → 'Static Site'"
echo "  3. Select your GitHub repository"
echo "  4. Configure:"
echo "     - Name: style-transfer-frontend"
echo "     - Root Directory: frontend"
echo "     - Build Command: echo \"Static site build\""
echo "     - Publish Directory: ./"
echo "  5. Click 'Create Static Site'"
echo "  6. Wait for deployment (~2-3 minutes)"
echo "  7. Note your frontend URL: https://YOUR-SITE.onrender.com"
echo ""
read -p "Enter your deployed frontend URL (or press Enter to skip): " FRONTEND_URL

if [ -n "$FRONTEND_URL" ]; then
    echo ""
    echo "🔒 IMPORTANT: Update CORS Settings"
    echo "-----------------------------------"
    echo "You need to update backend/main.py to allow requests from your frontend:"
    echo ""
    echo "Change this line:"
    echo "  allow_origins=[\"*\"],"
    echo ""
    echo "To:"
    echo "  allow_origins=[\"$FRONTEND_URL\"],"
    echo ""
    echo "Then commit and push the change. Render will auto-redeploy."
    echo ""
    read -p "Update CORS automatically? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Update main.py CORS
        sed -i.bak "s|allow_origins=\[\"\\*\"\]|allow_origins=[\"$FRONTEND_URL\"]|g" backend/main.py
        rm backend/main.py.bak 2>/dev/null || true
        
        git add backend/main.py
        git commit -m "Update CORS for production frontend"
        git push
        
        echo "✅ CORS updated and pushed"
    fi
fi

echo ""
echo "✨ Deployment Preparation Complete!"
echo "===================================="
echo ""
echo "📚 Next Steps:"
echo "  1. Monitor your deployments in Render dashboard"
echo "  2. Test your application at: $FRONTEND_URL"
echo "  3. Check backend API docs at: ${BACKEND_URL}/docs"
echo ""
echo "📖 For detailed instructions, see DEPLOYMENT.md"
echo ""
echo "🎉 Happy deploying!"
