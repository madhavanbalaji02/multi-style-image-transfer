#!/bin/bash

# Quick Project Transfer to HPC
# Run this from your Mac terminal

PROJECT_DIR="/Users/madhavanbalaji/Documents/CV/project"
HPC_USER="madbala"
HPC_HOST="bigred200.uits.iu.edu"
HPC_DIR="cv_project"

echo "📦 Transferring CV Project to IU Big Red 200"
echo "============================================="
echo ""
echo "Source: $PROJECT_DIR"
echo "Destination: ${HPC_USER}@${HPC_HOST}:~/${HPC_DIR}/"
echo ""

# Check if project directory exists
if [ ! -d "$PROJECT_DIR" ]; then
    echo "❌ Project directory not found: $PROJECT_DIR"
    exit 1
fi

cd "$PROJECT_DIR"

echo "🔄 Starting rsync transfer..."
echo ""

rsync -avz --progress \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='outputs/' \
    --exclude='uploads/' \
    --exclude='*.log' \
    --exclude='.venv/' \
    --exclude='venv/' \
    --exclude='.git/' \
    --exclude='node_modules/' \
    --exclude='.DS_Store' \
    --exclude='*.pth' \
    --exclude='*.ckpt' \
    --exclude='.cache/' \
    ./ ${HPC_USER}@${HPC_HOST}:~/${HPC_DIR}/

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Transfer complete!"
    echo ""
    echo "Next steps on HPC:"
    echo "1. ssh ${HPC_USER}@${HPC_HOST}"
    echo "2. cd ~/${HPC_DIR}/backend"
    echo "3. python3 -m venv ~/cv_env"
    echo "4. source ~/cv_env/bin/activate"
    echo "5. pip install -r requirements.txt"
else
    echo ""
    echo "❌ Transfer failed. Check your connection to HPC."
    exit 1
fi
