#!/bin/bash

# Transfer Van Gogh LoRA scripts to HPC
# Run this from your Mac

HPC_USER="madbala"
HPC_HOST="bigred200.uits.iu.edu"
HPC_PROJECT_DIR="sd_vangogh_lora"
LOCAL_SCRIPTS_DIR="/Users/madhavanbalaji/Documents/CV/project/hpc_scripts"

echo "📤 Transferring Van Gogh LoRA scripts to HPC"
echo "=============================================="
echo ""
echo "Source: $LOCAL_SCRIPTS_DIR"
echo "Destination: ${HPC_USER}@${HPC_HOST}:~/${HPC_PROJECT_DIR}/"
echo ""

# Create scripts directory if it doesn't exist
echo "Creating project directory on HPC..."
ssh ${HPC_USER}@${HPC_HOST} "mkdir -p ${HPC_PROJECT_DIR}"

# Transfer scripts
echo ""
echo "Transferring training and inference scripts..."

scp "${LOCAL_SCRIPTS_DIR}/train_vangogh_lora.py" \
    "${HPC_USER}@${HPC_HOST}:~/${HPC_PROJECT_DIR}/"

scp "${LOCAL_SCRIPTS_DIR}/test_vangogh_lora.py" \
    "${HPC_USER}@${HPC_HOST}:~/${HPC_PROJECT_DIR}/"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Scripts transferred successfully!"
    echo ""
    echo "Next steps on HPC:"
    echo "1. ssh ${HPC_USER}@${HPC_HOST}"
    echo "2. Request GPU: salloc -A c01949 -N 1 -n 1 --gres=gpu:1 --partition=gpu -t 2:00:00"
    echo "3. SSH to node: ssh \$(scontrol show hostname \$SLURM_NODELIST)"
    echo "4. cd ${HPC_PROJECT_DIR}"
    echo "5. source ~/sd_lora_env/bin/activate"
    echo "6. python train_vangogh_lora.py"
else
    echo ""
    echo "❌ Transfer failed!"
    exit 1
fi
