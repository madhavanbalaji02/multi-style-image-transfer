# IU Big Red 200 HPC Setup Guide

## Connection Information

- **Login Node**: `bigred200.uits.iu.edu`
- **Username**: `madbala`
- **Allocation**: `c01949`

## Step 1: Connect to HPC

### Option A: Using SSH Key (Recommended)

If your SSH key isn't already set up on Big Red 200, you need to copy it:

```bash
ssh-copy-id -i ~/.ssh/id_rsa.pub madbala@bigred200.uits.iu.edu
```

Enter your password when prompted. After this, you can connect without a password:

```bash
ssh -i ~/.ssh/id_rsa madbala@bigred200.uits.iu.edu
```

### Option B: Using Password

```bash
ssh madbala@bigred200.uits.iu.edu
```

Enter your IU password when prompted.

---

## Step 2: Request GPU Allocation

Once logged into Big Red 200:

```bash
salloc -A c01949 -N 1 -n 1 -t 2:00:00 --gres=gpu:1 --partition=gpu
```

**Parameters:**
- `-A c01949`: Your allocation account
- `-N 1`: 1 node
- `-n 1`: 1 task/core
- `-t 2:00:00`: 2 hours runtime
- `--gres=gpu:1`: Request 1 GPU
- `--partition=gpu`: Use GPU partition

**Wait for allocation message:**
```
salloc: Granted job allocation 12345
salloc: Waiting for resource configuration
salloc: Nodes nid00123 are ready for job
```

---

## Step 3: SSH to Allocated Node

After allocation is granted, connect to the specific node:

```bash
# Get the allocated node name (example: nid00123)
ssh $(scontrol show hostname $SLURM_NODELIST)
```

Or manually:
```bash
ssh nid00123  # Replace with your actual node
```

---

## Step 4: Load Python Environment

```bash
module load python/gpu/3.10.10
```

Verify:
```bash
python3 --version
which python3
```

---

## Step 5: Set Up Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv ~/cv_project_env

# Activate it
source ~/cv_project_env/bin/activate

# Upgrade pip
pip install --upgrade pip
```

---

## Step 6: Transfer Your Project to HPC

### From Your Mac:

```bash
# Compress your project
cd /Users/madhavanbalaji/Documents/CV
tar -czf project.tar.gz project/

# Copy to HPC
scp project.tar.gz madbala@bigred200.uits.iu.edu:~/

# Or use rsync (better for updates)
rsync -avz --exclude='*.pyc' --exclude='__pycache__' \
      --exclude='outputs' --exclude='uploads' \
      --exclude='.venv' --exclude='venv' \
      project/ madbala@bigred200.uits.iu.edu:~/cv_project/
```

### On HPC (after transfer):

```bash
# If using tar
cd ~
tar -xzf project.tar.gz

# Navigate to project
cd project/backend
```

---

## Step 7: Install Dependencies on HPC

```bash
# Activate your environment (if not already)
source ~/cv_project_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Note:** This will take several minutes as it installs PyTorch, TensorFlow, and other ML libraries.

---

## Step 8: Download Pre-trained Models

Your `gan_inference.py` references model files in `neural_style/models/`. You need to download these:

```bash
cd ~/cv_project/backend
mkdir -p neural_style/models

# Download PyTorch fast-neural-style models
cd neural_style/models

# Mosaic (Cubism)
wget https://download.pytorch.org/models/mosaic-9e94bfb3.pth -O mosaic.pth

# Candy (Expressionism)
wget https://download.pytorch.org/models/candy-9e94bfb3.pth -O candy.pth

# Udnie (Impressionism)
wget https://download.pytorch.org/models/udnie-9e94bfb3.pth -O udnie.pth

# Rain Princess
wget https://download.pytorch.org/models/rain-princess-9e94bfb3.pth -O rain_princess.pth
```

---

## Step 9: Test the Backend

```bash
cd ~/cv_project/backend

# Test GAN inference
python3 -c "from gan_inference import apply_gan; print('GAN module loaded successfully')"

# Test Diffusion inference (requires GPU)
python3 -c "from diffusion_inference import apply_diffusion; print('Diffusion module loaded successfully')"
```

---

## Step 10: Run the Backend Server

```bash
# Start the FastAPI server
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Access from your Mac:**
```
http://bigred200.uits.iu.edu:8000/docs
```

**Note:** You may need to set up SSH port forwarding if HPC firewall blocks direct access:

```bash
# On your Mac
ssh -L 8000:localhost:8000 madbala@bigred200.uits.iu.edu
```

Then access at `http://localhost:8000/docs`

---

## Useful HPC Commands

### Check Job Status
```bash
squeue -u madbala
```

### Check GPU Status
```bash
nvidia-smi
```

### Cancel Job
```bash
scancel <job_id>
```

### Check Allocation Balance
```bash
sbalance -a c01949
```

### Exit Compute Node
```bash
exit  # Returns to login node
```

### Exit HPC
```bash
exit  # Returns to your Mac
```

---

## Troubleshooting

### SSH Key Not Working
If SSH still asks for password after `ssh-copy-id`:

1. Check permissions on HPC:
```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
```

2. Verify key is in `~/.ssh/authorized_keys` on HPC

### Module Not Found
If `module load python/gpu/3.10.10` fails:

```bash
# List available Python modules
module avail python

# Load the correct version
module load python/gpu/<version>
```

### GPU Not Available
Check if you're on the compute node (not login node):

```bash
hostname  # Should show something like nid00123, not login1
nvidia-smi  # Should show GPU info
```

### Port Already in Use
If port 8000 is busy:

```bash
# Find and kill process
lsof -ti:8000 | xargs kill -9

# Or use a different port
uvicorn main:app --host 0.0.0.0 --port 8080
```

---

## Quick Reference

**Connect:**
```bash
ssh madbala@bigred200.uits.iu.edu
```

**Request GPU:**
```bash
salloc -A c01949 -N 1 -n 1 -t 2:00:00 --gres=gpu:1 --partition=gpu
```

**SSH to node:**
```bash
ssh $(scontrol show hostname $SLURM_NODELIST)
```

**Load Python:**
```bash
module load python/gpu/3.10.10
source ~/cv_project_env/bin/activate
```

**Run server:**
```bash
cd ~/cv_project/backend
uvicorn main:app --host 0.0.0.0 --port 8000
```
