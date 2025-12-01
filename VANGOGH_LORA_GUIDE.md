# Van Gogh LoRA Training Guide

Complete guide for training a Stable Diffusion LoRA model on Van Gogh paintings using IU Big Red 200 HPC.

## Overview

This project fine-tunes Stable Diffusion 1.5 using LoRA (Low-Rank Adaptation) to generate images in Van Gogh's artistic style.

**Key Features:**
- ✅ Uses PEFT library for efficient LoRA training
- ✅ Trains on ~400 Van Gogh paintings
- ✅ GPU-accelerated on Big Red 200
- ✅ Generates Van Gogh-style images from text prompts

---

## Quick Start

### On Your Mac

1. **Transfer scripts to HPC:**
   ```bash
   cd /Users/madhavanbalaji/Documents/CV/project
   ./transfer-lora-scripts.sh
   ```

### On HPC (via SSH)

2. **Connect and request GPU:**
   ```bash
   ssh madbala@bigred200.uits.iu.edu
   salloc -A c01949 -N 1 -n 1 --gres=gpu:1 --partition=gpu -t 2:00:00
   ssh $(scontrol show hostname $SLURM_NODELIST)
   ```

3. **Set up environment:**
   ```bash
   module load python/gpu/3.10.10
   python3 -m venv ~/sd_lora_env
   source ~/sd_lora_env/bin/activate
   pip install --upgrade pip
   ```

4. **Install dependencies:**
   ```bash
   cd ~/sd_vangogh_lora
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements_lora.txt
   ```

5. **Download Van Gogh dataset:**
   ```bash
   mkdir -p images output
   cd images
   wget https://efrosgans.eecs.berkeley.edu/cyclegan/datasets/vangogh2photo.zip
   unzip vangogh2photo.zip
   mv trainA/* .
   rm -rf trainA trainB vangogh2photo.zip
   cd ..
   ```

6. **Start training:**
   ```bash
   python train_vangogh_lora.py
   ```

7. **Generate test images:**
   ```bash
   python test_vangogh_lora.py
   ```

8. **Download results to Mac:**
   ```bash
   # Run from your Mac
   scp -r madbala@bigred200.uits.iu.edu:~/sd_vangogh_lora/generated_images_* .
   ```

---

## Detailed Guide

### Prerequisites

- IU Big Red 200 HPC access
- Allocation account: `c01949`
- ~10 GB disk space
- GPU node (required for training)

### Project Structure

```
sd_vangogh_lora/
├── images/                      # Van Gogh paintings (~400 images)
├── output/
│   ├── vangogh-lora-final/      # Final trained LoRA weights
│   └── checkpoint-epoch-*/      # Training checkpoints
├── generated_images_*/          # Generated test images
├── train_vangogh_lora.py       # Training script
├── test_vangogh_lora.py        # Inference script
└── requirements_lora.txt       # Python dependencies
```

### Training Details

**Model:** Stable Diffusion 1.5 (`runwayml/stable-diffusion-v1-5`)

**LoRA Configuration:**
- Rank (r): 8
- Alpha: 8
- Target modules: `to_q`, `to_k`, `to_v`, `to_out.0`
- Dropout: 0.0

**Training Parameters:**
- Epochs: 10
- Batch size: 1
- Learning rate: 1e-4
- Optimizer: AdamW
- Mixed precision: FP16

**Expected Training Time:** 30-60 minutes on A100 GPU

**Checkpoints:** Saved every 2 epochs in `output/checkpoint-epoch-*/`

### Dataset

**Source:** CycleGAN Van Gogh dataset  
**URL:** https://efrosgans.eecs.berkeley.edu/cyclegan/datasets/vangogh2photo.zip  
**Size:** ~400 paintings from Van Gogh's collection  
**Format:** JPG images  
**License:** Public domain (Van Gogh paintings)

### Generated Output

The inference script generates 6 different Van Gogh-style images:
1. Starry night over a village
2. Field of sunflowers
3. Landscape with cypress trees
4. Cafe terrace at night
5. Vase with irises
6. Portrait in expressive brushstrokes

**Output Resolution:** 512x512 pixels  
**Format:** PNG  
**Location:** `generated_images_<timestamp>/`

---

## Troubleshooting

### GPU Not Available

**Problem:** `CUDA not available` error

**Solution:**
```bash
# Verify you're on a GPU node
nvidia-smi

# If not, request GPU allocation
exit  # Return to login node
salloc -A c01949 -N 1 -n 1 --gres=gpu:1 --partition=gpu -t 2:00:00
ssh $(scontrol show hostname $SLURM_NODELIST)
```

### Out of Memory

**Problem:** `CUDA out of memory` error

**Solutions:**
1. Reduce batch size (already at 1)
2. Use gradient checkpointing:
   ```python
   unet.enable_gradient_checkpointing()
   ```
3. Request node with more GPU memory

### Slow Training

**Problem:** Training is very slow

**Check:**
```bash
# Verify GPU usage
nvidia-smi

# Check if using GPU (should show CUDA)
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Dataset Download Fails

**Problem:** wget fails to download dataset

**Solution:**
```bash
# Try alternative method
curl -O https://efrosgans.eecs.berkeley.edu/cyclegan/datasets/vangogh2photo.zip

# Or download on Mac and transfer
scp vangogh2photo.zip madbala@bigred200.uits.iu.edu:~/sd_vangogh_lora/images/
```

### Module Import Errors

**Problem:** `ModuleNotFoundError` for diffusers/peft/etc.

**Solution:**
```bash
# Ensure virtual environment is activated
source ~/sd_lora_env/bin/activate

# Reinstall dependencies
pip install -r requirements_lora.txt
```

---

## Advanced Usage

### Custom Prompts

Edit `test_vangogh_lora.py` to add your own prompts:

```python
prompts = [
    "your custom prompt here",
    "another prompt",
]
```

### Adjust LoRA Strength

When loading LoRA, you can adjust the strength:

```python
pipe.load_lora_weights("output/vangogh-lora-final", adapter_weight=0.8)
```

Values: 0.0 (no effect) to 1.0 (full effect)

### Training for More Epochs

Edit `train_vangogh_lora.py`:

```python
num_epochs = 20  # Increase from 10
```

### Using Different Base Models

Replace `model_id` in training script:

```python
model_id = "stabilityai/stable-diffusion-2-1"  # SD 2.1
# or
model_id = "runwayml/stable-diffusion-v1-5"    # SD 1.5 (default)
```

---

## File Management

### Check Disk Usage

```bash
du -sh ~/sd_vangogh_lora
```

### Clean Up Space

```bash
# Remove checkpoints (keep final model only)
rm -rf output/checkpoint-epoch-*

# Remove dataset (after training)
rm -rf images/

# Remove generated images after downloading
rm -rf generated_images_*
```

### Download Everything to Mac

```bash
# From your Mac
cd /Users/madhavanbalaji/Documents/CV
rsync -avz --progress \
    madbala@bigred200.uits.iu.edu:~/sd_vangogh_lora/ \
    vangogh_lora_backup/
```

---

## Next Steps

1. **Integrate with Web App:**
   - Add Van Gogh LoRA to your style transfer backend
   - Update `diffusion_inference.py` to load LoRA weights
   - Add "Van Gogh" option to frontend style selector

2. **Train More Styles:**
   - Monet, Picasso, Cezanne datasets available
   - Same process, different dataset URLs

3. **Improve Quality:**
   - Train for more epochs
   - Use larger LoRA rank (r=16 or r=32)
   - Add more training data

4. **Deploy Model:**
   - Convert LoRA to Safetensors format
   - Upload to Hugging Face Hub
   - Use in production app

---

## Resources

- **Diffusers Documentation:** https://huggingface.co/docs/diffusers
- **PEFT Documentation:** https://huggingface.co/docs/peft
- **LoRA Paper:** https://arxiv.org/abs/2106.09685
- **Stable Diffusion:** https://github.com/Stability-AI/stablediffusion

---

## Credits

- **Base Model:** Stable Diffusion 1.5 by RunwayML
- **Dataset:** Van Gogh paintings from CycleGAN project
- **Method:** LoRA fine-tuning with PEFT library
