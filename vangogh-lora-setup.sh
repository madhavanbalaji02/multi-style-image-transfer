#!/bin/bash

# Van Gogh LoRA Training - Complete Setup Script
# Run this on IU Big Red 200 HPC after connecting

set -e

echo "🎨 Van Gogh LoRA Training Setup"
echo "================================"
echo ""

# ===============================
# 1. REQUEST GPU ALLOCATION
# ===============================
echo "Step 1: Requesting GPU allocation..."
echo "Running: salloc -A c01949 -N 1 -n 1 --gres=gpu:1 --partition=gpu -t 2:00:00"
echo ""
echo "NOTE: This command will allocate a GPU node."
echo "After allocation, you'll need to SSH to the compute node."
echo ""
echo "Press Ctrl+C if you want to run this manually."
sleep 3

salloc -A c01949 -N 1 -n 1 --gres=gpu:1 --partition=gpu -t 2:00:00 &
ALLOC_PID=$!

echo "Waiting for allocation..."
wait $ALLOC_PID

echo ""
echo "Allocation granted! Now SSH to the compute node:"
echo "ssh \$(scontrol show hostname \$SLURM_NODELIST)"
echo ""
read -p "Press Enter after you've SSH'd to the compute node..."

# ===============================
# 2. LOAD GPU PYTHON ENVIRONMENT
# ===============================
echo ""
echo "Step 2: Loading Python GPU environment..."
module load python/gpu/3.10.10
python3 --version

# ===============================
# 3. CREATE VIRTUAL ENVIRONMENT
# ===============================
echo ""
echo "Step 3: Creating virtual environment..."
python3 -m venv ~/sd_lora_env
source ~/sd_lora_env/bin/activate
pip install --upgrade pip

echo "✅ Virtual environment created and activated"

# ===============================
# 4. INSTALL DEPENDENCIES
# ===============================
echo ""
echo "Step 4: Installing PyTorch and dependencies..."
echo "This will take several minutes..."

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install diffusers==0.27.2 transformers accelerate datasets safetensors pillow

echo "✅ Dependencies installed"

# ===============================
# 5. SET UP PROJECT DIRECTORIES
# ===============================
echo ""
echo "Step 5: Creating project directories..."
mkdir -p ~/sd_vangogh_lora/images
mkdir -p ~/sd_vangogh_lora/output
cd ~/sd_vangogh_lora

echo "✅ Project structure created"

# ===============================
# 6. DOWNLOAD VAN GOGH DATASET
# ===============================
echo ""
echo "Step 6: Downloading Van Gogh dataset..."
cd images

if [ ! -f vangogh2photo.zip ]; then
    wget https://efrosgans.eecs.berkeley.edu/cyclegan/datasets/vangogh2photo.zip
    unzip vangogh2photo.zip
    mv trainA/* .
    rm -rf trainA trainB vangogh2photo.zip
    echo "✅ Dataset downloaded and extracted"
else
    echo "⏭️  Dataset already exists"
fi

cd ..

echo ""
echo "Images in dataset: $(ls images/*.jpg | wc -l)"

# ===============================
# 7. CREATE TRAINING SCRIPT
# ===============================
echo ""
echo "Step 7: Creating training script..."

cat > train_lora.py << 'SCRIPT_EOF'
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers.utils import make_image_grid
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm

class SimpleImageDataset(Dataset):
    def __init__(self, folder):
        self.paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.jpg', '.png'))]
        print(f"Found {len(self.paths)} images")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB").resize((512, 512))
        # Convert to tensor and normalize to [-1, 1]
        img_tensor = torch.tensor(list(img.getdata())).reshape(512, 512, 3).float() / 127.5 - 1.0
        img_tensor = img_tensor.permute(2, 0, 1)
        return {"pixel_values": img_tensor}

print("🎨 Van Gogh LoRA Training")
print("=" * 50)

# ---------------------------
# Load SD 1.5 Base Model
# ---------------------------
print("\n📦 Loading Stable Diffusion 1.5...")
model_id = "runwayml/stable-diffusion-v1-5"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    safety_checker=None
).to(device)

unet = pipe.unet
text_encoder = pipe.text_encoder

# Freeze base model
unet.requires_grad_(False)
text_encoder.requires_grad_(False)

print("✅ Model loaded")

# ---------------------------
# Enable LoRA
# ---------------------------
print("\n🔧 Configuring LoRA...")
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=4,
    lora_alpha=4,
    target_modules=["to_q", "to_v"],
    lora_dropout=0.0,
    bias="none",
)

unet = get_peft_model(unet, lora_config)
unet.print_trainable_parameters()

print("✅ LoRA configured")

# ---------------------------
# Dataset
# ---------------------------
print("\n📂 Loading dataset...")
dataset = SimpleImageDataset("images")
loader = DataLoader(dataset, batch_size=1, shuffle=True)

print(f"✅ Dataset loaded: {len(dataset)} images")

# ---------------------------
# Training Setup
# ---------------------------
optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4)
num_epochs = 5

print(f"\n🏋️ Training for {num_epochs} epochs...")

# ---------------------------
# Training Loop
# ---------------------------
unet.train()

for epoch in range(num_epochs):
    epoch_loss = 0
    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for batch_idx, batch in enumerate(progress_bar):
        imgs = batch["pixel_values"].to(device, dtype=torch.float16)
        
        # Add noise
        noise = torch.randn_like(imgs)
        timesteps = torch.randint(
            0, 
            pipe.scheduler.config.num_train_timesteps, 
            (imgs.shape[0],), 
            device=device
        ).long()
        
        noisy = pipe.scheduler.add_noise(imgs, noise, timesteps)
        
        # Get text embeddings
        prompt_embeds = text_encoder(
            pipe.tokenizer(
                ["a painting in the style of Van Gogh"] * imgs.shape[0],
                padding="max_length",
                max_length=pipe.tokenizer.model_max_length,
                return_tensors="pt"
            ).input_ids.to(device)
        )[0]
        
        # Predict noise
        model_pred = unet(noisy, timesteps, encoder_hidden_states=prompt_embeds).sample
        
        # Calculate loss
        loss = torch.nn.functional.mse_loss(model_pred, noise)
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})
    
    avg_loss = epoch_loss / len(loader)
    print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")

print("\n✅ Training complete!")

# ---------------------------
# Save LoRA Weights
# ---------------------------
print("\n💾 Saving LoRA weights...")
unet.save_pretrained("output/vangogh-lora")
print("✅ LoRA weights saved to: output/vangogh-lora")

print("\n🎉 Training finished! Use the inference script to test.")
SCRIPT_EOF

echo "✅ Training script created: train_lora.py"

# ===============================
# 8. CREATE INFERENCE SCRIPT
# ===============================
echo ""
echo "Step 8: Creating inference script..."

cat > test_lora.py << 'INFER_EOF'
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

print("🎨 Van Gogh LoRA Inference Test")
print("=" * 50)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load base model
print("\n📦 Loading Stable Diffusion 1.5...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to(device)

# Load LoRA weights
print("🔧 Loading LoRA weights...")
pipe.unet = pipe.unet.from_pretrained("output/vangogh-lora")
pipe.unet.to(device)

print("✅ Model ready")

# Test prompts
prompts = [
    "A landscape in Van Gogh style",
    "A starry night over a village",
    "A sunset with swirling clouds",
    "A field of sunflowers"
]

print(f"\n🎨 Generating {len(prompts)} test images...")

for i, prompt in enumerate(prompts):
    print(f"\n{i+1}. Generating: '{prompt}'")
    image = pipe(prompt, num_inference_steps=50).images[0]
    
    filename = f"vangogh_test_{i+1}.png"
    image.save(filename)
    print(f"   ✅ Saved: {filename}")

print("\n🎉 All test images generated!")
print("Check the current directory for output images.")
INFER_EOF

echo "✅ Inference script created: test_lora.py"

# ===============================
# FINAL INSTRUCTIONS
# ===============================
echo ""
echo "═══════════════════════════════════════════════════"
echo "✨ Setup Complete!"
echo "═══════════════════════════════════════════════════"
echo ""
echo "📍 Current directory: $(pwd)"
echo "📊 Dataset size: $(ls images/*.jpg | wc -l) images"
echo ""
echo "Next steps:"
echo ""
echo "1️⃣  Start training:"
echo "   python train_lora.py"
echo ""
echo "2️⃣  After training completes (~30-60 minutes), test the model:"
echo "   python test_lora.py"
echo ""
echo "3️⃣  Download results to your Mac:"
echo "   scp madbala@bigred200.uits.iu.edu:~/sd_vangogh_lora/vangogh_test_*.png ."
echo ""
echo "📝 Logs will show training progress and loss values."
echo ""
