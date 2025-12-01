from diffusers import StableDiffusionPipeline
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm

class VanGoghDataset(Dataset):
    """Dataset loader for Van Gogh paintings"""
    def __init__(self, folder):
        self.paths = [
            os.path.join(folder, f) 
            for f in os.listdir(folder) 
            if f.endswith(('.jpg', '.png', '.jpeg'))
        ]
        print(f"Found {len(self.paths)} images")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB").resize((512, 512))
        # Normalize to [-1, 1]
        img_tensor = torch.tensor(list(img.getdata())).reshape(512, 512, 3).float()
        img_tensor = (img_tensor / 127.5) - 1.0
        img_tensor = img_tensor.permute(2, 0, 1)
        return {"pixel_values": img_tensor}


def main():
    print("🎨 Van Gogh LoRA Training with PEFT")
    print("=" * 60)
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n📱 Device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load Stable Diffusion 1.5
    print("\n📦 Loading Stable Diffusion 1.5...")
    model_id = "runwayml/stable-diffusion-v1-5"
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipe = pipe.to(device)
    
    unet = pipe.unet
    text_encoder = pipe.text_encoder
    
    # Freeze base models
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    print("✅ Base model loaded")
    
    # Configure LoRA with PEFT
    print("\n🔧 Configuring LoRA...")
    from peft import LoraConfig, get_peft_model
    
    lora_config = LoraConfig(
        r=8,  # Rank
        lora_alpha=8,  # Scaling
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],  # Attention layers
        lora_dropout=0.0,
        bias="none",
    )
    
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    
    print("✅ LoRA configured")
    
    # Load dataset
    print("\n📂 Loading Van Gogh dataset...")
    dataset = VanGoghDataset("images")
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    
    print(f"✅ Dataset loaded: {len(dataset)} images")
    
    # Training setup
    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4)
    num_epochs = 10
    
    print(f"\n🏋️ Starting training for {num_epochs} epochs...")
    print(f"   Batch size: 1")
    print(f"   Learning rate: 1e-4")
    print(f"   Total steps: {len(loader) * num_epochs}")
    
    # Training loop
    unet.train()
    global_step = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            imgs = batch["pixel_values"].to(device, dtype=torch.float16)
            
            # Random timestep
            noise = torch.randn_like(imgs)
            timesteps = torch.randint(
                0,
                pipe.scheduler.config.num_train_timesteps,
                (imgs.shape[0],),
                device=device
            ).long()
            
            # Add noise
            noisy_imgs = pipe.scheduler.add_noise(imgs, noise, timesteps)
            
            # Encode prompt
            prompt = "a painting in the style of Vincent Van Gogh"
            text_inputs = pipe.tokenizer(
                [prompt] * imgs.shape[0],
                padding="max_length",
                max_length=pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).to(device)
            
            text_embeddings = text_encoder(text_inputs.input_ids)[0]
            
            # Predict noise
            noise_pred = unet(
                noisy_imgs,
                timesteps,
                encoder_hidden_states=text_embeddings
            ).sample
            
            # Loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{epoch_loss / (progress_bar.n + 1):.4f}"
            })
        
        avg_epoch_loss = epoch_loss / len(loader)
        print(f"\n📊 Epoch {epoch+1} complete - Avg Loss: {avg_epoch_loss:.4f}")
        
        # Save checkpoint every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint_dir = f"output/checkpoint-epoch-{epoch+1}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            unet.save_pretrained(checkpoint_dir)
            print(f"💾 Checkpoint saved: {checkpoint_dir}")
    
    # Save final model
    print("\n💾 Saving final LoRA weights...")
    output_dir = "output/vangogh-lora-final"
    os.makedirs(output_dir, exist_ok=True)
    unet.save_pretrained(output_dir)
    
    print(f"✅ LoRA weights saved to: {output_dir}")
    print("\n🎉 Training completed successfully!")
    print("\nNext step: Run test_lora.py to generate test images")


if __name__ == "__main__":
    main()
