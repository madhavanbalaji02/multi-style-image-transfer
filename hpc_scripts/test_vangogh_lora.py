from diffusers import StableDiffusionPipeline
from peft import PeftModel
import torch
import os
from datetime import datetime

def main():
    print("🎨 Van Gogh LoRA - Inference Test")
    print("=" * 60)
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n📱 Device: {device}")
    
    # Load base model
    print("\n📦 Loading Stable Diffusion 1.5...")
    model_id = "runwayml/stable-diffusion-v1-5"
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    # Load LoRA weights
    lora_path = "output/vangogh-lora-final"
    
    if not os.path.exists(lora_path):
        print(f"\n❌ LoRA weights not found at: {lora_path}")
        print("Please train the model first using train_vangogh_lora.py")
        return
    
    print(f"\n🔧 Loading LoRA from: {lora_path}")
    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
    pipe = pipe.to(device)
    
    print("✅ Model ready with LoRA weights")
    
    # Test prompts
    prompts = [
        "a starry night over a village with swirling sky",
        "a field of bright yellow sunflowers under blue sky",
        "a landscape with cypress trees and rolling hills",
        "a cafe terrace at night with warm lights",
        "a vase with irises on a wooden table",
        "a portrait of a person in expressive brushstrokes",
    ]
    
    # Generation parameters
    num_inference_steps = 50
    guidance_scale = 7.5
    
    print(f"\n🎨 Generating {len(prompts)} images...")
    print(f"   Steps: {num_inference_steps}")
    print(f"   Guidance scale: {guidance_scale}")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"generated_images_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{i}/{len(prompts)} Generating: '{prompt}'")
        
        # Generate image
        image = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=torch.manual_seed(42 + i)  # For reproducibility
        ).images[0]
        
        # Save image
        filename = f"{output_dir}/vangogh_{i:02d}.png"
        image.save(filename)
        print(f"   ✅ Saved: {filename}")
        
        # Also save with descriptive name
        desc_filename = f"{output_dir}/vangogh_{i:02d}_{prompt[:30].replace(' ', '_')}.png"
        image.save(desc_filename)
    
    print(f"\n🎉 All images generated!")
    print(f"📁 Output directory: {output_dir}")
    print("\nTo download to your Mac, run:")
    print(f"   scp -r madbala@bigred200.uits.iu.edu:~/sd_vangogh_lora/{output_dir} .")


if __name__ == "__main__":
    main()
