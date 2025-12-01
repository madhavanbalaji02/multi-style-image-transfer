import torch
from diffusers import StableDiffusionImg2ImgPipeline
import os
import glob
from utils import load_image, save_image
from efficiency import Profiler, log_efficiency

def run_diffusion_style_transfer():
    print("Loading Stable Diffusion model...")
    model_id = "runwayml/stable-diffusion-v1-5"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        pipe = pipe.to(device)
        # Disable safety checker to avoid black images for art
        pipe.safety_checker = None
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    content_dir = 'project/data/content'
    style_dir = 'project/data/style'
    output_dir = 'project/outputs/diffusion'
    
    # Get all images
    content_paths = glob.glob(os.path.join(content_dir, '*'))
    style_paths = glob.glob(os.path.join(style_dir, '*'))
    
    # Filter for images
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    content_paths = [p for p in content_paths if os.path.splitext(p)[1].lower() in valid_exts]
    style_paths = [p for p in style_paths if os.path.splitext(p)[1].lower() in valid_exts]
    
    print(f"Found {len(content_paths)} content images and {len(style_paths)} style images.")

    if not content_paths or not style_paths:
        print("No images found. Please check data directories.")
        return

    csv_path = 'project/outputs/efficiency.csv'
    step_counts = [20, 50, 100]

    for content_path in content_paths:
        content_name = os.path.splitext(os.path.basename(content_path))[0]
        content_img = load_image(content_path).convert("RGB")
        
        for style_path in style_paths:
            style_name = os.path.splitext(os.path.basename(style_path))[0]
            # Clean style name for prompt (e.g. "art_nouveau" -> "Art Nouveau")
            clean_style = style_name.replace("_", " ").title()
            prompt = f"a portrait painting in {clean_style} style"
            
            print(f"Processing {content_name} with {style_name} (Prompt: {prompt})...")
            
            for steps in step_counts:
                print(f"  Steps: {steps}")
                
                with Profiler("Diffusion", device) as prof:
                    # Strength 0.75 is a good default for style transfer
                    images = pipe(prompt=prompt, image=content_img, strength=0.75, guidance_scale=7.5, num_inference_steps=steps).images
                    result_img = images[0]
                
                metrics = prof.get_metrics()
                
                # Save output
                output_subdir = os.path.join(output_dir, style_name)
                os.makedirs(output_subdir, exist_ok=True)
                output_path = os.path.join(output_subdir, f"{content_name}_{steps}.png")
                
                save_image(result_img, output_path)
                
                # Log efficiency
                log_efficiency(csv_path, {
                    'image': content_name,
                    'style': style_name,
                    'model': 'Stable Diffusion',
                    'steps': steps,
                    'time': metrics['time'],
                    'memory': metrics['memory']
                })

if __name__ == "__main__":
    run_diffusion_style_transfer()
