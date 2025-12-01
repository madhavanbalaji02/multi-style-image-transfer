import torch
import clip
import lpips
import numpy as np
import os
import glob
import csv
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from utils import load_image

def calculate_ssim(img1_pil, img2_pil):
    # Convert to grayscale for SSIM
    img1 = np.array(img1_pil.convert('L'))
    img2 = np.array(img2_pil.resize(img1_pil.size).convert('L'))
    return ssim(img1, img2, data_range=255)

def calculate_lpips(loss_fn, img1_tensor, img2_tensor):
    # img tensors should be (1, 3, H, W) normalized [-1, 1]
    return loss_fn(img1_tensor, img2_tensor).item()

def calculate_clip_scores(model, preprocess, device, content_pil, output_pil, style_prompt):
    # Preprocess
    content_input = preprocess(content_pil).unsqueeze(0).to(device)
    output_input = preprocess(output_pil).unsqueeze(0).to(device)
    text_input = clip.tokenize([style_prompt]).to(device)

    with torch.no_grad():
        content_features = model.encode_image(content_input)
        output_features = model.encode_image(output_input)
        text_features = model.encode_text(text_input)

        # Normalize
        content_features = content_features / content_features.norm(dim=1, keepdim=True)
        output_features = output_features / output_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # Cosine similarity
        content_score = (content_features @ output_features.T).item()
        style_score = (text_features @ output_features.T).item()

    return content_score, style_score

def pil_to_lpips(pil_img, device):
    # Convert PIL to tensor [-1, 1]
    img = np.array(pil_img).astype(np.float32) / 127.5 - 1.0
    img = img.transpose(2, 0, 1) # HWC -> CHW
    return torch.from_numpy(img).unsqueeze(0).to(device)

def run_metrics():
    print("Loading metrics models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # LPIPS
    loss_fn_lpips = lpips.LPIPS(net='alex').to(device)
    
    # CLIP
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    
    content_dir = 'project/data/content'
    output_base_dir = 'project/outputs'
    csv_path = os.path.join(output_base_dir, 'comparisons', 'metrics.csv')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Prepare CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'style', 'model', 'steps', 'ssim', 'lpips', 'clip_content', 'clip_style'])
    
    # Find all output images
    # GAN: outputs/gan/<style>/<content>.png
    # Diffusion: outputs/diffusion/<style>/<content>_<steps>.png
    
    models = ['gan', 'diffusion']
    
    for model_type in models:
        model_dir = os.path.join(output_base_dir, model_type)
        if not os.path.exists(model_dir):
            continue
            
        style_dirs = glob.glob(os.path.join(model_dir, '*'))
        for style_dir in style_dirs:
            style_name = os.path.basename(style_dir)
            clean_style = style_name.replace("_", " ").title()
            style_prompt = f"a portrait painting in {clean_style} style"
            
            img_paths = glob.glob(os.path.join(style_dir, '*.png'))
            for img_path in img_paths:
                filename = os.path.basename(img_path)
                name_no_ext = os.path.splitext(filename)[0]
                
                # Parse filename
                if model_type == 'gan':
                    content_name = name_no_ext
                    steps = 'N/A'
                else:
                    # Diffusion: content_steps
                    parts = name_no_ext.rsplit('_', 1)
                    if len(parts) == 2 and parts[1].isdigit():
                        content_name = parts[0]
                        steps = parts[1]
                    else:
                        content_name = name_no_ext
                        steps = 'Unknown'
                
                # Find original content image
                # Try extensions
                content_path = None
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    path = os.path.join(content_dir, content_name + ext)
                    if os.path.exists(path):
                        content_path = path
                        break
                
                if not content_path:
                    print(f"Warning: Content image for {filename} not found.")
                    continue
                
                print(f"Computing metrics for {model_type}/{style_name}/{filename}...")
                
                # Load images
                content_pil = load_image(content_path)
                output_pil = load_image(img_path)
                
                # SSIM
                ssim_score = calculate_ssim(content_pil, output_pil)
                
                # LPIPS
                content_lpips = pil_to_lpips(content_pil.resize((256, 256)), device) # Resize for speed/memory
                output_lpips = pil_to_lpips(output_pil.resize((256, 256)), device)
                lpips_score = calculate_lpips(loss_fn_lpips, content_lpips, output_lpips)
                
                # CLIP
                clip_content, clip_style = calculate_clip_scores(
                    clip_model, clip_preprocess, device, content_pil, output_pil, style_prompt
                )
                
                # Save row
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([content_name, style_name, model_type, steps, ssim_score, lpips_score, clip_content, clip_style])

if __name__ == "__main__":
    run_metrics()
