import torch
import clip
import lpips
import numpy as np
from skimage.metrics import structural_similarity as ssim
from utils import load_image

# Global model cache
_lpips_fn = None
_clip_model = None
_clip_preprocess = None

def load_metrics_models():
    global _lpips_fn, _clip_model, _clip_preprocess
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if _lpips_fn is None:
        print("Loading LPIPS model...")
        _lpips_fn = lpips.LPIPS(net='alex').to(device)
        
    if _clip_model is None:
        print("Loading CLIP model...")
        _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=device)
        
    return _lpips_fn, _clip_model, _clip_preprocess

def calculate_metrics(content_path, output_path, style_name):
    """
    Calculates SSIM, LPIPS, and CLIP scores.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lpips_fn, clip_model, clip_preprocess = load_metrics_models()
    
    # Load images
    content_pil = load_image(content_path)
    output_pil = load_image(output_path)
    
    # Resize output to match content for pixel-wise metrics if needed
    if content_pil.size != output_pil.size:
        output_pil = output_pil.resize(content_pil.size)
    
    # SSIM
    img1 = np.array(content_pil.convert('L'))
    img2 = np.array(output_pil.convert('L'))
    ssim_score = ssim(img1, img2, data_range=255)
    
    # LPIPS
    # Convert PIL to tensor [-1, 1]
    def pil_to_lpips(pil_img):
        img = np.array(pil_img.resize((256, 256))).astype(np.float32) / 127.5 - 1.0
        img = img.transpose(2, 0, 1) # HWC -> CHW
        return torch.from_numpy(img).unsqueeze(0).to(device)
        
    content_lpips = pil_to_lpips(content_pil)
    output_lpips = pil_to_lpips(output_pil)
    lpips_score = lpips_fn(content_lpips, output_lpips).item()
    
    # CLIP
    clean_style = style_name.replace("_", " ").title()
    style_prompt = f"a portrait painting in {clean_style} style"
    
    content_input = clip_preprocess(content_pil).unsqueeze(0).to(device)
    output_input = clip_preprocess(output_pil).unsqueeze(0).to(device)
    text_input = clip.tokenize([style_prompt]).to(device)
    
    with torch.no_grad():
        content_features = clip_model.encode_image(content_input)
        output_features = clip_model.encode_image(output_input)
        text_features = clip_model.encode_text(text_input)
        
        # Normalize
        content_features = content_features / content_features.norm(dim=1, keepdim=True)
        output_features = output_features / output_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        
        # Cosine similarity
        clip_content = (content_features @ output_features.T).item()
        clip_style = (text_features @ output_features.T).item()
        
    return {
        "ssim": float(ssim_score),
        "lpips": float(lpips_score),
        "clip_content": float(clip_content),
        "clip_style": float(clip_style)
    }
