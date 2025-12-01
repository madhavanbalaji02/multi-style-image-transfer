import torch
from diffusers import StableDiffusionImg2ImgPipeline
from peft import PeftModel
from utils import load_image, save_image
import os

# Global model cache
_diffusion_pipe = None
_vangogh_pipe = None

def load_diffusion_model():
    global _diffusion_pipe
    if _diffusion_pipe is None:
        print("Loading Stable Diffusion model...")
        model_id = "runwayml/stable-diffusion-v1-5"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            dtype = torch.float16 if device == "cuda" else torch.float32
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_id, 
                torch_dtype=dtype,
                use_safetensors=True
            )
            pipe = pipe.to(device)
            pipe.safety_checker = None
            
            if device == "cuda":
                pipe.enable_xformers_memory_efficient_attention()
                
            _diffusion_pipe = pipe
        except Exception as e:
            print(f"Error loading Diffusion model: {e}")
            raise e
            
    return _diffusion_pipe

def load_vangogh_lora_model():
    """Load Van Gogh LoRA fine-tuned model"""
    global _vangogh_pipe
    if _vangogh_pipe is None:
        print("Loading Van Gogh LoRA model...")
        model_id = "runwayml/stable-diffusion-v1-5"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        lora_path = os.path.join(os.path.dirname(__file__), "models/lora/lora_vangogh_final")
        
        try:
            dtype = torch.float16 if device == "cuda" else torch.float32
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                use_safetensors=True
            )
            
            # Load LoRA weights
            if os.path.exists(lora_path):
                print(f"Loading LoRA from: {lora_path}")
                pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
            else:
                print(f"WARNING: LoRA path not found: {lora_path}")
                print("Falling back to base Stable Diffusion model")
            
            pipe = pipe.to(device)
            pipe.safety_checker = None
            
            if device == "cuda":
                pipe.enable_xformers_memory_efficient_attention()
                
            _vangogh_pipe = pipe
        except Exception as e:
            print(f"Error loading Van Gogh LoRA model: {e}")
            raise e
            
    return _vangogh_pipe

def apply_diffusion(content_path, style_name, output_path):
    """
    Applies Diffusion-based style transfer (img2img).
    style_name: Name of the style (e.g., "cubism", "vangogh"). Used in prompt.
    """
    # Use Van Gogh LoRA model for vangogh style, otherwise use base model
    if style_name.lower() == "vangogh":
        pipe = load_vangogh_lora_model()
        prompt = "a painting in the style of Vincent Van Gogh with expressive brushstrokes and vibrant colors"
    else:
        pipe = load_diffusion_model()
        clean_style = style_name.replace("_", " ").title()
        prompt = f"a portrait painting in {clean_style} style"
    
    print(f"Running Diffusion with prompt: '{prompt}'")
    
    # Load content image
    content_img = load_image(content_path).convert("RGB")
    
    # Inference
    images = pipe(
        prompt=prompt, 
        image=content_img, 
        strength=0.75, 
        guidance_scale=7.5, 
        num_inference_steps=20
    ).images
    
    result_img = images[0]
    
    # Save
    save_image(result_img, output_path)
    
    return output_path
