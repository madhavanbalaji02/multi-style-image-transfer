import os
import PIL.Image
import numpy as np
import torch
import tensorflow as tf

def load_image(path_to_img, max_dim=512):
    """Loads an image from a file, resizes it, and returns a PIL Image."""
    img = PIL.Image.open(path_to_img)
    
    # Convert to RGB if not already
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    long_dim = max(img.size)
    scale = max_dim / long_dim
    
    if scale < 1:
        new_width = int(img.size[0] * scale)
        new_height = int(img.size[1] * scale)
        img = img.resize((new_width, new_height), PIL.Image.Resampling.LANCZOS)
        
    return img

def save_image(image, path):
    """Saves a PIL Image to a file."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path)

def pil_to_tf(pil_image):
    """Converts a PIL Image to a TensorFlow tensor."""
    img = np.array(pil_image)
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = img / 255.0
    img = img[tf.newaxis, :]
    return img

def tf_to_pil(tensor):
    """Converts a TensorFlow tensor to a PIL Image."""
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def pil_to_torch(pil_image, device='cpu'):
    """Converts a PIL Image to a PyTorch tensor (N, C, H, W)."""
    # Resize to multiple of 8 for diffusion models usually, but here we just convert
    # For diffusion, the pipeline handles PIL images directly usually.
    # This is for metrics if needed.
    img = np.array(pil_image).astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1) # HWC -> CHW
    tensor = torch.from_numpy(img).unsqueeze(0) # Add batch dim
    return tensor.to(device)

def torch_to_pil(tensor):
    """Converts a PyTorch tensor to a PIL Image."""
    tensor = tensor.cpu().detach()
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    tensor = tensor.permute(1, 2, 0) # CHW -> HWC
    tensor = tensor.numpy() * 255.0
    tensor = tensor.clip(0, 255).astype(np.uint8)
    return PIL.Image.fromarray(tensor)
