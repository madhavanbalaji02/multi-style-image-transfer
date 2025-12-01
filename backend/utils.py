import os
from PIL import Image
import numpy as np

def load_image(path_to_img, max_dim=512):
    """Loads and resizes an image, returns a PIL Image."""
    img = Image.open(path_to_img)

    if img.mode != "RGB":
        img = img.convert("RGB")

    long_dim = max(img.size)
    scale = max_dim / long_dim

    if scale < 1:
        new_width = int(img.size[0] * scale)
        new_height = int(img.size[1] * scale)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return img


def save_image(image, path):
    """Saves a PIL Image to a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path)


# ----------- TF replacement helper functions -----------

def pil_to_tf(pil_image):
    """
    Converts a PIL Image to a NumPy array with batch dimension.
    Keeps function name for backwards compatibility.
    """
    img = np.array(pil_image).astype("float32") / 255.0
    return img[np.newaxis, ...]   # [1,H,W,3]


def tf_to_pil(tensor):
    """
    Converts a NumPy array back to PIL Image.
    Name kept for compatibility with old TF code.
    """
    if tensor.ndim == 4:   # [1,H,W,3]
        tensor = tensor[0]

    tensor = (tensor * 255).clip(0, 255).astype("uint8")
    return Image.fromarray(tensor)
