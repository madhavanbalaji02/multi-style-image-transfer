import torch
from torchvision import transforms
from PIL import Image
import os

BASE_MODEL_DIR = os.path.join(os.path.dirname(__file__), "neural_style/models")

VALID_STYLES = {
    "cubism": "mosaic.pth",
    "expressionism": "candy.pth",
    "impressionism": "udnie.pth",
    "rain_princess": "rain_princess.pth"
}

_MODEL_CACHE = {}

def _load_style_model(style):
    if style not in VALID_STYLES:
        raise KeyError(f"Unknown style '{style}'. Valid styles: {list(VALID_STYLES.keys())}")

    model_file = VALID_STYLES[style]
    model_path = os.path.join(BASE_MODEL_DIR, model_file)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file missing: {model_path}")

    if model_file in _MODEL_CACHE:
        return _MODEL_CACHE[model_file]

    from neural_style.neural_style import TransformerNet

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TransformerNet()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    _MODEL_CACHE[model_file] = (model, device)
    return model, device


def apply_gan(content_path, style, output_path):
    model, device = _load_style_model(style)

    image = Image.open(content_path).convert("RGB")
    transform = transforms.ToTensor()
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output_tensor = model(input_tensor).cpu()

    output_tensor = output_tensor.squeeze().clamp(0, 1)
    out_img = transforms.ToPILImage()(output_tensor)
    out_img.save(output_path)
