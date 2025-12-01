import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import sys
from types import ModuleType

# Mock 'config' and 'config.hyperparameters' to satisfy pickled model dependencies
if 'config' not in sys.modules:
    config_mock = ModuleType('config')
    sys.modules['config'] = config_mock
    
    hyperparameters_mock = ModuleType('config.hyperparameters')
    
    # Add a dummy Config class to hyperparameters
    class Config:
        pass
    hyperparameters_mock.Config = Config
    
    sys.modules['config.hyperparameters'] = hyperparameters_mock
    config_mock.hyperparameters = hyperparameters_mock

import config

# CycleGAN Generator Architecture (ResNet-based)
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_residual_blocks=9):
        super(GeneratorResNet, self).__init__()

        # Initial convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, 7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# Model cache
_cyclegan_model = None

def load_cyclegan_model():
    """Load CycleGAN model from your friend's training"""
    global _cyclegan_model
    
    if _cyclegan_model is not None:
        return _cyclegan_model
    
    model_path = os.path.join(os.path.dirname(__file__), "models/gan/cycle_gan_model.pth")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"CycleGAN model not found: {model_path}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize generator
    generator = GeneratorResNet()
    
    # Load weights (PyTorch 2.6+ requires weights_only=False for custom classes)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'generator' in checkpoint:
            generator.load_state_dict(checkpoint['generator'])
        elif 'model_state_dict' in checkpoint:
            generator.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            generator.load_state_dict(checkpoint['state_dict'])
        else:
            generator.load_state_dict(checkpoint)
    else:
        generator.load_state_dict(checkpoint)
    
    generator.to(device).eval()
    _cyclegan_model = (generator, device)
    
    print(f"✅ Loaded CycleGAN model from: {model_path}")
    return generator, device


def apply_gan(content_path, style, output_path):
    """
    Apply CycleGAN style transfer
    Note: CycleGAN model is style-agnostic, trained for general artistic style transfer
    """
    generator, device = load_cyclegan_model()
    
    # Load and preprocess image
    image = Image.open(content_path).convert("RGB")
    
    # CycleGAN expects normalized input [-1, 1]
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to model's expected input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Generate styled image
    with torch.no_grad():
        output_tensor = generator(input_tensor)
    
    # Denormalize from [-1, 1] to [0, 1]
    output_tensor = (output_tensor + 1) / 2.0
    output_tensor = output_tensor.cpu().squeeze().clamp(0, 1)
    
    # Convert to PIL and resize back to original size
    out_img = transforms.ToPILImage()(output_tensor)
    out_img = out_img.resize(image.size, Image.LANCZOS)
    out_img.save(output_path)
    
    print(f"✅ CycleGAN style transfer complete: {output_path}")
