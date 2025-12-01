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

    # Add a dummy Hyperparameters class matching friend's repo
    class Hyperparameters:
        def __init__(self):
            # Attributes found in friend's repo
            self.dataset_path = 'monet2photo'
            self.input_nc = 3
            self.output_nc = 3
            self.ngf = 64
            self.ndf = 64
            self.batch_size = 1
            self.n_epochs = 100
            self.n_epochs_decay = 100
            self.lr = 0.0002
            self.beta1 = 0.5
            self.beta2 = 0.999
            self.lambda_A = 10.0
            self.lambda_B = 10.0
            self.pool_size = 50
            self.img_height = 256
            self.img_width = 256
            self.n_residual_blocks = 9
    hyperparameters_mock.Hyperparameters = Hyperparameters
    
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
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Check if it's a full checkpoint or just weights
        if isinstance(checkpoint, dict) and "G_BA" in checkpoint:
            print("Detected full CycleGAN checkpoint. Extracting G_BA (Photo -> Painting)...")
            # G_BA is typically the generator for Domain B (Photo) -> Domain A (Painting)
            state_dict = checkpoint["G_BA"]
            
            # Remove 'module.' prefix if present (from DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace("module.", "")
                new_state_dict[name] = v
            generator.load_state_dict(new_state_dict)
            
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
             generator.load_state_dict(checkpoint["model"])
             
        else:
            generator.load_state_dict(checkpoint)
            
    except Exception as e:
        print(f"Error loading GAN model: {e}")
        # Fallback: Try G_AB if G_BA fails or wasn't found
        if isinstance(checkpoint, dict) and "G_AB" in checkpoint:
             print("Retrying with G_AB...")
             generator.load_state_dict(checkpoint["G_AB"])
        else:
            raise e

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
