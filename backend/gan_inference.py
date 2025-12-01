"""
GAN Inference for CycleGAN Van Gogh Style Transfer
Uses friend's proper implementation from GitHub repo
"""
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
    
    # Add Hyperparameters class matching friend's repo
    class Hyperparameters:
        def __init__(self):
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

# Friend's proper Generator architecture
class ResidualBlock(nn.Module):
    """Residual block with instance normalization."""
    
    def __init__(self, in_channels: int):
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Generator(nn.Module):
    """
    Generator network for CycleGAN.
    Architecture: c7s1-64, d128, d256, R256 x 9, u128, u64, c7s1-3
    """
    
    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        ngf: int = 64,
        n_blocks: int = 9,
        use_dropout: bool = False
    ):
        super(Generator, self).__init__()
        
        # Initial convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, 7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, 3, stride=2, padding=1),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(inplace=True)
            ]
        
        # Residual blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResidualBlock(ngf * mult)]
        
        # Upsampling
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    ngf * mult,
                    int(ngf * mult / 2),
                    3,
                    stride=2,
                    padding=1,
                    output_padding=1
                ),
                nn.InstanceNorm2d(int(ngf * mult / 2)),
                nn.ReLU(inplace=True)
            ]
        
        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, 7),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)


# Global model cache
_cyclegan_model = None

def load_cyclegan_model():
    """Load the CycleGAN model (cached)."""
    global _cyclegan_model
    
    if _cyclegan_model is not None:
        return _cyclegan_model
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(os.path.dirname(__file__), "models", "gan", "cycle_gan_model.pth")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"GAN model not found at: {model_path}")
    
    # Initialize generator with friend's architecture
    generator = Generator(
        input_nc=3,
        output_nc=3,
        ngf=64,
        n_blocks=9
    )
    
    # Load weights
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Check if it's a full checkpoint or just weights
        if isinstance(checkpoint, dict) and "G_BA" in checkpoint:
            print("Detected full CycleGAN checkpoint. Extracting G_BA (Photo -> Painting)...")
            state_dict = checkpoint["G_BA"]
            
            # Remove 'module.' prefix if present (from DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace("module.", "")
                new_state_dict[name] = v
            generator.load_state_dict(new_state_dict, strict=False)
            
        elif isinstance(checkpoint, dict) and "G_AB" in checkpoint:
            print("Using G_AB (Painting -> Photo direction)...")
            state_dict = checkpoint["G_AB"]
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace("module.", "")
                new_state_dict[name] = v
            generator.load_state_dict(new_state_dict, strict=False)
            
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
             generator.load_state_dict(checkpoint["model"], strict=False)
             
        else:
            generator.load_state_dict(checkpoint, strict=False)
            
    except Exception as e:
        print(f"Error loading GAN model: {e}")
        raise e

    generator.to(device).eval()
    _cyclegan_model = (generator, device)
    
    print(f"✅ Loaded CycleGAN model from: {model_path}")
    return _cyclegan_model


def apply_gan(content_path, style, output_path):
    """Apply GAN style transfer."""
    generator, device = load_cyclegan_model()
    
    # Load and preprocess image
    image = Image.open(content_path).convert('RGB')
    
    # Transform matching friend's inference code
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Generate styled image
    with torch.no_grad():
        styled_tensor = generator(image_tensor)
    
    # Denormalize and save
    styled_tensor = (styled_tensor + 1) / 2.0  # From [-1, 1] to [0, 1]
    styled_tensor = styled_tensor.clamp(0, 1)
    
    # Convert to PIL Image
    to_pil = transforms.ToPILImage()
    styled_image = to_pil(styled_tensor.squeeze(0).cpu())
    
    # Save
    styled_image.save(output_path, quality=95)
    print(f"✅ CycleGAN style transfer complete: {output_path}")
