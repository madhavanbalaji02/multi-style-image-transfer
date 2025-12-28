import gradio as gr
import spaces
import numpy as np
from PIL import Image
import os
import torch
import torch.nn as nn
from torchvision import transforms

import sys
from types import ModuleType

print(f"Gradio Version: {gr.__version__}")

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

# ===== CycleGAN Style Transfer =====
print("Loading CycleGAN style transfer model...")
# Force CPU loading for consistency with localhost
DEVICE = torch.device("cpu")

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
    """CycleGAN Generator network."""
    
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9):
        super(Generator, self).__init__()
        
        # Initial convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, 7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling
        for i in range(2):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, 3, stride=2, padding=1),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(inplace=True)
            ]
        
        # Residual blocks
        for i in range(n_blocks):
            model += [ResidualBlock(ngf * 4)]
        
        # Upsampling
        for i in range(2):
            mult = 2 ** (2 - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), 3,
                                 stride=2, padding=1, output_padding=1),
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
    
    def forward(self, x):
        return self.model(x)


# Global model cache
cyclegan_model = None
CYCLEGAN_ERROR = None

def load_cyclegan():
    global cyclegan_model, CYCLEGAN_ERROR
    if cyclegan_model is None:
        try:
            print("Loading CycleGAN from checkpoint...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            generator = Generator().to(device)
            
            # Load checkpoint
            model_path = "./cyclegan/cycle_gan_model.pth"
            if os.path.exists(model_path):
                # weights_only=False is required for pickled models with custom classes
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                
                # Use G_AB (Van Gogh style - matches localhost implementation)
                if isinstance(checkpoint, dict) and "G_AB" in checkpoint:
                    print("Loading G_AB (CycleGAN generator)...")
                    state_dict = checkpoint["G_AB"]
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        name = k.replace("module.", "")
                        new_state_dict[name] = v
                    generator.load_state_dict(new_state_dict, strict=False)
                    print("Loaded G_AB generator")
                elif isinstance(checkpoint, dict) and "G_BA" in checkpoint:
                    print("Fallback: Using G_BA...")
                    state_dict = checkpoint["G_BA"]
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        name = k.replace("module.", "")
                        new_state_dict[name] = v
                    generator.load_state_dict(new_state_dict, strict=False)
                else:
                    raise ValueError("Neither G_AB nor G_BA found in checkpoint")
                
                generator.eval()
                cyclegan_model = (generator, device)
                print("CycleGAN model loaded!")
                CYCLEGAN_ERROR = None
            else:
                raise FileNotFoundError(f"Model file not found: {model_path}")
        except Exception as e:
            error_msg = f"Failed to load CycleGAN: {str(e)}"
            print(error_msg)
            CYCLEGAN_ERROR = error_msg
            import traceback
            traceback.print_exc()
            return None
    
    return cyclegan_model


def cyclegan_style_transfer(content_image):
    """Apply CycleGAN Van Gogh style transfer."""
    if content_image is None:
        return None
        
    # Attempt to load model
    model_data = load_cyclegan()
    
    # Check for loading errors
    global CYCLEGAN_ERROR
    if CYCLEGAN_ERROR:
        raise gr.Error(CYCLEGAN_ERROR)
        
    if model_data is None:
        raise gr.Error("Model failed to load for unknown reasons")
        
    try:
        generator, device = model_data
        
        # Transform
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        img_tensor = transform(content_image).unsqueeze(0).to(device)
        
        # Generate
        with torch.no_grad():
            output = generator(img_tensor)
        
        # Denormalize
        output = (output + 1) / 2.0
        output = output.clamp(0, 1)
        
        # Convert to PIL
        to_pil = transforms.ToPILImage()
        return to_pil(output.squeeze(0).cpu())
        
    except Exception as e:
        print(f"CycleGAN inference error: {e}")
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Inference failed: {str(e)}")

@spaces.GPU(duration=60)
def lora_style_transfer(content_image, strength):
    """Van Gogh LoRA style transfer - runs on GPU via ZeroGPU"""
    # Lazy import to avoid startup crashes on macOS
    print("Lazy importing diffusers...")
    from diffusers import StableDiffusionImg2ImgPipeline
    from peft import PeftModel
    print("Diffusers imported successfully.")
    
    if content_image is None:
        return None
    try:
        # Load model inside GPU context
        # Determine device
        if torch.cuda.is_available():
            device = "cuda"
        # Avoid MPS on mac due to mutex crashes - fall back to CPU
        else:
            device = "cpu"
            
        print(f"Loading SD + LoRA on {device}...")
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32, 
            use_safetensors=True
        ).to(device)
        
        # Load LoRA weights if available
        if os.path.exists("./lora_weights"):
            pipe.unet = PeftModel.from_pretrained(pipe.unet, "./lora_weights")
        
        pipe.safety_checker = None
        print("LoRA loaded on GPU!")
        
        # Run inference
        img = content_image.convert("RGB").resize((512, 512))
        result = pipe(
            prompt="painting in the style of Vincent Van Gogh with brushstrokes",
            image=img,
            strength=strength,
            guidance_scale=7.5,
            num_inference_steps=20
        ).images[0]
        return result
    except Exception as e:
        print(f"LoRA error: {e}")
        import traceback
        traceback.print_exc()
        return content_image

# ===== Gradio UI =====
gan_interface = gr.Interface(
    fn=cyclegan_style_transfer,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Image(type="pil", label="Van Gogh CycleGAN Style"),
    title="🎨 Van Gogh CycleGAN",
    description="CycleGAN style transfer trained on Van Gogh paintings (~10 sec)",
)

lora_interface = gr.Interface(
    fn=lora_style_transfer,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Slider(0.4, 0.8, value=0.6, step=0.1, label="Style Strength")
    ],
    outputs=gr.Image(type="pil", label="Van Gogh LoRA Result"),
    title="🌟 Van Gogh LoRA (GPU)",
    description="Custom fine-tuned model - uses ZeroGPU (~30 sec)",
)

demo = gr.TabbedInterface(
    [gan_interface, lora_interface],
    ["CycleGAN (Van Gogh)", "LoRA (Fine-tuned)"],
    title="Multi-Style Image Style Transfer"
)

demo.queue().launch(server_name="0.0.0.0", server_port=7860)
