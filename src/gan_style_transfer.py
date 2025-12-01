import tensorflow as tf
import tensorflow_hub as hub
import os
import time
import glob
from utils import load_image, save_image, pil_to_tf, tf_to_pil
from efficiency import log_efficiency

# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

def run_gan_style_transfer():
    print("Loading GAN model...")
    # Load the model. This is the one used in the repo's notebook (or similar)
    hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
    hub_module = hub.load(hub_handle)
    
    content_dir = 'project/data/content'
    style_dir = 'project/data/style'
    output_dir = 'project/outputs/gan'
    
    # Get all images
    content_paths = glob.glob(os.path.join(content_dir, '*'))
    style_paths = glob.glob(os.path.join(style_dir, '*'))
    
    # Filter for images
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    content_paths = [p for p in content_paths if os.path.splitext(p)[1].lower() in valid_exts]
    style_paths = [p for p in style_paths if os.path.splitext(p)[1].lower() in valid_exts]
    
    print(f"Found {len(content_paths)} content images and {len(style_paths)} style images.")
    
    if not content_paths or not style_paths:
        print("No images found. Please check data directories.")
        return

    csv_path = 'project/outputs/efficiency.csv'
    
    for content_path in content_paths:
        content_name = os.path.splitext(os.path.basename(content_path))[0]
        
        # Load content image once
        content_img_pil = load_image(content_path)
        content_img = pil_to_tf(content_img_pil)

        for style_path in style_paths:
            style_name = os.path.splitext(os.path.basename(style_path))[0]
            print(f"Processing {content_name} with {style_name}...")
            
            # Load style image
            style_img_pil = load_image(style_path)
            style_img = pil_to_tf(style_img_pil)
            
            # Style Transfer
            start_time = time.time()
            outputs = hub_module(tf.constant(content_img), tf.constant(style_img))
            stylized_img = outputs[0]
            end_time = time.time()
            duration = end_time - start_time
            
            # Save output
            output_subdir = os.path.join(output_dir, style_name)
            os.makedirs(output_subdir, exist_ok=True)
            output_path = os.path.join(output_subdir, f"{content_name}.png")
            
            result_pil = tf_to_pil(stylized_img)
            save_image(result_pil, output_path)
            
            # Log efficiency
            # Note: Memory profiling for TF is complex here, logging 0 or N/A
            log_efficiency(csv_path, {
                'image': content_name,
                'style': style_name,
                'model': 'GAN (TF-Hub)',
                'steps': 'N/A',
                'time': duration,
                'memory': 0 # Placeholder as we can't easily measure TF GPU peak mem with torch tools
            })

if __name__ == "__main__":
    run_gan_style_transfer()
