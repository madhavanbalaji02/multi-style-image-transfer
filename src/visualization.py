import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from PIL import Image
import numpy as np

def create_comparison_grid():
    content_dir = 'project/data/content'
    gan_dir = 'project/outputs/gan'
    diffusion_dir = 'project/outputs/diffusion'
    output_dir = 'project/outputs/comparisons'
    os.makedirs(output_dir, exist_ok=True)

    # Get styles
    styles = [os.path.basename(d) for d in glob.glob(os.path.join(gan_dir, '*'))]
    if not styles:
        styles = [os.path.basename(d) for d in glob.glob(os.path.join(diffusion_dir, '*'))]
    
    # Get content images
    content_images = [os.path.basename(f) for f in glob.glob(os.path.join(content_dir, '*')) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for style in styles:
        for content in content_images:
            content_name = os.path.splitext(content)[0]
            
            # Paths
            content_path = os.path.join(content_dir, content)
            gan_path = os.path.join(gan_dir, style, f"{content_name}.png")
            # Diffusion: use 50 steps as default
            diff_path = os.path.join(diffusion_dir, style, f"{content_name}_50.png")
            
            images = []
            titles = []
            
            if os.path.exists(content_path):
                images.append(Image.open(content_path))
                titles.append("Content")
            
            if os.path.exists(gan_path):
                images.append(Image.open(gan_path))
                titles.append("GAN Output")
            
            if os.path.exists(diff_path):
                images.append(Image.open(diff_path))
                titles.append("Diffusion (50 steps)")
            
            if not images:
                continue
                
            # Plot
            fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
            if len(images) == 1:
                axes = [axes]
            
            for ax, img, title in zip(axes, images, titles):
                ax.imshow(img)
                ax.set_title(title)
                ax.axis('off')
            
            plt.suptitle(f"Style: {style} | Content: {content_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{style}_{content_name}_grid.png"))
            plt.close()

def generate_charts():
    csv_path = 'project/outputs/comparisons/metrics.csv'
    if not os.path.exists(csv_path):
        print("Metrics CSV not found.")
        return
        
    df = pd.read_csv(csv_path)
    output_dir = 'project/outputs/comparisons'
    
    # Bar charts for metrics
    metrics = ['ssim', 'lpips', 'clip_content', 'clip_style']
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        # Group by model and calculate mean
        means = df.groupby('model')[metric].mean()
        means.plot(kind='bar', color=['skyblue', 'salmon'])
        plt.title(f'Average {metric.upper()} by Model')
        plt.ylabel(metric.upper())
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}_comparison.png"))
        plt.close()

    # Efficiency charts
    eff_csv_path = 'project/outputs/efficiency.csv'
    if os.path.exists(eff_csv_path):
        eff_df = pd.read_csv(eff_csv_path)
        
        # Runtime
        plt.figure(figsize=(10, 6))
        eff_means = eff_df.groupby('model')['time'].mean()
        eff_means.plot(kind='bar', color=['lightgreen', 'orange'])
        plt.title('Average Inference Time (s)')
        plt.ylabel('Time (s)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "runtime_comparison.png"))
        plt.close()

if __name__ == "__main__":
    create_comparison_grid()
    generate_charts()
