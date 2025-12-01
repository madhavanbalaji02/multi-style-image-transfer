# Multi-Style Image Style Transfer

Transform photos into art using GANs and Stable Diffusion models, including a custom Van Gogh LoRA fine-tuned model.

![Project Demo](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Diffusion](https://img.shields.io/badge/Stable%20Diffusion-v1.5-orange)

## 🎨 Features

- **Multiple Art Styles**: Cubism, Expressionism, Fauvism, Renaissance, Pop Art, Impressionism
- **Van Gogh LoRA**: Custom fine-tuned model trained on 400 Van Gogh paintings (HPC GPU-trained)
- **Dual Models**: 
  - GAN (Fast neural style transfer)
  - Stable Diffusion (High-quality img2img)
- **Web Interface**: Beautiful, modern UI with real-time generation

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd project
```

2. **Install backend dependencies**
```bash
cd backend
pip install -r requirements.txt
```

3. **Download Van Gogh LoRA model** (optional)
```bash
# Model weights not included in repo due to size
# Contact repo owner or train your own (see HPC_SETUP.md)
mkdir -p models/lora
# Place lora_vangogh_final/ folder here
```

4. **Start the application**
```bash
# Terminal 1 - Backend
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 - Frontend
cd frontend
python3 -m http.server 3000
```

5. **Open browser**
```
http://localhost:3000
```

## 📁 Project Structure

```
project/
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── gan_inference.py        # GAN style transfer
│   ├── diffusion_inference.py  # Stable Diffusion + LoRA
│   ├── metrics.py              # Quality metrics (LPIPS, CLIP)
│   ├── utils.py                # Helper functions
│   ├── requirements.txt        # Python dependencies
│   └── models/                 # Model weights (not in repo)
│
├── frontend/
│   ├── index.html              # Web UI
│   ├── styles.css              # Styling
│   └── script.js               # Frontend logic
│
├── hpc_scripts/
│   ├── train_vangogh_lora.py   # LoRA training script
│   ├── test_vangogh_lora.py    # LoRA inference test
│   └── requirements_lora.txt   # Training dependencies
│
├── .gitignore
├── README.md
├── DEPLOYMENT.md               # Render.com deployment guide
├── HPC_SETUP.md                # IU Big Red 200 training guide
└── LOCAL_DEV.md                # Local development reference
```

## 🎨 Available Styles

### Standard Diffusion Styles
- Cubism
- Expressionism
- Fauvism
- Renaissance
- Pop Art
- Impressionism

### Custom LoRA Model ⭐
- **Van Gogh (LoRA Fine-tuned)**
  - Trained on 400 authentic Van Gogh paintings
  - 10 epochs on A100 GPU
  - Authentic brushstrokes and color palette
  - 6.1 MB LoRA adapter

## 🔧 API Endpoints

- `POST /upload` - Upload image for processing
- `POST /generate` - Generate styled image
- `GET /result/{filename}` - Retrieve generated image
- `POST /metrics` - Calculate quality metrics
- `GET /docs` - Interactive API documentation

## 📊 Model Details

### GAN Model
- **Architecture**: Fast Neural Style Transfer (PyTorch)
- **Styles**: Mosaic, Candy, Udnie, Rain Princess
- **Speed**: ~2-5 seconds per image

### Diffusion Model
- **Base**: Stable Diffusion v1.5
- **Pipeline**: Img2Img
- **Speed**: ~20-30 seconds per image
- **Quality**: High-resolution, detailed

### Van Gogh LoRA
- **Base**: SD v1.5 + PEFT LoRA
- **Rank**: 8
- **Target Modules**: to_q, to_k, to_v, to_out.0
- **Training**: 400 Van Gogh paintings, 10 epochs
- **Size**: 6.1 MB (adapter only)

## 🎓 Training Your Own LoRA

See [HPC_SETUP.md](HPC_SETUP.md) for complete guide on training custom LoRA models on IU Big Red 200 HPC.

**Quick steps:**
1. Request GPU allocation
2. Install dependencies
3. Download dataset
4. Run `train_vangogh_lora.py`
5. Download trained weights

## 🚀 Deployment

### Local Development
```bash
./start-local.sh  # Starts both backend and frontend
```

### Production (Render.com)
See [DEPLOYMENT.md](DEPLOYMENT.md) for complete guide.

## 📝 Requirements

### Backend
- fastapi
- uvicorn
- torch
- diffusers
- transformers
- peft (for LoRA)
- pillow
- numpy

### Frontend
- Modern web browser
- No build step required

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- **Stable Diffusion** by RunwayML
- **PEFT** by Hugging Face
- **Van Gogh Dataset** from CycleGAN project
- **IU Big Red 200** HPC Resources

## 📧 Contact

For questions about the Van Gogh LoRA model or training setup, please open an issue.

---

**Note:** Model weights (`*.safetensors`, `*.pth`) are not included in this repository due to size. Train your own using the provided scripts or contact the maintainer.
