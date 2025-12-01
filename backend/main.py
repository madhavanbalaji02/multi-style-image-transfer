from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import shutil
import os
import uuid
import asyncio

app = FastAPI(title="Multi-Style Image Style Transfer API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Serve output files
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


# -------- Lazy imports (safe for macOS) --------

def get_apply_gan():
    from gan_inference import apply_gan
    return apply_gan

def get_apply_diffusion():
    from diffusion_inference import apply_diffusion
    return apply_diffusion

def get_calculate_metrics():
    from metrics import calculate_metrics
    return calculate_metrics


# ---------------- ROUTES ----------------

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        ext = os.path.splitext(file.filename)[1] or ".jpg"
        filename = f"{uuid.uuid4()}{ext}"
        path = os.path.join(UPLOAD_DIR, filename)

        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {"filename": filename}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate")
async def generate_style(filename: str = Form(...), style: str = Form(...), model_type: str = Form(...)):
    try:
        content_path = os.path.join(UPLOAD_DIR, filename)
        if not os.path.exists(content_path):
            raise HTTPException(status_code=404, detail="Uploaded image not found")
        
        results = {}
        unique_id = str(uuid.uuid4())
        
        # Van Gogh style: GAN uses friend's CycleGAN, Diffusion uses your LoRA
        # Other styles: Only Diffusion available
        
        if model_type == "gan" or model_type == "both":
            # GAN only works for Van Gogh (friend's CycleGAN model)
            if style.lower() == "vangogh":
                gan_output = os.path.join(OUTPUT_DIR, f"gan_{style}_{unique_id}.jpeg")
                try:
                    apply_gan_func = get_apply_gan()
                    await asyncio.to_thread(apply_gan_func, content_path, style, gan_output)
                    results["gan"] = f"/outputs/{os.path.basename(gan_output)}"
                except Exception as e:
                    print(f"GAN generation error: {e}")
                    results["gan"] = None
            else:
                # GAN not available for non-Van Gogh styles
                results["gan"] = None
                if model_type == "gan":
                    raise HTTPException(status_code=400, detail="GAN model only available for Van Gogh style. Please select Stable Diffusion for other styles.")
        
        if model_type == "diffusion" or model_type == "both":
            diff_output = os.path.join(OUTPUT_DIR, f"diff_{style}_{unique_id}.jpeg")
            try:
                apply_diffusion_func = get_apply_diffusion()
                await asyncio.to_thread(apply_diffusion_func, content_path, style, diff_output)
                results["diffusion"] = f"/outputs/{os.path.basename(diff_output)}"
            except Exception as e:
                print(f"Diffusion generation error: {e}")
                results["diffusion"] = None
        
        return results
    
    except Exception as e:
        print(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/result/{filename}")
async def get_result(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="File not found")


@app.get("/metrics")
async def get_metrics_api(filename: str, style: str, model: str):
    content_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(content_path):
        raise HTTPException(status_code=404, detail="Content not found")

    if model == "gan":
        output_filename = f"gan_{style}_{filename}"
    elif model == "diffusion":
        output_filename = f"diff_{style}_{filename}"
    else:
        raise HTTPException(status_code=400, detail="Invalid model type")

    output_path = os.path.join(OUTPUT_DIR, output_filename)
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Output not found")

    try:
        calculate_metrics = get_calculate_metrics()
        metrics = await asyncio.to_thread(
            calculate_metrics, content_path, output_path, style
        )
        return metrics

    except Exception as e:
        print("Metrics error:", e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
