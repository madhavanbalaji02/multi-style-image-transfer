---
title: Multi-Style Image Style Transfer
emoji: 🎨
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 5.12.0

app_file: app.py
pinned: false
license: mit
---

# 🎨 Multi-Style Image Style Transfer

Transform photos into art using GAN and custom Van Gogh LoRA!

## Features
- **GAN**: Fast Starry Night style transfer (~10 sec)
- **LoRA**: Custom fine-tuned Van Gogh model (~30 sec with ZeroGPU)

## Models
- GAN: TensorFlow Hub Arbitrary Style Transfer
- LoRA: SD v1.5 + custom adapter trained on 400 Van Gogh paintings
