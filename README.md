# Image Resolution Upscaler

A deep-learning **4× super-resolution** system built with PyTorch. Upload any image and get a crisp 4× upscale in seconds — powered by either a diffusion model or a fast custom transformer.

---

## Web UI

```bash
python app.py
# → http://localhost:7860
```

Two upscaling modes available in the browser:

| Tab | Model | Speed | Quality |
|---|---|---|---|
| **Stable Diffusion ×4** | stabilityai/stable-diffusion-x4-upscaler | ~30 s | Photorealistic, diffusion-guided detail |
| **Custom Model** | Locally trained Fastv2 transformer | ~50 ms | Faithful, PSNR-optimised |

---

## Quick Start

**Requirements:** Python 3.10+

```bash
git clone https://github.com/GhanshyamPaunikar/Image-Resolution-Upscaler.git
cd Image-Resolution-Upscaler
pip install -r requirements.txt
```

**Train a model (needed for the Custom Model tab):**

```bash
# Drop .jpg / .png images into images/training_set/
python quick_train.py      # trains Fastv2 for 100 epochs, saves checkpoint
```

**Launch:**

```bash
python app.py
```

---

## Architecture — Fastv2

A lightweight residual super-resolution network (~3.6 M parameters):

1. **Shallow feature extraction** — single 3×3 conv
2. **16 Residual Channel-Attention Blocks (RCAB)** — dual conv + SE-style attention
3. **Global residual** — skip connection from shallow to deep features
4. **Pixel-shuffle upsampler** — sub-pixel convolution for ×4
5. **Reconstruction conv** — projects back to RGB, output clamped to [0, 1]

---

## Command-Line Tools

```bash
# Single-image inference
python inference.py --image path/to/img.jpg --model Fastv2

# Batch evaluation (PSNR / SSIM)
python eval_metrics.py --data_dir path/to/test --model Fastv2

# Throughput benchmark
python benchmark.py --model Fastv2

# A/B comparison between two checkpoints
python ab_test.py --data_dir path/to/images --model_a Fastv2 --model_b OtherModel

# LAM attribution visualisation
python lam.py --image_path img.jpg --model Fastv2

# Real-time screen overlay
python overlay.py --model Fastv2
```

---

## Project Structure

```
├── app.py              # Gradio web UI (SD x4 + Custom Model tabs)
├── quick_train.py      # Fast training script — Fastv2, 100 epochs
├── train.py            # Full training script (resumable, perceptual loss)
├── inference.py        # CLI single-image upscale
├── eval_metrics.py     # Batch PSNR / SSIM evaluation
├── benchmark.py        # Inference throughput test
├── ab_test.py          # Head-to-head model comparison
├── lam.py              # Local Attribution Map visualisation
├── overlay.py          # Real-time screen-region overlay
├── requirements.txt
├── models/
│   └── Fastv2/
│       ├── model.py            # TransformerModel definition
│       └── checkpoints/        # Saved .pth files
├── images/
│   └── training_set/           # Drop training images here
└── tools/
    └── utils.py                # Checkpoint & model discovery helpers
```

---

## Tech Stack

| Library | Purpose |
|---|---|
| PyTorch | Model training & inference |
| diffusers | Stable Diffusion x4 pipeline |
| Gradio | Web UI |
| scikit-image | PSNR / SSIM metrics |
| torchvision | Image transforms |
| Pillow | Image I/O |
| OpenCV | Real-time overlay |
| NumPy / SciPy | Numerical utilities |

---

## Device Support

| Device | Training | SD x4 | Custom Model |
|---|---|---|---|
| NVIDIA CUDA | ✓ | ✓ | ✓ |
| Apple MPS | ✓ | CPU fallback | ✓ |
| CPU | ✓ | ✓ | ✓ |

> The SD x4 pipeline always runs on CPU with float32 — MPS float16 produces incorrect outputs with this model.
