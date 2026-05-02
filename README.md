# Image Resolution Upscaler

A deep-learning super-resolution system built with PyTorch. It upscales images by **2×, 3×, 4×, or 6×** using Transformer-based neural networks, with full support for multi-scale training, perceptual loss, quantized inference, and a real-time screen overlay.

---

## Features

- **Multi-scale upscaling** — train and infer at 2×, 3×, 4×, and 6× simultaneously using a single model
- **Modular model architecture** — plug in any model by placing `models/<name>/model.py` with a `TransformerModel` class
- **Perceptual loss** — optional LPIPS (VGG) loss alongside L1 for sharper, more realistic textures
- **AMP training** — automatic mixed precision on CUDA and MPS for faster training
- **Quantized inference** — dynamic `qint8` quantization on `nn.Linear` layers to shrink model footprint
- **torch.compile** — optional JIT compilation for accelerated inference
- **Real-time overlay** — `app_overlay.py` captures any on-screen window and upscales it live via OpenCV
- **A/B testing** — compare two model checkpoints head-to-head with PSNR, SSIM, and LPIPS
- **LAM visualisation** — Local Attribution Maps to visualise which input pixels each output pixel depends on
- **Cross-platform** — CUDA, Apple MPS, and CPU all supported

---

## Project Structure

```
├── train.py            # Training script (multi-scale, resumable checkpoints)
├── inference.py        # Single-image upscale with PSNR / SSIM reporting
├── eval_metrics.py     # Batch evaluation on a dataset (Y-channel PSNR / SSIM)
├── benchmark.py        # Throughput benchmark across images
├── speed_test.py       # Per-image inference timing
├── ab_test.py          # A/B comparison of two model checkpoints
├── lam.py              # Local Attribution Map visualisation
├── overlay.py          # Screen-region real-time upscale overlay (mss)
├── app_overlay.py      # Window-selection real-time overlay (cross-platform)
├── requirements.txt    # Python dependencies
└── models/
    └── <ModelName>/
        ├── model.py    # TransformerModel definition
        └── checkpoints/
```

---

## Installation

**Python 3.10+ is recommended.**

```bash
git clone https://github.com/GhanshyamPaunikar/Image-Resolution-Upscaler.git
cd Image-Resolution-Upscaler
pip install -r requirements.txt
```

> **GPU note:** The `requirements.txt` pins CUDA 12.1 wheels. For a different CUDA version or CPU-only install, replace the `torch`/`torchvision` lines with the appropriate wheel from [pytorch.org](https://pytorch.org/get-started/locally/).

---

## Training

```bash
python train.py \
  --data_dir /path/to/images \
  --model Fastv2 \
  --epochs 1000 \
  --batch_size 32 \
  --use_perceptual \
  --lpips_weight 0.4
```

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | required | Directory of training images (`.jpg`) |
| `--model` | `Fastv2` | Model name → loads `models/<model>/model.py` |
| `--epochs` | `1000` | Total training epochs |
| `--batch_size` | `32` | Samples per batch |
| `--lr` | `2e-4` | Learning rate (overrides checkpoint LR if set) |
| `--use_perceptual` | off | Add LPIPS perceptual loss |
| `--lpips_weight` | `0.4` | Weight for LPIPS term |
| `--checkpoint_dir` | `models/<model>/checkpoints` | Where to save/resume checkpoints |
| `--num_workers` | `1` | DataLoader worker threads |
| `--checkpoint_interval` | `1` | Save every N epochs |

Training interleaves 2×, 3×, 4×, and 6× scale pairs in round-robin order each epoch. Checkpoints are saved as `model_epoch_<N>.pth` and automatically resumed on the next run.

---

## Inference

Upscale a single image and compare against bicubic:

```bash
python inference.py \
  --image_path path/to/image.jpg \
  --model Fastv2 \
  --scale 4 \
  --downscale \
  --out upscaled.jpg
```

| Argument | Default | Description |
|---|---|---|
| `--image_path` | required | Input image |
| `--model` | `SwinBased` | Model architecture to use |
| `--scale` | `3` | Upscale factor (2, 3, 4, or 6) |
| `--downscale` | off | Auto-downscale input by `scale` before upscaling |
| `--res_in` | `None` | Fixed input resolution key (overridden by `--downscale`) |
| `--out` | `model.jpg` | Path for upscaled output |
| `--inp` | `input.jpg` | Path to save the downscaled input |
| `--quantize` | off | Apply dynamic `qint8` quantization |
| `--compile` | off | Enable `torch.compile` |

After inference, SSIM and PSNR are printed for both the model output and the bicubic baseline.

---

## Batch Evaluation

Compute average PSNR and SSIM (Y-channel) over an entire dataset:

```bash
python eval_metrics.py \
  --data_dir /path/to/test/images \
  --model Fastv2 \
  --scale 4 \
  --max_images 100
```

Results are printed to the console and saved to `models/<model>/score_<date>_scale<N>_<datadir>.txt`.

---

## A/B Model Comparison

Compare two checkpoints on the same dataset:

```bash
python ab_test.py \
  --data_dir /path/to/images \
  --model_a Fastv2 \
  --model_b SwinBased \
  --scale 4
```

Outputs total and average MSE + LPIPS for both models side-by-side.

---

## Real-Time Screen Overlay

Upscale any live window in real time:

```bash
# Cross-platform window picker (macOS Quartz / Windows / Linux mss)
python app_overlay.py --model Fastv2 --scale 3

# Direct screen-region capture
python overlay.py --top 100 --left 100 --width 1280 --height 720 --model Fastv2
```

The overlay renders the upscaled output in an always-on-top OpenCV window. Press **`q`** to quit.

---

## LAM Visualisation

Visualise which input pixels contribute to each output pixel:

```bash
python lam.py --image_path path/to/image.jpg --model Fastv2 --scale 4
```

An interactive matplotlib window lets you click pixels to generate attribution heatmaps.

---

## Supported Scale Factors

| Factor | LR Input | HR Output |
|--------|----------|-----------|
| 2×     | 48 × 48  | 96 × 96   |
| 3×     | 48 × 48  | 144 × 144 |
| 4×     | 48 × 48  | 192 × 192 |
| 6×     | 48 × 48  | 288 × 288 |

Custom resolutions and non-square inputs are supported — set the `scale_pairs` list in `train.py`.

---

## Tech Stack

| Library | Purpose |
|---|---|
| PyTorch | Model training & inference |
| torchvision | Transforms, pretrained backbones |
| LPIPS | Perceptual loss (VGG) |
| scikit-image | PSNR / SSIM metrics |
| OpenCV | Real-time overlay rendering |
| Pillow | Image I/O |
| Numba | Optional fused GPU post-processing |
| mss | Cross-platform screen capture |
| NumPy / SciPy | Numerical utilities |
