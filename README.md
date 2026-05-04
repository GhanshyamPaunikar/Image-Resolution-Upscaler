# Image Resolution Upscaler

4× super-resolution powered by **Stable Diffusion x4 Upscaler**. Upload an image, hit upscale, get a crisp 4× result.

---

## Run

```bash
pip install -r requirements.txt
python app.py
# → http://localhost:7860
```

First launch downloads the model (~1.7 GB) once and caches it.

---

## How it works

Uses [`stabilityai/stable-diffusion-x4-upscaler`](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler) — a latent diffusion model conditioned on a low-resolution input. It hallucinates realistic high-frequency detail guided by the prompt, producing sharp 4× outputs.

**Controls:**

| Control | What it does |
|---|---|
| Prompt | Guides the added detail — leave blank for a sensible default |
| Diffusion steps | More steps = better quality, slower (30 is a good balance) |
| Noise level | 0 = faithful to input · 100 = more creative / painterly |

---

## Project Structure

```
├── app.py              # Gradio web UI
├── requirements.txt    # Python dependencies
└── tools/
    └── utils.py        # Shared helpers
```

---

## Tech Stack

| Library | Purpose |
|---|---|
| PyTorch | Tensor computation |
| diffusers | Stable Diffusion x4 pipeline |
| Gradio | Web UI |
| Pillow | Image I/O |

---

## Notes

- The pipeline runs on **CPU with float32** on all platforms. MPS (Apple Silicon) float16 produces incorrect outputs with this model — CPU fallback is intentional.
- Input images are clamped to 128 × 128 before upscaling so the 4× output stays within 512 × 512.
