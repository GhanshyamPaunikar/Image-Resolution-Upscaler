#!/usr/bin/env python
"""
app.py — Gradio web interface for the Image Resolution Upscaler.

Run with:
    python app.py
Then open http://localhost:7860
"""
import importlib
import logging
import os
import time

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from torchvision import transforms

from tools.utils import get_available_models, get_latest_checkpoint

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

DEVICE = torch.device(
    "mps"  if torch.backends.mps.is_available()  else
    "cuda" if torch.cuda.is_available()           else
    "cpu"
)
log.info("Device: %s", DEVICE)

SCALE = 4  # fixed upscale factor throughout

# ---------------------------------------------------------------------------
# SD x4 Upscaler pipeline
# MPS + float16 produces numerically incorrect outputs with this model;
# force CPU + float32 for the diffusion pipeline regardless of available device.
# ---------------------------------------------------------------------------
_sd_pipe = None
_SD_DEVICE = "cpu"
_SD_DTYPE  = torch.float32


def _get_sd_pipe():
    global _sd_pipe
    if _sd_pipe is None:
        from diffusers import StableDiffusionUpscalePipeline
        log.info("Loading SD x4 Upscaler on CPU (float32)…")
        _sd_pipe = StableDiffusionUpscalePipeline.from_pretrained(
            "stabilityai/stable-diffusion-x4-upscaler",
            torch_dtype=_SD_DTYPE,
        ).to(_SD_DEVICE)
        _sd_pipe.enable_attention_slicing()
        log.info("SD x4 Upscaler ready.")
    return _sd_pipe


def run_sd_upscale(
    image: Image.Image | None,
    prompt: str,
    steps: int,
    noise: float,
    downscale_first: bool,
) -> tuple[Image.Image | None, str]:
    if image is None:
        return None, "Upload an image to get started."

    orig = image.convert("RGB")

    if downscale_first:
        ow, oh = orig.size
        lr = orig.resize((ow // SCALE, oh // SCALE), Image.LANCZOS)
    else:
        lr = orig

    # Clamp LR so output stays ≤ 512×512 (model limit)
    max_lr_side = 128
    w, h = lr.size
    if max(w, h) > max_lr_side:
        scale_down = max_lr_side / max(w, h)
        lr = lr.resize((int(w * scale_down), int(h * scale_down)), Image.LANCZOS)

    try:
        pipe = _get_sd_pipe()
    except Exception as e:
        return None, f"Failed to load SD pipeline: {e}"

    t0 = time.perf_counter()
    result = pipe(
        prompt=prompt or "high resolution, sharp, detailed",
        image=lr,
        num_inference_steps=steps,
        noise_level=int(noise),
    ).images[0]
    elapsed = time.perf_counter() - t0

    info = "\n".join([
        f"Model    stabilityai/stable-diffusion-x4-upscaler",
        f"Device   CPU  (float32)",
        f"Steps    {steps}   Noise {int(noise)}",
        f"Input    {lr.size[0]} × {lr.size[1]}   →   Output  {result.size[0]} × {result.size[1]}",
        f"Time     {elapsed:.1f} s",
    ])
    return result, info


# ---------------------------------------------------------------------------
# Custom transformer model
# ---------------------------------------------------------------------------
_custom_cache: dict[str, nn.Module] = {}


def _load_custom_model(model_name: str, quantize: bool) -> nn.Module:
    key = f"{model_name}|{quantize}"
    if key not in _custom_cache:
        module = importlib.import_module(f"models.{model_name}.model")
        model = module.TransformerModel().to(DEVICE)
        ckpt_path, epoch = get_latest_checkpoint(f"models/{model_name}/checkpoints")
        log.info("Loading %s epoch %d", model_name, epoch)
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state)
        model.eval()
        if quantize:
            model = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        _custom_cache[key] = model
    return _custom_cache[key]


def run_custom_upscale(
    image: Image.Image | None,
    model_name: str,
    downscale_first: bool,
    quantize: bool,
) -> tuple[Image.Image | None, Image.Image | None, str]:
    if image is None:
        return None, None, "Upload an image to get started."

    available = get_available_models()
    if not available:
        return None, None, "No trained models found.\nRun:  python quick_train.py"
    if model_name not in available:
        return None, None, f"Model '{model_name}' not found. Available: {', '.join(available)}"

    try:
        model = _load_custom_model(model_name, quantize)
    except Exception as e:
        return None, None, f"Load error: {e}"

    to_tensor = transforms.ToTensor()
    to_pil    = transforms.ToPILImage()
    orig = image.convert("RGB")

    lr = orig.resize((orig.size[0] // SCALE, orig.size[1] // SCALE), Image.BICUBIC) \
         if downscale_first else orig

    lr_t = to_tensor(lr).unsqueeze(0).to(DEVICE)
    t0 = time.perf_counter()
    with torch.no_grad():
        if DEVICE.type in ("cuda", "mps"):
            with torch.amp.autocast(device_type=DEVICE.type, dtype=torch.float16):
                out = model(lr_t, upscale_factor=SCALE)
        else:
            out = model(lr_t, upscale_factor=SCALE)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    sr      = to_pil(out.squeeze(0).cpu().clamp(0, 1))
    bicubic = lr.resize((lr.size[0] * SCALE, lr.size[1] * SCALE), Image.BICUBIC)

    lines = [
        f"Model    {model_name}   ×4",
        f"Device   {DEVICE.type.upper()}",
        f"Input    {lr.size[0]} × {lr.size[1]}   →   Output  {sr.size[0]} × {sr.size[1]}",
        f"Time     {elapsed_ms:.1f} ms",
    ]

    if downscale_first:
        ref     = orig.resize(sr.size, Image.BICUBIC) if orig.size != sr.size else orig
        ref_arr = np.array(ref).astype(np.float32) / 255.0
        sr_arr  = np.array(sr).astype(np.float32) / 255.0
        bic_arr = np.array(bicubic.resize(ref.size, Image.BICUBIC)).astype(np.float32) / 255.0
        sr_psnr  = compare_psnr(ref_arr, sr_arr,  data_range=1.0)
        sr_ssim  = compare_ssim(ref_arr, sr_arr,  data_range=1.0, channel_axis=-1)
        bic_psnr = compare_psnr(ref_arr, bic_arr, data_range=1.0)
        bic_ssim = compare_ssim(ref_arr, bic_arr, data_range=1.0, channel_axis=-1)
        lines += [
            "",
            f"PSNR     Model {sr_psnr:.2f} dB   Bicubic {bic_psnr:.2f} dB   Gain +{sr_psnr-bic_psnr:.2f} dB",
            f"SSIM     Model {sr_ssim:.4f}   Bicubic {bic_ssim:.4f}   Gain +{sr_ssim-bic_ssim:.4f}",
        ]

    return sr, bicubic, "\n".join(lines)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
_CSS = """
/* ── Reset / base ─────────────────────────────────────────────────────────── */
body, .gradio-container { background: #f0f2f5 !important; }
footer { display: none !important; }

/* ── Header ───────────────────────────────────────────────────────────────── */
.app-header {
  text-align: center;
  padding: 2rem 0 0.25rem;
}
.app-header h2 {
  font-size: 1.5rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: #111827;
  margin: 0;
}
.app-sub {
  text-align: center;
  color: #6b7280;
  font-size: 0.8rem;
  margin: 0.2rem 0 1.5rem;
}

/* ── Cards ────────────────────────────────────────────────────────────────── */
.card {
  background: #ffffff;
  border-radius: 12px !important;
  box-shadow: 0 1px 3px rgba(0,0,0,.10), 0 1px 2px rgba(0,0,0,.08) !important;
  border: none !important;
  padding: 1.25rem !important;
}

/* ── Tabs ─────────────────────────────────────────────────────────────────── */
.tab-nav button {
  font-weight: 500;
  font-size: 0.875rem;
  border-radius: 8px 8px 0 0 !important;
}

/* ── Inputs / outputs ─────────────────────────────────────────────────────── */
.block { border-radius: 8px !important; border: 1px solid #e5e7eb !important; }
label { font-weight: 500 !important; font-size: 0.8rem !important; color: #374151 !important; }

/* ── Primary button ───────────────────────────────────────────────────────── */
button.primary {
  background: #4f46e5 !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
  letter-spacing: 0.01em !important;
  box-shadow: 0 1px 2px rgba(79,70,229,.3) !important;
}
button.primary:hover {
  background: #4338ca !important;
}

/* ── Metrics box ──────────────────────────────────────────────────────────── */
textarea { font-family: "JetBrains Mono", "Fira Code", monospace !important; font-size: 0.78rem !important; }
"""

_THEME = gr.themes.Base(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="gray",
    font=gr.themes.GoogleFont("Inter"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
).set(
    body_background_fill="#f0f2f5",
    block_background_fill="white",
    block_border_width="1px",
    block_border_color="#e5e7eb",
    block_radius="12px",
    input_background_fill="white",
    button_primary_background_fill="#4f46e5",
    button_primary_background_fill_hover="#4338ca",
    button_primary_text_color="white",
)


def _model_choices():
    m = get_available_models()
    return m if m else ["(train first — run quick_train.py)"]


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Image Upscaler · 4×") as demo:

        gr.Markdown("## Image Upscaler", elem_classes="app-header")
        gr.Markdown("4× super-resolution · Stable Diffusion · Custom Transformer", elem_classes="app-sub")

        with gr.Tabs():

            # ================================================================
            # Tab 1 — Stable Diffusion x4
            # ================================================================
            with gr.TabItem("Stable Diffusion x4"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=280):
                        sd_input = gr.Image(type="pil", label="Input", height=260)
                        sd_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="high resolution, sharp, detailed",
                            lines=2,
                        )
                        sd_steps = gr.Slider(10, 75, value=30, step=5,
                                             label="Diffusion steps")
                        sd_noise = gr.Slider(0, 100, value=0, step=5,
                                             label="Noise level  (0 = faithful · 100 = creative)")
                        sd_downscale = gr.Checkbox(
                            value=False,
                            label="Downscale ÷4 before upscaling",
                            info="Use only for benchmarking",
                        )
                        sd_btn = gr.Button("Upscale ×4", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        sd_out     = gr.Image(type="pil", label="Output · 4×", height=440)
                        sd_metrics = gr.Textbox(label="Info", lines=6, interactive=False,
                                                placeholder="Results will appear here…")

                sd_btn.click(
                    fn=run_sd_upscale,
                    inputs=[sd_input, sd_prompt, sd_steps, sd_noise, sd_downscale],
                    outputs=[sd_out, sd_metrics],
                )

            # ================================================================
            # Tab 2 — Custom Transformer model
            # ================================================================
            with gr.TabItem("Custom Model"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=280):
                        cm_input = gr.Image(type="pil", label="Input", height=260)
                        with gr.Row():
                            cm_model = gr.Dropdown(
                                choices=_model_choices(), value=None,
                                label="Model", scale=3, interactive=True,
                            )
                            cm_refresh = gr.Button("⟳", scale=1, min_width=44, size="sm")
                        with gr.Row():
                            cm_downscale = gr.Checkbox(value=True,  label="Downscale ÷4 first",
                                                       info="Enables PSNR / SSIM metrics")
                            cm_quantize  = gr.Checkbox(value=False, label="Int8 quantize",
                                                       info="Faster · slight quality trade-off")
                        cm_btn = gr.Button("Upscale ×4", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        with gr.Tabs():
                            with gr.TabItem("SR Output"):
                                cm_out = gr.Image(type="pil", label="Model · ×4", height=400)
                            with gr.TabItem("Bicubic"):
                                cm_bic = gr.Image(type="pil", label="Bicubic baseline · ×4", height=400)
                        cm_metrics = gr.Textbox(label="Metrics", lines=6, interactive=False,
                                                placeholder="Results will appear here…")

                cm_btn.click(
                    fn=run_custom_upscale,
                    inputs=[cm_input, cm_model, cm_downscale, cm_quantize],
                    outputs=[cm_out, cm_bic, cm_metrics],
                )
                cm_refresh.click(
                    fn=lambda: gr.update(choices=_model_choices()),
                    outputs=cm_model,
                )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(theme=_THEME, css=_CSS, show_error=True, inbrowser=True)
