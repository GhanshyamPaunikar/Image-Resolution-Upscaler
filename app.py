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

from pathlib import Path

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

    # Cap input so output doesn't blow up memory (max 512×512 output = 128×128 input)
    max_lr_side = 128
    if downscale_first:
        ow, oh = orig.size
        lr = orig.resize((ow // 4, oh // 4), Image.LANCZOS)
    else:
        lr = orig

    # Clamp LR so output stays ≤ 512×512
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

    lines = [
        f"Model:       Stable Diffusion x4 Upscaler",
        f"Device:      {_SD_DEVICE.upper()}  (float32)",
        f"Steps:       {steps}  |  Noise level: {int(noise)}",
        f"Prompt:      \"{prompt or 'high resolution, sharp, detailed'}\"",
        f"Input:       {lr.size[0]} × {lr.size[1]}",
        f"Output:      {result.size[0]} × {result.size[1]}",
        f"Time:        {elapsed:.1f} s",
    ]
    return result, "\n".join(lines)


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
    scale: int,
    downscale_first: bool,
    quantize: bool,
) -> tuple[Image.Image | None, Image.Image | None, str]:
    if image is None:
        return None, None, "Upload an image to get started."

    available = get_available_models()
    if not available:
        return None, None, (
            "No trained models in models/.\n"
            "Run:  python quick_train.py"
        )
    if model_name not in available:
        return None, None, f"Model '{model_name}' not found. Available: {', '.join(available)}"

    try:
        model = _load_custom_model(model_name, quantize)
    except Exception as e:
        return None, None, f"Load error: {e}"

    to_tensor = transforms.ToTensor()
    to_pil    = transforms.ToPILImage()
    orig = image.convert("RGB")

    lr = orig.resize((orig.size[0] // scale, orig.size[1] // scale), Image.BICUBIC) \
         if downscale_first else orig

    lr_t = to_tensor(lr).unsqueeze(0).to(DEVICE)
    t0 = time.perf_counter()
    with torch.no_grad():
        if DEVICE.type in ("cuda", "mps"):
            with torch.amp.autocast(device_type=DEVICE.type, dtype=torch.float16):
                out = model(lr_t, upscale_factor=scale)
        else:
            out = model(lr_t, upscale_factor=scale)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    sr      = to_pil(out.squeeze(0).cpu().clamp(0, 1))
    bicubic = lr.resize((lr.size[0] * scale, lr.size[1] * scale), Image.BICUBIC)

    lines = [
        f"Device:      {DEVICE.type.upper()}",
        f"Inference:   {elapsed_ms:.1f} ms",
        f"Input:       {lr.size[0]} × {lr.size[1]}",
        f"Output:      {sr.size[0]} × {sr.size[1]}",
        f"Model:       {model_name}  (scale ×{scale})",
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
            f"Model   — PSNR: {sr_psnr:.2f} dB  |  SSIM: {sr_ssim:.4f}",
            f"Bicubic — PSNR: {bic_psnr:.2f} dB  |  SSIM: {bic_ssim:.4f}",
            f"Gain    — PSNR: +{sr_psnr-bic_psnr:.2f} dB  |  SSIM: +{sr_ssim-bic_ssim:.4f}",
        ]

    return sr, bicubic, "\n".join(lines)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
_CSS = """
.app-title  { text-align:center; padding: 1rem 0 0.1rem; }
.app-sub    { text-align:center; color:#64748b; font-size:0.9rem; margin-bottom:1.2rem; }
footer      { display:none !important; }
"""

_THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
)


def _model_choices():
    m = get_available_models()
    return m if m else ["(train first — run quick_train.py)"]


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Image Resolution Upscaler") as demo:

        gr.Markdown("## Image Resolution Upscaler", elem_classes="app-title")
        gr.Markdown(
            "Stable Diffusion x4 · Custom Transformer · CUDA / MPS / CPU",
            elem_classes="app-sub",
        )

        with gr.Tabs():

            # ================================================================
            # Tab 1 — Stable Diffusion x4
            # ================================================================
            with gr.TabItem("Stable Diffusion x4"):
                gr.Markdown(
                    "Uses **stabilityai/stable-diffusion-x4-upscaler** — "
                    "diffusion-based, produces photorealistic detail. "
                    "First run loads the model (~7 s), subsequent runs are faster."
                )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        sd_input  = gr.Image(type="pil", label="Input Image", height=280)
                        sd_prompt = gr.Textbox(
                            label="Prompt (optional)",
                            placeholder="high resolution, sharp, detailed",
                            lines=2,
                        )
                        sd_steps  = gr.Slider(10, 75, value=30, step=5, label="Diffusion Steps  (more = better quality, slower)")
                        sd_noise  = gr.Slider(0, 100, value=0, step=5,  label="Noise Level  (0 = faithful to input, higher = more creative)")
                        sd_downscale = gr.Checkbox(value=False, label="Downscale input ÷4 first (only for benchmarking — upload a blurry image instead)")
                        sd_btn    = gr.Button("Upscale with SD x4 ↑", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        sd_out     = gr.Image(type="pil", label="SD x4 Output", height=420)
                        sd_metrics = gr.Textbox(label="Info", lines=8, interactive=False,
                                                placeholder="Results will appear here…")

                sd_btn.click(
                    fn=run_sd_upscale,
                    inputs=[sd_input, sd_prompt, sd_steps, sd_noise, sd_downscale],
                    outputs=[sd_out, sd_metrics],
                )

            # ================================================================
            # Tab 2 — Custom Transformer model
            # ================================================================
            with gr.TabItem("Custom Model (Fastv2)"):
                gr.Markdown(
                    "Uses your locally trained **Fastv2** (or any other model in `models/`). "
                    "Fast inference, all scale factors supported."
                )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        cm_input = gr.Image(type="pil", label="Input Image", height=260)
                        with gr.Row():
                            cm_model = gr.Dropdown(
                                choices=_model_choices(), value=None,
                                label="Model", scale=3, interactive=True,
                            )
                            cm_refresh = gr.Button("⟳", scale=1, min_width=48, size="sm")
                        cm_scale     = gr.Radio([2, 3, 4, 6], value=4, label="Scale Factor")
                        with gr.Row():
                            cm_downscale = gr.Checkbox(value=True,  label="Downscale input first", info="Enables PSNR/SSIM")
                            cm_quantize  = gr.Checkbox(value=False, label="Quantize (int8)",       info="Faster, slightly lower quality")
                        cm_btn = gr.Button("Upscale ↑", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        with gr.Tabs():
                            with gr.TabItem("SR Output"):
                                cm_out  = gr.Image(type="pil", label="Model Output", height=380)
                            with gr.TabItem("Bicubic Baseline"):
                                cm_bic  = gr.Image(type="pil", label="Bicubic", height=380)
                        cm_metrics = gr.Textbox(label="Metrics & Info", lines=8, interactive=False,
                                                placeholder="Results will appear here…")

                cm_btn.click(
                    fn=run_custom_upscale,
                    inputs=[cm_input, cm_model, cm_scale, cm_downscale, cm_quantize],
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
