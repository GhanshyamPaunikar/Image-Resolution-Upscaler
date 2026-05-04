#!/usr/bin/env python
"""
app.py — Gradio web interface for the Image Resolution Upscaler.

Run with:
    python app.py

Then open http://localhost:7860 in your browser.
"""
import importlib
import logging
import time
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

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)
log.info("Using device: %s", DEVICE)

# ---------------------------------------------------------------------------
# Model cache — avoid reloading on every click
# ---------------------------------------------------------------------------
_cache: dict[str, nn.Module] = {}


def _load_model(model_name: str, quantize: bool) -> nn.Module:
    key = f"{model_name}|{quantize}"
    if key not in _cache:
        module = importlib.import_module(f"models.{model_name}.model")
        model = module.TransformerModel().to(DEVICE)

        ckpt_path, epoch = get_latest_checkpoint(f"models/{model_name}/checkpoints")
        log.info("Loading %s — epoch %d from %s", model_name, epoch, ckpt_path)

        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state)
        model.eval()

        if quantize:
            model = torch.ao.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )

        _cache[key] = model
    return _cache[key]


# ---------------------------------------------------------------------------
# Core inference function
# ---------------------------------------------------------------------------

def run_upscale(
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
        msg = (
            "No trained models found in models/.\n"
            "Train a model first:\n"
            "  python train.py --data_dir /path/to/images --model Fastv2"
        )
        return None, None, msg

    if model_name not in available:
        return None, None, f"Model '{model_name}' not found. Available: {', '.join(available)}"

    try:
        model = _load_model(model_name, quantize)
    except FileNotFoundError as e:
        return None, None, f"Checkpoint error: {e}"
    except Exception as e:
        return None, None, f"Failed to load model: {e}"

    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()
    orig = image.convert("RGB")

    if downscale_first:
        ow, oh = orig.size
        lr = orig.resize((ow // scale, oh // scale), Image.BICUBIC)
    else:
        lr = orig

    lr_t = to_tensor(lr).unsqueeze(0).to(DEVICE)

    t0 = time.perf_counter()
    with torch.no_grad():
        if DEVICE.type in ("cuda", "mps"):
            with torch.amp.autocast(device_type=DEVICE.type, dtype=torch.float16):
                out = model(lr_t, upscale_factor=scale)
        else:
            out = model(lr_t, upscale_factor=scale)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    sr = to_pil(out.squeeze(0).cpu().clamp(0, 1))
    bicubic = lr.resize((lr.size[0] * scale, lr.size[1] * scale), Image.BICUBIC)

    lines = [
        f"Device:      {DEVICE.type.upper()}",
        f"Inference:   {elapsed_ms:.1f} ms",
        f"Input size:  {lr.size[0]} × {lr.size[1]}",
        f"Output size: {sr.size[0]} × {sr.size[1]}",
        f"Model:       {model_name}  (scale ×{scale})",
    ]

    if downscale_first:
        ref = orig.resize(sr.size, Image.BICUBIC) if orig.size != sr.size else orig
        ref_arr = np.array(ref).astype(np.float32) / 255.0
        sr_arr = np.array(sr).astype(np.float32) / 255.0
        bic_arr = np.array(bicubic.resize(ref.size, Image.BICUBIC)).astype(np.float32) / 255.0

        sr_psnr = compare_psnr(ref_arr, sr_arr, data_range=1.0)
        sr_ssim = compare_ssim(ref_arr, sr_arr, data_range=1.0, channel_axis=-1)
        bic_psnr = compare_psnr(ref_arr, bic_arr, data_range=1.0)
        bic_ssim = compare_ssim(ref_arr, bic_arr, data_range=1.0, channel_axis=-1)

        lines += [
            "",
            f"Model   — PSNR: {sr_psnr:.2f} dB  |  SSIM: {sr_ssim:.4f}",
            f"Bicubic — PSNR: {bic_psnr:.2f} dB  |  SSIM: {bic_ssim:.4f}",
            f"Gain    — PSNR: +{sr_psnr - bic_psnr:.2f} dB  |  SSIM: +{sr_ssim - bic_ssim:.4f}",
        ]

    return sr, bicubic, "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

_CSS = """
.app-header { text-align: center; padding: 1rem 0 0.25rem; }
.app-sub    { text-align: center; color: #64748b; font-size: 0.95rem; margin-bottom: 1.5rem; }
.run-btn    { font-size: 1.1rem !important; }
footer      { display: none !important; }
"""

_THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
)


def _model_choices() -> list[str]:
    m = get_available_models()
    return m if m else ["(no models found — train first)"]


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Image Resolution Upscaler") as demo:

        gr.Markdown("## Image Resolution Upscaler", elem_classes="app-header")
        gr.Markdown(
            "Deep-learning super-resolution · 2× · 3× · 4× · 6× · "
            "Transformer-based · CUDA / MPS / CPU",
            elem_classes="app-sub",
        )

        with gr.Row(equal_height=False):
            # ---- Left column: controls ----
            with gr.Column(scale=1, min_width=300):
                input_img = gr.Image(
                    type="pil",
                    label="Input Image",
                    height=260,
                )

                with gr.Group():
                    with gr.Row():
                        model_dd = gr.Dropdown(
                            choices=_model_choices(),
                            value=None,
                            label="Model",
                            scale=3,
                            interactive=True,
                        )
                        refresh_btn = gr.Button("⟳", scale=1, min_width=48, size="sm")

                    scale_radio = gr.Radio(
                        choices=[2, 3, 4, 6],
                        value=4,
                        label="Scale Factor",
                    )

                    with gr.Row():
                        downscale_chk = gr.Checkbox(
                            value=True,
                            label="Downscale input first",
                            info="Enables PSNR/SSIM comparison",
                        )
                        quantize_chk = gr.Checkbox(
                            value=False,
                            label="Quantize (int8)",
                            info="Smaller model, slightly lower quality",
                        )

                run_btn = gr.Button(
                    "Upscale ↑",
                    variant="primary",
                    size="lg",
                    elem_classes="run-btn",
                )

            # ---- Right column: results ----
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("Super-Resolution Output"):
                        sr_img = gr.Image(
                            type="pil",
                            label="Model Output",
                            height=380,
                        )
                    with gr.TabItem("Bicubic Baseline"):
                        bic_img = gr.Image(
                            type="pil",
                            label="Bicubic (for comparison)",
                            height=380,
                        )

                metrics_box = gr.Textbox(
                    label="Metrics & Info",
                    lines=7,
                    interactive=False,
                    placeholder="Run upscaling to see results here…",
                )

        # ---- Event wiring ----
        run_btn.click(
            fn=run_upscale,
            inputs=[input_img, model_dd, scale_radio, downscale_chk, quantize_chk],
            outputs=[sr_img, bic_img, metrics_box],
        )

        refresh_btn.click(
            fn=lambda: gr.update(choices=_model_choices()),
            outputs=model_dd,
        )

        # ---- Examples (shown only when images/ exists) ----
        examples_dir = Path("images")
        example_imgs = sorted(examples_dir.glob("*.jpg"))[:4] if examples_dir.is_dir() else []
        if example_imgs:
            gr.Examples(
                examples=[[str(p), "Fastv2", 4, True, False] for p in example_imgs],
                inputs=[input_img, model_dd, scale_radio, downscale_chk, quantize_chk],
                label="Example Images",
            )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(theme=_THEME, css=_CSS, show_error=True, inbrowser=True)
