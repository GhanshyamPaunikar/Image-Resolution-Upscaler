#!/usr/bin/env python
"""
app.py — Gradio web interface for the Image Resolution Upscaler.

Run with:
    python app.py
Then open http://localhost:7860
"""
import logging
import os
import time

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import gradio as gr
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# MPS + float16 produces numerically incorrect outputs with this model;
# force CPU + float32 regardless of available device.
_SD_DEVICE = "cpu"
_SD_DTYPE  = torch.float32
_sd_pipe   = None


def _get_pipe():
    global _sd_pipe
    if _sd_pipe is None:
        from diffusers import StableDiffusionUpscalePipeline
        log.info("Loading SD x4 Upscaler…")
        _sd_pipe = StableDiffusionUpscalePipeline.from_pretrained(
            "stabilityai/stable-diffusion-x4-upscaler",
            torch_dtype=_SD_DTYPE,
        ).to(_SD_DEVICE)
        _sd_pipe.enable_attention_slicing()
        log.info("SD x4 Upscaler ready.")
    return _sd_pipe


def upscale(
    image: Image.Image | None,
    prompt: str,
    steps: int,
    noise: float,
) -> tuple[Image.Image | None, str]:
    if image is None:
        return None, "Upload an image to get started."

    orig = image.convert("RGB")

    # Clamp input so output stays ≤ 512×512 (model hard limit)
    w, h = orig.size
    max_side = 128
    if max(w, h) > max_side:
        ratio = max_side / max(w, h)
        orig = orig.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

    try:
        pipe = _get_pipe()
    except Exception as e:
        return None, f"Failed to load pipeline: {e}"

    t0 = time.perf_counter()
    result = pipe(
        prompt=prompt.strip() or "high resolution, sharp, detailed",
        image=orig,
        num_inference_steps=steps,
        noise_level=int(noise),
    ).images[0]
    elapsed = time.perf_counter() - t0

    info = "\n".join([
        f"Model     stabilityai/stable-diffusion-x4-upscaler",
        f"Input     {orig.size[0]} × {orig.size[1]}   →   Output  {result.size[0]} × {result.size[1]}",
        f"Steps     {steps}   |   Noise  {int(noise)}",
        f"Time      {elapsed:.1f} s",
    ])
    return result, info


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
_CSS = """
body, .gradio-container { background: #f0f2f5 !important; }
footer { display: none !important; }

.app-header { text-align: center; padding: 2.5rem 0 0.25rem; }
.app-header h1 {
  font-size: 1.75rem; font-weight: 700;
  letter-spacing: -0.03em; color: #111827; margin: 0;
}
.app-sub {
  text-align: center; color: #6b7280;
  font-size: 0.82rem; margin: 0.3rem 0 2rem;
}

.block { border-radius: 10px !important; border: 1px solid #e5e7eb !important; }
label { font-weight: 500 !important; font-size: 0.8rem !important; color: #374151 !important; }

button.primary {
  background: #4f46e5 !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
  font-size: 1rem !important;
  letter-spacing: 0.01em !important;
  box-shadow: 0 1px 3px rgba(79,70,229,.35) !important;
}
button.primary:hover { background: #4338ca !important; }

textarea {
  font-family: "JetBrains Mono", "Fira Code", monospace !important;
  font-size: 0.78rem !important;
}
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
    block_radius="10px",
    input_background_fill="white",
    button_primary_background_fill="#4f46e5",
    button_primary_background_fill_hover="#4338ca",
    button_primary_text_color="white",
)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Image Upscaler · 4×") as demo:

        gr.Markdown("# Image Upscaler", elem_classes="app-header")
        gr.Markdown(
            "Stable Diffusion x4 · 4× super-resolution",
            elem_classes="app-sub",
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=300):
                img_in  = gr.Image(type="pil", label="Input image", height=280)
                prompt  = gr.Textbox(
                    label="Prompt (optional)",
                    placeholder="high resolution, sharp, detailed",
                    lines=2,
                )
                steps   = gr.Slider(10, 75, value=30, step=5, label="Diffusion steps")
                noise   = gr.Slider(0, 100, value=0, step=5,
                                    label="Noise level  ·  0 = faithful   100 = creative")
                btn     = gr.Button("Upscale ×4", variant="primary", size="lg")

            with gr.Column(scale=2):
                img_out = gr.Image(type="pil", label="Output · 4×", height=480)
                info    = gr.Textbox(label="Info", lines=5, interactive=False,
                                     placeholder="Results will appear here…")

        btn.click(
            fn=upscale,
            inputs=[img_in, prompt, steps, noise],
            outputs=[img_out, info],
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(theme=_THEME, css=_CSS, show_error=True, inbrowser=True)
