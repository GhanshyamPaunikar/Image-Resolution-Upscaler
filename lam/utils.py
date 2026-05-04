"""Image-conversion helpers shared across the LAM pipeline."""
import cv2
import numpy as np
import torch
from PIL import Image


def PIL2Tensor(img: Image.Image) -> torch.Tensor:
    """PIL RGB image → float32 tensor [0, 1] of shape (C, H, W)."""
    arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def Tensor2PIL(t: torch.Tensor) -> Image.Image:
    """Float32 tensor (C, H, W) clamped to [0,1] → PIL RGB image."""
    arr = t.clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return Image.fromarray((arr * 255).astype(np.uint8))


def pil_to_cv2(img: Image.Image) -> np.ndarray:
    """PIL RGB → OpenCV BGR uint8 array."""
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def cv2_to_pil(arr: np.ndarray) -> Image.Image:
    """OpenCV BGR float/uint8 array → PIL RGB image."""
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def make_pil_grid(images: list[Image.Image], gap: int = 4) -> Image.Image:
    """Stitch a list of PIL images side-by-side into a single image."""
    if not images:
        raise ValueError("images list is empty")
    h = max(img.height for img in images)
    images = [img.resize((img.width, h)) if img.height != h else img for img in images]
    total_w = sum(img.width for img in images) + gap * (len(images) - 1)
    grid = Image.new("RGB", (total_w, h), (255, 255, 255))
    x = 0
    for img in images:
        grid.paste(img, (x, 0))
        x += img.width + gap
    return grid
