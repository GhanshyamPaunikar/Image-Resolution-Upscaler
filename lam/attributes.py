"""
Attribution functions for LAM (Local Attribution Map).

Reference: "LAM: Explain Your Super-resolution Networks with Local Attribution Maps"
           (https://arxiv.org/abs/2011.11036)
"""
import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Attribution objective
# ---------------------------------------------------------------------------

def attr_grad(tensor: torch.Tensor) -> torch.Tensor:
    """Sum of absolute gradients — used as the scalar attribution objective."""
    return tensor.abs().sum()


def attribution_objective(attr_func, h: int, w: int, window: int = 16):
    """
    Return a closure that, given a model output tensor, sums the absolute
    values inside the (h, w) window — the quantity whose gradient we trace.
    """
    def objective(output: torch.Tensor) -> torch.Tensor:
        # output shape: (1, C, H, W)
        patch = output[0, :, h: h + window, w: w + window]
        return attr_func(patch)
    return objective


# ---------------------------------------------------------------------------
# Saliency post-processing
# ---------------------------------------------------------------------------

def saliency_map_PG(
    interpolated_grad: np.ndarray,
    result: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the final saliency map from integrated path gradients.

    Returns:
        grad_map : (H, W) float array
        result   : unchanged result passed through for convenience
    """
    # Average over colour channels and interpolation steps
    grad = np.mean(np.abs(interpolated_grad), axis=(0, 1))  # (H, W)
    grad = grad / (grad.max() + 1e-8)
    return grad, result


def grad_abs_norm(grad: np.ndarray) -> np.ndarray:
    """Absolute value + min-max normalisation of a gradient map."""
    g = np.abs(grad)
    g = (g - g.min()) / (g.max() - g.min() + 1e-8)
    return g


def vis_saliency(saliency: np.ndarray, zoomin: int = 4) -> Image.Image:
    """
    Render a (H, W) saliency map as a heatmap PIL image, optionally upsampled.
    """
    h, w = saliency.shape[:2]
    saliency_uint8 = (saliency * 255).clip(0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(saliency_uint8, cv2.COLORMAP_JET)
    heat = cv2.resize(heat, (w * zoomin, h * zoomin), interpolation=cv2.INTER_NEAREST)
    return Image.fromarray(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB))


def vis_saliency_kde(saliency: np.ndarray, sigma: float = 3.0) -> Image.Image:
    """
    KDE-smoothed version of vis_saliency — applies a Gaussian filter before
    rendering the heatmap.
    """
    smoothed = gaussian_filter(saliency, sigma=sigma)
    smoothed = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min() + 1e-8)
    return vis_saliency(smoothed, zoomin=1)
