"""
Gaussian-blur integrated-gradient path for LAM attribution.

The key idea: interpolate between a maximally-blurred (baseline) image and the
original via a series of Gaussian blur levels, then integrate gradients of the
attribution objective along that path.
"""
import numpy as np
import torch
from scipy.ndimage import gaussian_filter


class GaussianBlurPath:
    """
    Defines an interpolation path from a heavily blurred image to the original
    using decreasing Gaussian sigma values.

    Args:
        sigma   : peak blur sigma (applied at step 0)
        fold    : number of interpolation steps
        l       : exponent controlling how quickly sigma decays
    """

    def __init__(self, sigma: float = 1.2, fold: int = 50, l: int = 9):
        self.sigma = sigma
        self.fold = fold
        self.l = l

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            image: (C, H, W) float32 array in [0, 1]

        Returns:
            interpolated : (fold, C, H, W) — the path of blurred images
            sigmas       : (fold,) array of sigma values used
        """
        c, h, w = image.shape
        sigmas = np.array(
            [self.sigma * (1 - k / self.fold) ** self.l for k in range(self.fold)]
        )
        interpolated = np.stack(
            [
                np.stack(
                    [gaussian_filter(image[ch], sigma=s) for ch in range(c)],
                    axis=0,
                )
                for s in sigmas
            ],
            axis=0,
        ).astype(np.float32)  # (fold, C, H, W)
        return interpolated, sigmas


def Path_gradient(
    image: np.ndarray,
    model: torch.nn.Module,
    attr_objective,
    blur_path_func: GaussianBlurPath,
    upscale_factor: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute integrated gradients along the Gaussian-blur path.

    Args:
        image           : (C, H, W) float32 numpy array of the LR image
        model           : super-resolution model (already on the right device)
        attr_objective  : callable(output_tensor) → scalar tensor
        blur_path_func  : GaussianBlurPath instance
        upscale_factor  : scale factor passed to model.forward

    Returns:
        interpolated_grads : (fold, C, H, W) gradient array
        result_numpy       : (C, H, W) model output for the original image
        interpolated       : (fold, C, H, W) the blurred path images
    """
    device = next(model.parameters()).device
    interpolated, _ = blur_path_func(image)  # (fold, C, H, W)

    grads = []
    for step_img in interpolated:
        inp = torch.from_numpy(step_img).unsqueeze(0).to(device)  # (1, C, H, W)
        inp.requires_grad_(True)

        with torch.enable_grad():
            output = model(inp, upscale_factor=upscale_factor)
            loss = attr_objective(output)
            loss.backward()

        grads.append(inp.grad.detach().cpu().numpy().squeeze(0))  # (C, H, W)

    interpolated_grads = np.stack(grads, axis=0)  # (fold, C, H, W)

    # Run model once without grad to get the actual output
    with torch.no_grad():
        orig_inp = torch.from_numpy(image).unsqueeze(0).to(device)
        result = model(orig_inp, upscale_factor=upscale_factor)
    result_numpy = result.squeeze(0).cpu().numpy()  # (C, H, W)

    return interpolated_grads, result_numpy, interpolated
