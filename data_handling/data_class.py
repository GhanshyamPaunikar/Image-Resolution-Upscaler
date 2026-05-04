import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

_DEFAULT_SCALE_PAIR = {"lr": (48, 48), "hr": (192, 192), "factor": 4}


class highres_img_dataset(Dataset):
    """
    Loads images from a local directory and produces (lr_tensor, hr_tensor) pairs.

    scale_pair format:
        {"lr": (H, W), "hr": (H, W), "factor": int}
    """

    def __init__(self, data_dir: str, scale_pair: Optional[dict] = None):
        self.data_dir = Path(data_dir)
        sp = scale_pair or _DEFAULT_SCALE_PAIR

        self.image_files = sorted(
            p for p in self.data_dir.iterdir()
            if p.suffix.lower() in _IMG_EXTS
        )
        if not self.image_files:
            raise ValueError(f"No images found in {data_dir}")

        self._lr_tf = transforms.Compose([
            transforms.Resize(sp["lr"], interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])
        self._hr_tf = transforms.Compose([
            transforms.Resize(sp["hr"], interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(self.image_files[idx]).convert("RGB")
        return self._lr_tf(img), self._hr_tf(img)


class highres_img_dataset_online(Dataset):
    """Placeholder — streams images from an online source (not yet implemented)."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Online dataset is not implemented. "
            "Use highres_img_dataset with a local --data_dir instead."
        )
