"""
quick_train.py — minimal training script tuned for speed.

Trains Fastv2 on images/training_set for a fixed number of steps
using scale-factor 4 only (48×48 LR → 192×192 HR).
Run: python quick_train.py
"""
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image

# ── Dataset ────────────────────────────────────────────────────────────────

class PairDataset(torch.utils.data.Dataset):
    def __init__(self, folder: str, hr_size: int = 192, scale: int = 4):
        exts = (".jpg", ".jpeg", ".png")
        self.files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in exts)
        lr_size = hr_size // scale
        self.hr_tf = transforms.Compose([
            transforms.Resize((hr_size, hr_size), transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])
        self.lr_tf = transforms.Compose([
            transforms.Resize((lr_size, lr_size), transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        return self.lr_tf(img), self.hr_tf(img)


# ── Training ────────────────────────────────────────────────────────────────

def main():
    SCALE      = 4
    EPOCHS     = 100
    BATCH      = 8
    LR         = 2e-4
    DATA_DIR   = "images/training_set"
    CKPT_DIR   = "models/Fastv2/checkpoints"
    LOG_EVERY  = 10   # print every N epochs

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    from models.Fastv2.model import TransformerModel
    model = TransformerModel().to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n:,}")

    dataset   = PairDataset(DATA_DIR, hr_size=192, scale=SCALE)
    loader    = DataLoader(dataset, batch_size=BATCH, shuffle=True, num_workers=0)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.L1Loss()

    Path(CKPT_DIR).mkdir(parents=True, exist_ok=True)

    use_amp = device.type in ("cuda", "mps")

    print(f"\nTraining {EPOCHS} epochs on {len(dataset)} images (scale ×{SCALE})…\n")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for lr_b, hr_b in loader:
            lr_b, hr_b = lr_b.to(device), hr_b.to(device)
            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                    out = model(lr_b, upscale_factor=SCALE)
                    loss = criterion(out, hr_b)
            else:
                out = model(lr_b, upscale_factor=SCALE)
                loss = criterion(out, hr_b)
            loss.backward()
            optimizer.step()
            running += loss.item()
        scheduler.step()

        if epoch % LOG_EVERY == 0 or epoch == 1:
            avg = running / len(loader)
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:4d}/{EPOCHS}  loss={avg:.5f}  lr={lr_now:.2e}")

    ckpt_path = Path(CKPT_DIR) / f"model_epoch_{EPOCHS}.pth"
    torch.save(
        {"epoch": EPOCHS, "model_state_dict": model.state_dict()},
        ckpt_path,
    )
    print(f"\nCheckpoint saved: {ckpt_path}")


if __name__ == "__main__":
    main()
