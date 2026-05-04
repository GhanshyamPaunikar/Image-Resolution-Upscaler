import os
import re
from pathlib import Path

resolutions: dict[str, tuple[int, int]] = {
    "360":  (360,  640),
    "480":  (480,  854),
    "720":  (720,  1280),
    "1080": (1080, 1920),
    "1440": (1440, 2560),
    "4k":   (2160, 3840),
}


def get_latest_checkpoint(checkpoint_dir: str) -> tuple[str, int]:
    """Return (path, epoch) for the highest-epoch .pth file in checkpoint_dir."""
    d = Path(checkpoint_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    checkpoints = list(d.glob("*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No .pth checkpoints found in: {checkpoint_dir}")

    def _epoch(p: Path) -> int:
        m = re.search(r"epoch[_\-](\d+)", p.name)
        return int(m.group(1)) if m else 0

    latest = max(checkpoints, key=_epoch)
    return str(latest), _epoch(latest)


def get_available_models() -> list[str]:
    """Return names of model directories that contain a model.py file."""
    models_dir = Path("models")
    if not models_dir.is_dir():
        return []
    return sorted(
        d.name for d in models_dir.iterdir()
        if d.is_dir() and (d / "model.py").exists()
    )
