from pathlib import Path

import torch
import torch.nn as nn

from .arch import DetectionCNN
from .load_data import CLASS_NAMES
from .train import ANCHOR_SIZES


def store_model(model: nn.Module, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(path: str | Path, device: torch.device | str) -> DetectionCNN:
    model = DetectionCNN(n_anchors=len(ANCHOR_SIZES), n_classes=len(CLASS_NAMES))
    init_model(model, path, device)
    model.eval()

    return model


def init_model(model:  nn.Module, 
               path:   str | Path,
               device: torch.device | str
) -> nn.Module:

    state = torch.load(
        path,
        map_location=device,
        weights_only=True
    )

    model.to(device)
    model.load_state_dict(state)

    return model
