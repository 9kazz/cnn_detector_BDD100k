import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


CLASS_NAMES = (
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motor",
    "bike",
    "traffic light",
    "traffic sign",
)
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}


class DatasetBDD100K(Dataset):
    """Load RGB images and bounding boxes xyxy from a BDD100K."""

    image_dir: str | Path
    frames:    list[dict[str, Any]]

    def __init__(
        self,
        image_dir:       str | Path,
        annotation_path: str | Path,
    ):
        self.image_dir = Path(image_dir)

        with Path(annotation_path).open(encoding="utf-8") as annotation_file:
            self.frames = json.load(annotation_file)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index: int):
        frame = self.frames[index]
        with Image.open(self.image_dir / frame["name"]) as source:
            image = source.convert("RGB")

        image_tensor = to_tensor(image)

        boxes  = []
        labels = []
        for annotation in frame.get("labels"):
            box = annotation.get("box2d")
            if box is None:
                continue
            boxes.append([box["x1"], box["y1"], box["x2"], box["y2"]])
            labels.append(CLASS_TO_ID[annotation["category"]])

        target = {
            "boxes":  torch.tensor(boxes,  dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        return image_tensor, target


def collate_fn(batch):
    """Stack images of the same size and keep targets as a list."""
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)
