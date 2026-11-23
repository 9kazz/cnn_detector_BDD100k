from math import prod

import torch
import torch.nn as nn


class DetectionCNN(nn.Module):
    """Map a batch of images to [B, H, W, n_anchors, n_params]."""
    def __init__(self, n_anchors: int, n_classes: int):
        super().__init__()
        self.backbone = Backbone()
        self.head     = Head(self.backbone.out_channels, n_anchors, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.head(x)
        return x


class Backbone(nn.Module):
    '''Extract feature map from original image: (3, 720, 1280) --> (128, 45, 80).'''

    out_channels:  int = 128
    output_stride: int
    layers:        nn.Sequential

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),  

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, self.out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
        )
        self.output_stride = prod(
            layer.stride[0] 
            for layer in self.layers
            if isinstance(layer, nn.Conv2d)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Head(nn.Module):
    """Predict [H, W, n_anchors, n_params] via feature map."""

    in_channels: int
    n_anchors:   int
    n_params:    int
    predict_anchors: nn.Conv2d

    def __init__(self, 
                 in_channels: int,
                 n_anchors:   int, 
                 n_classes:   int):
        super().__init__()
        self.n_anchors   = n_anchors
        self.in_channels = in_channels
        self.n_params    = 5 + n_classes # [dx, dy, dw, dh, objectness, class1, ..., classN]
        self.predict_anchors = nn.Conv2d(self.in_channels, n_anchors * self.n_params, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred = self.predict_anchors(x)
        return pred.permute(0, 2, 3, 1).unflatten(-1, (self.n_anchors, self.n_params))
