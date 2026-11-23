import torch
from torch.utils.data import DataLoader

from .anchors import calc_resulting_bboxes, match_anchors
from .arch import DetectionCNN
from .inference import postprocess_predictions
from .loss import calc_loss
from .train import ANCHOR_SIZES


@torch.inference_mode()
def run_benchmark(
    model:                DetectionCNN,
    loader:               DataLoader,
    device:               torch.device | str,
    confidence_threshold: float = 0.25,
    nms_iou_threshold:    float = 0.5,
    max_detections:       int | None = None,
) -> float:
    """Run detection and average the training loss over labeled test images."""

    model.to(device)
    model.eval()
    
    stride = model.backbone.output_stride
    anchor_sizes = torch.tensor(ANCHOR_SIZES, device=device, dtype=torch.float32)
    total_loss = 0.0
    n_images   = 0

    for images, targets in loader:
        images = images.to(device)
        predictions = calc_resulting_bboxes(model(images), anchor_sizes, stride)
        postprocess_predictions(
            predictions, images.shape[-2:], confidence_threshold, nms_iou_threshold, max_detections
        )
        pos_pred, neg_pred, gt_boxes, gt_labels = match_anchors(
            predictions, targets, anchor_sizes, stride
        )
        loss = calc_loss(pos_pred, neg_pred, gt_boxes, gt_labels)
        total_loss += loss.item() * len(images)
        n_images   += len(images)

    if n_images == 0:
        raise ValueError("Benchmark loader produced no images")
    test_loss = total_loss / n_images
    print(f"Test loss: {test_loss:.6f}", flush=True)
    return test_loss
