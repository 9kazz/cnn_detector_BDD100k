import torch
from torchvision.ops import batched_nms, clip_boxes_to_image

from .anchors import calc_resulting_bboxes
from .arch import DetectionCNN
from .train import ANCHOR_SIZES


@torch.inference_mode()
def run_inference(
    model:                DetectionCNN,
    images:               torch.Tensor,
    confidence_threshold: float = 0.25,
    nms_iou_threshold:    float = 0.5,
    max_detections:       int | None = None,
) -> list[dict[str, torch.Tensor]]:
    """Run detection on float32 images [B, 3, H, W] in [0, 1].

    max_detections limits output per image to the highest scores after NMS.
    None keeps all surviving detections.
    """

    model.eval()
    device = next(model.parameters()).device
    images = images.to(device)
    
    predictions = model(images)
    predictions = calc_resulting_bboxes(
        predictions, ANCHOR_SIZES, model.backbone.output_stride
    )
    return postprocess_predictions(
        predictions, images.shape[-2:], confidence_threshold, nms_iou_threshold, max_detections
    )


def postprocess_predictions(
    predictions:          torch.Tensor,
    image_size:           tuple[int, int],
    confidence_threshold: float = 0.25,
    nms_iou_threshold:    float = 0.5,
    max_detections:       int | None = None,
) -> list[dict[str, torch.Tensor]]:
    """Filter decoded [B, H, W, A, P] predictions into per-image detections.

    Boxes are pixel xyxy coordinates. Scores are objectness times the
    highest class probability. Each anchor contributes at most one detection.
    image_size is (height, width). Clip boxes to these bounds and discard
    boxes with zero area before NMS.
    """
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be in [0, 1]")
    if not 0 <= nms_iou_threshold <= 1:
        raise ValueError("nms_iou_threshold must be in [0, 1]")
    if max_detections is not None and max_detections < 0:
        raise ValueError("max_detections must be non-negative or None")

    predictions = predictions.flatten(1, 3).float()
    objectness  = predictions[..., 4].sigmoid()
    class_probs = predictions[..., 5:].softmax(dim=-1)
    class_scores, class_ids = class_probs.max(dim=-1)
    scores = objectness * class_scores

    detections = []
    for image_predictions, image_scores, image_labels in zip(predictions, scores, class_ids):
        confident  = image_scores >= confidence_threshold
        pred_boxes = clip_boxes_to_image(image_predictions[confident, :4], image_size)
        selected_scores = image_scores[confident]
        labels          = image_labels[confident]

        valid = (pred_boxes[:, 2] > pred_boxes[:, 0]) & (pred_boxes[:, 3] > pred_boxes[:, 1])
        pred_boxes = pred_boxes[valid]
        selected_scores = selected_scores[valid]
        labels = labels[valid]

        keep = batched_nms(pred_boxes, selected_scores, labels, nms_iou_threshold)
        if max_detections is not None:
            keep = keep[:max_detections]
        detections.append({
            "boxes":  pred_boxes[keep],
            "scores": selected_scores[keep],
            "labels": labels[keep],
        })
    return detections
