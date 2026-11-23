from collections.abc import Sequence

import torch
from torchvision.ops import box_iou


def match_anchors(
    pred_targets:      torch.Tensor,
    gt_targets:        list[dict[str, torch.Tensor]],
    init_anchor_sizes: torch.Tensor,
    stride:            int,
    pos_iou_threshold: float = 0.5,
    neg_iou_threshold: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select decoded predictions [B, H, W, A, P] by matching original anchors.

    Anchor sizes are (width, height). GT boxes and prediction boxes use pixel
    xyxy coordinates.

    Returns (pos_predictions, neg_predictions, matched_gt_boxes, matched_gt_labels)
    with shapes [N_pos, P], [N_neg, P], [N_pos, 4], and [N_pos], respectively.
    N_pos and N_neg count selected anchors across the batch. Each GT box and
    label corresponds to the positive prediction at the same index.
    """
    if not 0 <= neg_iou_threshold <= pos_iou_threshold <= 1:
        raise ValueError("Expected 0 <= negative_iou_threshold <= positive_iou_threshold <= 1")

    device = pred_targets.device
    batch_size, height, width, _, _ = pred_targets.shape
    predictions = pred_targets.flatten(1, 3)

    if batch_size == 0:
        raise ValueError("Batch cannot be empty")

    init_anchor_sizes = init_anchor_sizes.to(device=device, dtype=torch.float32)
    centers      = anchor_centers(height, width, stride, init_anchor_sizes)
    half_sizes   = init_anchor_sizes / 2
    anchors      = torch.cat((centers - half_sizes, centers + half_sizes), dim=-1).reshape(-1, 4)
    n_anchors    = anchors.size(0)

    pos_predictions   = []
    neg_predictions   = []
    matched_gt_boxes  = []
    matched_gt_labels = []

    for image_pred, gt_target in zip(predictions, gt_targets, strict=True):
        gt_boxes  = gt_target["boxes"].to(device=device,  dtype=torch.float32)
        gt_labels = gt_target["labels"].to(device=device, dtype=torch.int64)
        n_gt = gt_boxes.size(0)
        
        if n_gt > n_anchors:
            raise ValueError(f"Cannot match {n_gt} GT objects to {n_anchors} anchors")
        if n_gt == 0:
            pos_predictions.append(image_pred[:0])
            neg_predictions.append(image_pred)
            matched_gt_boxes.append(gt_boxes)
            matched_gt_labels.append(gt_labels)
            continue

        ious = box_iou(anchors, gt_boxes)
        best_ious, matched_gt = ious.max(dim=1)
        positive = best_ious >= pos_iou_threshold
        negative = best_ious < neg_iou_threshold

        # Keep threshold decisions for other anchors, and force one match per GT.
        for _ in range(n_gt):
            pair_index   = ious.argmax()
            anchor_index = pair_index // n_gt
            gt_index = pair_index % n_gt
            positive[anchor_index]   = True
            negative[anchor_index]   = False
            matched_gt[anchor_index] = gt_index
            ious[anchor_index, :] = -1
            ious[:, gt_index]     = -1

        pos_predictions.append(image_pred[positive])
        neg_predictions.append(image_pred[negative])
        matched_gt_boxes.append(gt_boxes[matched_gt[positive]])
        matched_gt_labels.append(gt_labels[matched_gt[positive]])

    return (
        torch.cat(pos_predictions),
        torch.cat(neg_predictions),
        torch.cat(matched_gt_boxes),
        torch.cat(matched_gt_labels),
    )


def anchor_centers(
    height:    int, 
    width:     int, 
    stride:    int, 
    reference: torch.Tensor
) -> torch.Tensor:
    """Return cell centers in pixel (x, y) order with shape [H, W, 1, 2]."""

    rows    = torch.arange(height, device=reference.device, dtype=reference.dtype)
    columns = torch.arange(width,  device=reference.device, dtype=reference.dtype)
    grid_y, grid_x = torch.meshgrid(rows, columns, indexing="ij")
    grid_xy = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(-2)
    centers = (grid_xy  + 0.5) * stride
    return centers


def calc_resulting_bboxes(
    pred_targets:       torch.Tensor,
    init_anchor_shapes: torch.Tensor | Sequence[Sequence[float]],
    stride: int,
) -> torch.Tensor:
    """Decode [B, H, W, A, P] predictions into pixel-space xyxy and scores.

    Initial anchor shapes are (width, height) in pixels. Each anchor is centered
    in its grid cell. The first four channels are (dx, dy, dw, dh).
    Center offsets are relative to the anchor size. Size offsets are log ratios.
    """
    _, height, width, n_anchors, _ = pred_targets.shape
    anchor_shapes = torch.as_tensor(
        init_anchor_shapes, device=pred_targets.device, dtype=pred_targets.dtype
    ).reshape(n_anchors, 2)

    # Share each cell center across all anchors in that cell.
    centers      = anchor_centers(height, width, stride, pred_targets)
    centers      = centers + pred_targets[..., :2] * anchor_shapes
    half_sizes   = pred_targets[..., 2:4].exp() * anchor_shapes / 2
    top_left     = centers - half_sizes
    bottom_right = centers + half_sizes
    return torch.cat((top_left, bottom_right, pred_targets[..., 4:]), dim=-1)
