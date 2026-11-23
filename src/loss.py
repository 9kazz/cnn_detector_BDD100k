import torch
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou_loss


def calc_loss(
    pos_pred:          torch.Tensor,
    neg_pred:          torch.Tensor,
    matched_gt_boxes:  torch.Tensor,
    matched_gt_labels: torch.Tensor,
    *,
    coef_box: float = 1.0,
    coef_cls: float = 1.0,
    coef_obj: float = 1.0,
    coef_neg: float = 1.0,
) -> torch.Tensor:
    """Combine box, classification, and objectness loss for positive anchors
    with objectness loss for negative anchors. Average each group separately
    and weight negative objectness by `coef_neg`.
    """
    loss_box, loss_cls, loss_obj_pos = calc_positive_loss(
        pos_pred, matched_gt_boxes, matched_gt_labels
    )
    loss_obj_neg = calc_negative_loss(neg_pred)
    
    total_obj_loss = loss_obj_pos + (coef_neg * loss_obj_neg)
    return (coef_box * loss_box) + (coef_cls * loss_cls) + (coef_obj * total_obj_loss)

def calc_positive_loss(
    pos_predictions:   torch.Tensor,
    matched_gt_boxes:  torch.Tensor,
    matched_gt_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the box, classification, and objectness losses for positive anchors."""

    loss_box =            calc_box_loss(pos_predictions[:, :4], matched_gt_boxes)
    loss_cls = calc_classification_loss(pos_predictions[:, 5:], matched_gt_labels)
    loss_obj =     calc_objectness_loss(pos_predictions[:, 4],  target_value=1)
    return loss_box, loss_cls, loss_obj


def calc_negative_loss(neg_predictions: torch.Tensor) -> torch.Tensor:
    """Return the mean objectness loss for negative anchors."""
    return calc_objectness_loss(neg_predictions[:, 4], target_value=0)


def calc_objectness_loss(pred_logits:  torch.Tensor, 
                    target_value: int
) -> torch.Tensor:
    if pred_logits.numel() == 0:
        return pred_logits.sum()
    
    targets = torch.full_like(pred_logits, target_value)
    return F.binary_cross_entropy_with_logits(pred_logits, targets, reduction="mean")


def calc_classification_loss(pred_logits: torch.Tensor, 
                        gt_labels:   torch.Tensor
) -> torch.Tensor:
    if pred_logits.size(0) == 0:
        return pred_logits.sum()
    return F.cross_entropy(pred_logits, gt_labels, reduction="mean")


def calc_box_loss(pred_boxes: torch.Tensor, 
             gt_boxes:   torch.Tensor
) -> torch.Tensor:
    if pred_boxes.numel() == 0:
        return pred_boxes.sum()
    return generalized_box_iou_loss(pred_boxes, gt_boxes, reduction="mean")
