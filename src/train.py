import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

from .anchors import match_anchors, calc_resulting_bboxes
from .loss import calc_loss


ANCHOR_SIZES = (
    (24, 40), 
    (64, 36), 
    (160, 96)
)


def run_training(model:     nn.Module,
                 train_ldr: DataLoader,
                 val_ldr:   DataLoader,
                 optimizer: torch.optim.Optimizer,
                 n_epoch:   int,
                 device:    torch.device | str,
                 log_dir:   str | Path = "train_log.csv",
) -> nn.Module:
    """Train the model and log mean batch losses by epoch."""

    device = torch.device(device)
    output_stride = model.backbone.output_stride

    init_anchor_sizes = torch.tensor(ANCHOR_SIZES, device=device, dtype=torch.float32)

    log = Path(log_dir).open("a", encoding="utf-8", newline="")
    with log as log_file:
        writer = csv.writer(log_file)
        if log_file.tell() == 0:
            writer.writerow(("epoch", "train_loss", "val_loss"))

        for epoch in tqdm(range(1, n_epoch + 1)):
            model.train()
            train_loss_sum = 0.0
            n_images = 0
            for images, ref_targets in train_ldr:
                optimizer.zero_grad()
                loss = calc_batch_loss(model, images, ref_targets, init_anchor_sizes, output_stride, device)
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item() * len(images)
                n_images += len(images)

            if n_images == 0:
                raise ValueError("Training loader produced no images")
            train_loss = train_loss_sum / n_images
            val_loss   = calc_val_loss(model, val_ldr, init_anchor_sizes, device)
            writer.writerow((epoch, train_loss, val_loss))
            log_file.flush()

    return model


@torch.no_grad()
def calc_val_loss(
    model:             nn.Module,
    val_ldr:           DataLoader,
    init_anchor_sizes: torch.Tensor,
    device:            torch.device | str,
) -> float:
    """Compute the mean batch loss in evaluation mode."""

    model.eval()
    output_stride = model.backbone.output_stride
    val_loss_sum  = 0.0
    n_images = 0

    for images, ref_targets in val_ldr:
        loss = calc_batch_loss(model, images, ref_targets, init_anchor_sizes, output_stride, device)
        val_loss_sum += loss.item() * len(images)
        n_images += len(images)

    if n_images == 0:
        raise ValueError("Validation loader produced no images")
    return val_loss_sum / n_images


def calc_batch_loss(
    model:             nn.Module,
    images:            torch.Tensor,
    ref_targets:       list[dict[str, torch.Tensor]],
    init_anchor_sizes: torch.Tensor,
    output_stride:     int,
    device:            torch.device | str,
) -> torch.Tensor:
    images = images.to(device)

    pred_targets = model(images)
    pred_targets = calc_resulting_bboxes(pred_targets, init_anchor_sizes, output_stride) #TODO: it's better to filter ignored anchors before calculate resulting bboxes to avoid useless actions

    pos_pred, neg_pred, matched_gt_boxes, matched_gt_labels = match_anchors(
        pred_targets, ref_targets, init_anchor_sizes, output_stride
    )
    return calc_loss(pos_pred,         neg_pred,
                     matched_gt_boxes, matched_gt_labels,
                     coef_obj=1,       coef_box=1,
                     coef_cls=1,       coef_neg=1
                    )
