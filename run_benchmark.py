import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.benchmark import run_benchmark
from src.load_data import DatasetBDD100K, collate_fn
from src.store_load_model import load_model
from src.visualization import get_train_log_metrics, show_train_log_plot


def main() -> float:
    parser = argparse.ArgumentParser(
        description="Evaluate detection loss on a labeled test set and plot training losses.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device",                                         type=torch.device, default='cuda' if torch.cuda.is_available() else 'cpu', help='cpu, cuda or cuda:0')
    parser.add_argument("--image-dir",            "--image_dir",            type=Path,         default=Path('dataset/images/test'),                    help='Test image directory')
    parser.add_argument("--annotation-path",      "--annotation_path",      type=Path,         default=Path('dataset/test.json'),                      help='Labeled test annotations')
    parser.add_argument("--model-dir",            "--model_dir",            type=Path,         required=True,                                          help='Model weights path')
    parser.add_argument("--train-log",            "--train_log",            type=Path,         default=None,                                           help='Training CSV with epoch, train_loss, val_loss')
    parser.add_argument("--log-dir",              "--log_dir",              type=Path,         default=None,                                           help='Directory for loss.png; omit to display')
    parser.add_argument("--batch-size",           "--batch_size",           type=int,          default=16,                                             help='Images per batch')
    parser.add_argument("--confidence-threshold", "--confidence_threshold", type=float,        default=0.25,                                           help='Confidence threshold before NMS')
    parser.add_argument("--nms-iou-threshold",    "--nms_iou_threshold",    type=float,        default=0.5,                                            help='NMS IoU threshold')
    parser.add_argument("--max-detections",       "--max_detections",       type=int,          default=None,                                           help='Maximum detections per image after NMS')
    args = parser.parse_args()
    validate_cli_args(args, parser)

    dataset = DatasetBDD100K(args.image_dir, args.annotation_path)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    model = load_model(args.model_dir, args.device)
    test_loss = run_benchmark(
        model, loader, args.device, args.confidence_threshold, args.nms_iou_threshold, args.max_detections
    )

    if args.train_log is not None:
        metrics = get_train_log_metrics(args.train_log)

        print(f"Train loss (epoch {metrics['epoch'][-1]}): {metrics['train_loss'][-1]:.6f}")
        print(f"Validation loss (epoch {metrics['epoch'][-1]}): {metrics['val_loss'][-1]:.6f}")

        if args.log_dir is not None:
            args.log_dir.mkdir(parents=True, exist_ok=True)
        path = args.log_dir / "loss.png" if args.log_dir is not None else None
        show_train_log_plot(metrics, path=path)

    return test_loss


def validate_cli_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available. Use --device cpu.")
    if not 0 <= args.confidence_threshold <= 1:
        parser.error("--confidence-threshold must be in [0, 1]")
    if not 0 <= args.nms_iou_threshold <= 1:
        parser.error("--nms-iou-threshold must be in [0, 1]")
    if args.max_detections is not None and args.max_detections < 0:
        parser.error("--max-detections must be non-negative")
    if not args.image_dir.is_dir():
        parser.error("--image-dir must point to the test image directory")
    if not args.annotation_path.is_file():
        parser.error("--annotation-path must point to test annotations with boxes and labels")


if __name__ == "__main__":
    main()
