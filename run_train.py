import argparse
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.arch import DetectionCNN
from src.load_data import CLASS_NAMES, DatasetBDD100K, collate_fn
from src.train import ANCHOR_SIZES, run_training
from src.store_load_model import store_model, init_model


def make_loaders(dataset_dir: Path, batch_size: int):
    train_dataset = DatasetBDD100K(
        dataset_dir / "images" / "train", dataset_dir / "train.json"
    )
    val_dataset = DatasetBDD100K(
        dataset_dir / "images" / "val", dataset_dir / "val.json"
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    return train_loader, val_loader


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the anchor-based CNN on BDD100K.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device",                                 type=torch.device, default='cuda' if torch.cuda.is_available() else 'cpu', help='Training device, e.g. cpu, cuda, or cuda:0')
    parser.add_argument("--model-dir",        "--model_dir",        type=Path,         default=Path('model.pt'),                               help='Output model weights path')
    parser.add_argument("--log-dir",          "--log_dir",          type=Path,         default=Path('train.csv'),                              help='Output CSV path')
    parser.add_argument("--epoch",                                  type=int,          default=20,                                             help='Number of training epochs')
    parser.add_argument("--dataset-dir",      "--dataset_dir",      type=Path,         default=Path(__file__).parent / 'dataset',              help='Directory with images, train.json, and val.json')
    parser.add_argument("--batch-size",       "--batch_size",       type=int,          default=16,                                             help='Images per batch')
    parser.add_argument("--init-weights-dir", "--init_weights_dir", type=Path,         default=None,                                           help='Initial model weights from a .pt file')
    args = parser.parse_args()
    validate_cli_args(args, parser)

    train_loader, val_loader = make_loaders(args.dataset_dir, args.batch_size)

    model = DetectionCNN(n_anchors=len(ANCHOR_SIZES), n_classes=len(CLASS_NAMES)).to(args.device)
    
    if args.init_weights_dir is not None:
        model = init_model(model, args.init_weights_dir, args.device)
        
    optimizer = Adam(model.parameters(), lr=1e-3)
    
    args.log_dir.parent.mkdir(parents=True, exist_ok=True)
    run_training(
        model, train_loader, val_loader, optimizer,
        n_epoch=args.epoch, device=args.device, log_dir=args.log_dir,
    )
    
    store_model(model, args.model_dir)


def validate_cli_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.epoch < 1:
        parser.error("--epoch must be positive")
    if args.device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available. Use --device cpu.")


if __name__ == "__main__":
    main()
