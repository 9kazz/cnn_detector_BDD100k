import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from load_data import DatasetBDD100K, collate_fn


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


def main():
    parser = argparse.ArgumentParser(description="Load one BDD100K batch per split.")
    parser.add_argument("--dataset-dir", type=Path, default=Path(__file__).parent / "dataset")
    parser.add_argument("--batch-size",  type=int,  default=16)
    
    args = parser.parse_args()

    train_loader, val_loader = make_loaders(args.dataset_dir, args.batch_size)

if __name__ == "__main__":
    main()
