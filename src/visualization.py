import csv
from pathlib import Path

import matplotlib.pyplot as plt


def get_train_log_metrics(path: str | Path) -> dict:
    metrics = {"epoch": [], "train_loss": [], "val_loss": []}
    with open(path, "r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            metrics["epoch"].append(int(row["epoch"]))
            metrics["train_loss"].append(float(row["train_loss"]))
            metrics["val_loss"].append(float(row["val_loss"]))
    if not metrics["epoch"]:
        raise ValueError("Training log contains no epochs")
    return metrics


def show_train_log_plot(metrics: dict, 
                        path: str | Path | None = None
) -> None:
    fig, ax = plt.subplots(figsize=(16, 7), layout="constrained")
    ax.plot(metrics["epoch"], metrics["train_loss"], label="Train")
    ax.plot(metrics["epoch"], metrics["val_loss"], label="Validation")
    ax.set_title("Object detection loss")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if path is not None:
        fig.savefig(path, dpi=150)
    else:
        plt.show()
    plt.close(fig)
