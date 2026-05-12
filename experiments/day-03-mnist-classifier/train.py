r"""Day 3 PyTorch: complete MNIST training, checkpoint, and inference.

Run training:
    .\.venv\Scripts\python.exe experiments/day-03-mnist-classifier/train.py --epochs 3

Run one-sample inference after training:
    .\.venv\Scripts\python.exe experiments/day-03-mnist-classifier/train.py --infer-only --sample-index 0
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]

os.environ.setdefault("MPLCONFIGDIR", str((PROJECT_ROOT / ".matplotlib-cache").resolve()))

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@dataclass
class EpochMetrics:
    train_loss: float
    val_loss: float
    val_accuracy: float


class MnistMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_dataloaders(data_dir: Path, batch_size: int) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = loss_fn(logits, labels)
            predictions = logits.argmax(dim=1)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (predictions == labels).sum().item()
            total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def save_checkpoint(model: nn.Module, output_dir: Path, epoch: int, accuracy: float) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_mnist_mlp.pt"
    torch.save(
        {
            "epoch": epoch,
            "accuracy": accuracy,
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    return checkpoint_path


def save_curves(metrics: list[EpochMetrics], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(metrics) + 1))
    train_losses = [item.train_loss for item in metrics]
    val_losses = [item.val_loss for item in metrics]
    val_accuracies = [item.val_accuracy for item in metrics]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_losses, label="train loss")
    axes[0].plot(epochs, val_losses, label="val loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs, val_accuracies, label="val accuracy", color="tab:green")
    axes[1].set_xlabel("epoch")
    axes[1].set_title("Validation accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()

    fig.tight_layout()
    plot_path = output_dir / "training_curves.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return plot_path


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = select_device()
    print(f"Using device: {device}")

    train_loader, val_loader = build_dataloaders(args.data_dir, args.batch_size)
    model = MnistMlp().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_accuracy = 0.0
    metrics: list[EpochMetrics] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_accuracy = evaluate(model, val_loader, loss_fn, device)
        metrics.append(EpochMetrics(train_loss, val_loss, val_accuracy))

        print(
            f"epoch {epoch:02d} | "
            f"train_loss {train_loss:.4f} | "
            f"val_loss {val_loss:.4f} | "
            f"val_acc {val_accuracy:.4f}"
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            checkpoint_path = save_checkpoint(model, args.output_dir, epoch, val_accuracy)
            print(f"  saved best checkpoint: {checkpoint_path}")

    curve_path = save_curves(metrics, args.output_dir)
    print(f"Saved curves: {curve_path}")
    print(f"Best validation accuracy: {best_accuracy:.4f}")


def load_model(checkpoint_path: Path, device: torch.device) -> MnistMlp:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = MnistMlp().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with accuracy {checkpoint['accuracy']:.4f}")
    return model


def infer(args: argparse.Namespace) -> None:
    device = select_device()
    print(f"Using device: {device}")
    checkpoint_path = args.output_dir / "best_mnist_mlp.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Run training first.")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    dataset = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
    image, label = dataset[args.sample_index]

    model = load_model(checkpoint_path, device)
    with torch.no_grad():
        logits = model(image.unsqueeze(0).to(device))
        probabilities = torch.softmax(logits, dim=1)
        prediction = probabilities.argmax(dim=1).item()
        confidence = probabilities[0, prediction].item()

    print(f"Sample index: {args.sample_index}")
    print(f"True label: {label}")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = args.output_dir / "sample_prediction.png"
    plt.figure(figsize=(3, 3))
    plt.imshow(image.squeeze(0), cmap="gray")
    plt.title(f"pred={prediction}, label={label}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()
    print(f"Saved sample prediction: {plot_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=EXPERIMENT_DIR / "data")
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR / "outputs")
    parser.add_argument("--infer-only", action="store_true")
    parser.add_argument("--sample-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.infer_only:
        infer(args)
    else:
        train(args)


if __name__ == "__main__":
    main()


