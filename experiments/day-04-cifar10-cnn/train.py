r"""Day 4 PyTorch: CIFAR-10 CNN training, benchmark logging, and inference.

Run training:
    .\.venv\Scripts\python.exe experiments/day-04-cifar10-cnn/train.py --epochs 5 --batch-size 128

Run one-sample inference after training:
    .\.venv\Scripts\python.exe experiments/day-04-cifar10-cnn/train.py --infer-only --sample-index 0
"""

from __future__ import annotations

import argparse
import os
import time
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


CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


@dataclass
class EpochMetrics:
    train_loss: float
    val_loss: float
    val_accuracy: float
    epoch_seconds: float


class Cifar10Cnn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        return self.classifier(features)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_dataloaders(data_dir: Path, batch_size: int) -> tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform)
    val_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=val_transform)

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
    checkpoint_path = output_dir / "best_cifar10_cnn.pt"
    torch.save(
        {
            "epoch": epoch,
            "accuracy": accuracy,
            "model_state_dict": model.state_dict(),
            "classes": CIFAR10_CLASSES,
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


def get_peak_memory_mb(device: torch.device) -> float | None:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024**2)
    return None


def append_benchmark_log(
    log_path: Path,
    device: torch.device,
    batch_size: int,
    epochs: int,
    metrics: list[EpochMetrics],
    peak_memory_mb: float | None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    best_accuracy = max(item.val_accuracy for item in metrics)
    total_seconds = sum(item.epoch_seconds for item in metrics)
    memory_text = f"{peak_memory_mb:.1f} MB" if peak_memory_mb is not None else "not available"

    with log_path.open("a", encoding="utf-8") as file:
        file.write("\n## CIFAR-10 CNN run\n\n")
        file.write(f"- device: `{device}`\n")
        file.write(f"- batch size: `{batch_size}`\n")
        file.write(f"- epochs: `{epochs}`\n")
        file.write(f"- best val accuracy: `{best_accuracy:.4f}`\n")
        file.write(f"- total seconds: `{total_seconds:.1f}`\n")
        file.write(f"- peak CUDA memory: `{memory_text}`\n")


def train(args: argparse.Namespace) -> None:
    device = select_device()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train_loader, val_loader = build_dataloaders(args.data_dir, args.batch_size)
    model = Cifar10Cnn().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"device: {device}")
    print(f"batch_size: {args.batch_size}")
    print(f"train batches: {len(train_loader)}, val batches: {len(val_loader)}")

    metrics: list[EpochMetrics] = []
    best_accuracy = 0.0

    for epoch in range(1, args.epochs + 1):
        started_at = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_accuracy = evaluate(model, val_loader, loss_fn, device)
        epoch_seconds = time.perf_counter() - started_at

        metrics.append(EpochMetrics(train_loss, val_loss, val_accuracy, epoch_seconds))

        print(
            f"epoch {epoch:02d} | "
            f"train_loss {train_loss:.4f} | "
            f"val_loss {val_loss:.4f} | "
            f"val_acc {val_accuracy:.4f} | "
            f"seconds {epoch_seconds:.1f}"
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            checkpoint_path = save_checkpoint(model, args.output_dir, epoch, val_accuracy)
            print(f"  saved best checkpoint: {checkpoint_path}")

    curves_path = save_curves(metrics, args.output_dir)
    peak_memory_mb = get_peak_memory_mb(device)
    append_benchmark_log(args.benchmark_log, device, args.batch_size, args.epochs, metrics, peak_memory_mb)

    print(f"saved curves: {curves_path}")
    print(f"updated benchmark log: {args.benchmark_log}")


def load_model(checkpoint_path: Path, device: torch.device) -> Cifar10Cnn:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = Cifar10Cnn().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with accuracy {checkpoint['accuracy']:.4f}")
    return model


def denormalize_cifar10(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor((0.4914, 0.4822, 0.4465)).view(3, 1, 1)
    std = torch.tensor((0.2470, 0.2435, 0.2616)).view(3, 1, 1)
    return torch.clamp(image.cpu() * std + mean, 0.0, 1.0)


def infer(args: argparse.Namespace) -> None:
    device = select_device()
    checkpoint_path = args.output_dir / "best_cifar10_cnn.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Run training first.")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    dataset = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=transform)
    image, label = dataset[args.sample_index]

    model = load_model(checkpoint_path, device)
    with torch.no_grad():
        logits = model(image.unsqueeze(0).to(device))
        probabilities = torch.softmax(logits, dim=1)
        confidence, prediction = probabilities.max(dim=1)

    predicted_label = int(prediction.item())
    confidence_value = float(confidence.item())

    print(f"true label: {label} ({CIFAR10_CLASSES[label]})")
    print(f"prediction: {predicted_label} ({CIFAR10_CLASSES[predicted_label]})")
    print(f"confidence: {confidence_value:.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = args.output_dir / "sample_prediction.png"
    display_image = denormalize_cifar10(image).permute(1, 2, 0).numpy()
    plt.figure(figsize=(4, 4))
    plt.imshow(display_image)
    plt.axis("off")
    plt.title(f"true={CIFAR10_CLASSES[label]}, pred={CIFAR10_CLASSES[predicted_label]} ({confidence_value:.2%})")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()
    print(f"Saved sample prediction: {plot_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=EXPERIMENT_DIR / "data")
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR / "outputs")
    parser.add_argument("--benchmark-log", type=Path, default=PROJECT_ROOT / "logs" / "gpu-benchmark.md")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
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
