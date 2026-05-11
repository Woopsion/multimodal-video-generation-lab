
"""Day 2 PyTorch basics: tensors, autograd, and linear regression.

Run:
    uv run python experiments/pytorch-basics/linear_regression.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib.pyplot as plt
import torch


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autograd_demo(device: torch.device) -> None:
    x = torch.tensor([2.0], device=device, requires_grad=True)
    y = x * x + 3 * x + 1
    y.backward()

    print("Autograd demo")
    print(f"  x: {x.item():.1f}")
    print(f"  y = x^2 + 3x + 1: {y.item():.1f}")
    print(f"  dy/dx: {x.grad.item():.1f}")


def make_dataset(n_samples: int, noise_std: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    x = torch.linspace(-3.0, 3.0, steps=n_samples).unsqueeze(1)
    noise = torch.randn_like(x) * noise_std
    y = 2.0 * x + 1.0 + noise
    return x.to(device), y.to(device)


def train_linear_regression(
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    lr: float,
) -> tuple[torch.nn.Module, list[float]]:
    model = torch.nn.Linear(1, 1).to(x.device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    losses: list[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            print(f"epoch {epoch:03d} | loss {losses[-1]:.6f}")

    return model, losses


def save_plot(x: torch.Tensor, y: torch.Tensor, model: torch.nn.Module, losses: list[float], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    x_cpu = x.detach().cpu()
    y_cpu = y.detach().cpu()

    model.eval()
    with torch.no_grad():
        pred_cpu = model(x).detach().cpu()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(x_cpu, y_cpu, s=14, label="data")
    axes[0].plot(x_cpu, pred_cpu, color="tab:red", label="model")
    axes[0].set_title("Linear fit")
    axes[0].legend()

    axes[1].plot(losses)
    axes[1].set_title("Training loss")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("MSE")

    fig.tight_layout()
    plot_path = output_dir / "linear_regression_result.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    print(f"Saved plot: {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--n-samples", type=int, default=128)
    parser.add_argument("--noise-std", type=float, default=0.4)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/pytorch-basics/outputs"))
    args = parser.parse_args()

    device = select_device()
    print(f"Using device: {device}")

    autograd_demo(device)
    x, y = make_dataset(args.n_samples, args.noise_std, device)
    model, losses = train_linear_regression(x, y, args.epochs, args.lr)

    weight = model.weight.detach().cpu().item()
    bias = model.bias.detach().cpu().item()
    print(f"Learned function: y = {weight:.3f}x + {bias:.3f}")

    save_plot(x, y, model, losses, args.output_dir)


if __name__ == "__main__":
    main()
