"""Inspect gradients for the Day 2 linear regression experiment.

Run:
    uv run python experiments/day-02-pytorch-autograd-linear-regression/gradient_debug.py
"""

from __future__ import annotations

import argparse

import torch


def make_one_sample_model(lr: float) -> tuple[torch.nn.Linear, torch.optim.Optimizer]:
    model = torch.nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(1.0)
        model.bias.fill_(0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    return model, optimizer


def grad_value(param: torch.nn.Parameter) -> str:
    if param.grad is None:
        return "None"
    return f"{param.grad.item():.6f}"


def one_sample_loss(model: torch.nn.Linear) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.tensor([[2.0]])
    y = torch.tensor([[5.0]])
    pred = model(x)
    error = pred - y
    loss = torch.nn.functional.mse_loss(pred, y)
    return x, y, error, loss


def print_model_state(label: str, model: torch.nn.Linear) -> None:
    print(label)
    print(f"  weight:      {model.weight.item(): .6f}")
    print(f"  bias:        {model.bias.item(): .6f}")
    print(f"  weight.grad: {grad_value(model.weight)}")
    print(f"  bias.grad:   {grad_value(model.bias)}")


def single_step_demo(lr: float) -> None:
    print("\n=== Single sample gradient ===")
    model, optimizer = make_one_sample_model(lr)
    x, y, error, loss = one_sample_loss(model)

    print(f"x: {x.item():.1f}, y: {y.item():.1f}")
    print(f"pred = weight * x + bias = {model.weight.item():.1f} * {x.item():.1f} + {model.bias.item():.1f}")
    print(f"pred:  {model(x).item():.6f}")
    print(f"error: {error.item():.6f}")
    print(f"loss:  {loss.item():.6f}")

    optimizer.zero_grad()
    loss.backward()

    print_model_state("After loss.backward():", model)
    print("Manual formulas for this one sample:")
    print("  d loss / d weight = 2 * error * x = 2 * (-3) * 2 = -12")
    print("  d loss / d bias   = 2 * error     = 2 * (-3)     = -6")

    optimizer.step()
    print_model_state(f"After optimizer.step() with lr={lr}:", model)
    print("Note: optimizer.step() updates parameters, but it does not clear gradients.")


def zero_grad_demo(lr: float) -> None:
    print("\n=== Why optimizer.zero_grad() is needed ===")
    model, optimizer = make_one_sample_model(lr)

    _, _, _, loss = one_sample_loss(model)
    loss.backward()
    print_model_state("After first backward():", model)

    _, _, _, loss = one_sample_loss(model)
    loss.backward()
    print_model_state("After second backward() without zero_grad():", model)

    optimizer.zero_grad()
    print_model_state("After optimizer.zero_grad():", model)

    print("PyTorch accumulates gradients by default.")
    print("Without zero_grad(), the second backward adds new gradients onto the old gradients.")


def full_batch_trace(epochs: int, lr: float) -> None:
    print("\n=== Full-batch training trace ===")
    torch.manual_seed(42)

    x = torch.linspace(-3.0, 3.0, steps=128).unsqueeze(1)
    noise = torch.randn_like(x) * 0.4
    y = 2.0 * x + 1.0 + noise

    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    print("Each epoch uses all 128 samples, computes one mean MSE loss, then updates once.")
    print("epoch | loss      | weight    | bias      | weight.grad | bias.grad")
    print("------+-----------+-----------+-----------+-------------+----------")

    for epoch in range(1, epochs + 1):
        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()

        print(
            f"{epoch:5d} | "
            f"{loss.item():9.6f} | "
            f"{model.weight.item():9.6f} | "
            f"{model.bias.item():9.6f} | "
            f"{model.weight.grad.item():11.6f} | "
            f"{model.bias.grad.item():8.6f}"
        )

        optimizer.step()

    print_model_state("After final optimizer.step():", model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.05)
    args = parser.parse_args()

    single_step_demo(args.lr)
    zero_grad_demo(args.lr)
    full_batch_trace(args.epochs, args.lr)


if __name__ == "__main__":
    main()
