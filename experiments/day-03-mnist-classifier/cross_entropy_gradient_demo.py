r"""Tiny CrossEntropyLoss gradient demo.

This script does not use MNIST. It uses tiny fixed tensors so every number can
be inspected by hand.

Run:
    .\.venv\Scripts\python.exe experiments/day-03-mnist-classifier/cross_entropy_gradient_demo.py

What it demonstrates:
    hidden -> Linear -> logits -> softmax -> CrossEntropyLoss -> gradients
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def line(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def section(title: str) -> None:
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)


def show(name: str, value: torch.Tensor) -> None:
    print(f"{name}:")
    print(value.detach())
    print(f"shape = {tuple(value.shape)}\n")


def build_fixed_linear() -> nn.Linear:
    """Build a Linear(2, 3) layer with deterministic parameters.

    PyTorch stores Linear weights as:
        weight.shape = [out_features, in_features]

    Here:
        in_features = 2   -> hidden has 2 features: h_0, h_1
        out_features = 3  -> three classes: class 0, class 1, class 2
    """

    linear = nn.Linear(in_features=2, out_features=3, bias=True)

    with torch.no_grad():
        linear.weight.copy_(
            torch.tensor(
                [
                    [0.20, -0.30],  # class 0: W_0,0 and W_0,1
                    [-0.40, 0.10],  # class 1: W_1,0 and W_1,1
                    [0.05, 0.60],  # class 2: W_2,0 and W_2,1
                ],
                dtype=torch.float32,
            )
        )
        linear.bias.copy_(
            torch.tensor(
                [0.10, -0.20, 0.05],  # b_0, b_1, b_2
                dtype=torch.float32,
            )
        )

    return linear


def explain_logits_formula(hidden: torch.Tensor, linear: nn.Linear) -> None:
    """Print the explicit z_i = W_i,0*h_0 + W_i,1*h_1 + b_i formula."""

    h0 = hidden[0, 0].item()
    h1 = hidden[0, 1].item()

    print("For this single sample:")
    print(f"h_0 = {h0:.4f}, h_1 = {h1:.4f}\n")

    for class_index in range(linear.out_features):
        w0 = linear.weight[class_index, 0].item()
        w1 = linear.weight[class_index, 1].item()
        b = linear.bias[class_index].item()
        z = w0 * h0 + w1 * h1 + b

        print(
            f"z_{class_index} = "
            f"W_{class_index},0*h_0 + W_{class_index},1*h_1 + b_{class_index}"
        )
        print(
            f"    = ({w0:.4f})*({h0:.4f}) "
            f"+ ({w1:.4f})*({h1:.4f}) "
            f"+ ({b:.4f})"
        )
        print(f"    = {z:.4f}\n")


def run_single_sample_demo() -> None:
    line("Part 1: single sample, 3 classes, 2 hidden features")

    torch.manual_seed(0)
    linear = build_fixed_linear()

    # This is the input to the final Linear layer.
    # In MNIST MLP, this would be the 128-dimensional hidden vector after ReLU.
    # Here we use only 2 dimensions so the math is visible.
    hidden = torch.tensor([[1.50, -2.00]], dtype=torch.float32)

    # Class-index label. It means:
    #   label = 0 -> class 0 is correct
    # PyTorch CrossEntropyLoss accepts class-index labels directly.
    label = torch.tensor([0], dtype=torch.long)

    section("1. Fixed input and Linear parameters")
    show("hidden", hidden)
    show("linear.weight", linear.weight)
    show("linear.bias", linear.bias)
    print("label = 0, meaning class 0 is the correct class.\n")

    section("2. Forward pass: hidden -> logits")
    explain_logits_formula(hidden, linear)

    logits = linear(hidden)
    show("logits = linear(hidden)", logits)

    section("3. Softmax: logits -> probabilities")
    probabilities = torch.softmax(logits, dim=1)
    show("probabilities = softmax(logits)", probabilities)

    print("Softmax definition:")
    print("p_i = exp(z_i) / S")
    print("S = exp(z_0) + exp(z_1) + exp(z_2)")
    print("So p_0 is exactly exp(z_0) / S.\n")

    section("4. One-hot target and CrossEntropyLoss")
    one_hot = F.one_hot(label, num_classes=3).float()
    show("one_hot target", one_hot)

    loss = F.cross_entropy(logits, label, reduction="mean")
    manual_loss = -torch.log(probabilities[0, label.item()])

    print("Because the correct class is 0:")
    print("L = -ln(p_0)")
    print(f"PyTorch loss = {loss.item():.6f}")
    print(f"manual -ln(p_0) = {manual_loss.item():.6f}\n")

    section("5. Backward: dL/dlogits = probabilities - one_hot")
    linear.zero_grad()
    loss.backward()

    manual_dlogits = probabilities - one_hot
    show("manual dL/dlogits = probabilities - one_hot", manual_dlogits)

    print("For class 0:")
    print("dL/dz_0 = p_0 - y_0")
    print(f"         = {probabilities[0, 0].item():.6f} - 1")
    print(f"         = {manual_dlogits[0, 0].item():.6f}\n")

    print("For class 1:")
    print("dL/dz_1 = p_1 - y_1")
    print(f"         = {probabilities[0, 1].item():.6f} - 0")
    print(f"         = {manual_dlogits[0, 1].item():.6f}\n")

    print("For class 2:")
    print("dL/dz_2 = p_2 - y_2")
    print(f"         = {probabilities[0, 2].item():.6f} - 0")
    print(f"         = {manual_dlogits[0, 2].item():.6f}\n")

    section("6. Chain rule: dL/dW_i,k = dL/dz_i * dz_i/dW_i,k")
    print("For the final Linear layer:")
    print("z_i = W_i,0*h_0 + W_i,1*h_1 + b_i")
    print("Therefore:")
    print("dz_i/dW_i,0 = h_0")
    print("dz_i/dW_i,1 = h_1")
    print("dz_i/db_i = 1\n")

    manual_weight_grad = manual_dlogits.T @ hidden
    manual_bias_grad = manual_dlogits.sum(dim=0)

    show("manual weight gradient = dL/dlogits.T @ hidden", manual_weight_grad)
    show("PyTorch linear.weight.grad", linear.weight.grad)

    show("manual bias gradient = dL/dlogits summed by class", manual_bias_grad)
    show("PyTorch linear.bias.grad", linear.bias.grad)

    print("Check:")
    print("manual gradients and PyTorch .grad should be the same.\n")

    section("7. One SGD update and the effect on logits")
    lr = 0.10
    old_logits = logits.detach().clone()

    optimizer = torch.optim.SGD(linear.parameters(), lr=lr)
    optimizer.step()

    new_logits = linear(hidden).detach()

    print(f"learning_rate = {lr}")
    print("Update rule:")
    print("W = W - learning_rate * W.grad")
    print("b = b - learning_rate * b.grad\n")

    show("old logits before optimizer.step()", old_logits)
    show("new logits after optimizer.step()", new_logits)

    print("Because class 0 is correct:")
    print("- z_0 should move upward for this same hidden input.")
    print("- wrong-class logits usually move downward for this same hidden input.")


def run_batch_mean_demo() -> None:
    line("Part 2: batch mean gradient, 2 samples")

    linear = build_fixed_linear()

    # Two samples in one batch.
    # Sample 0: correct class is 0.
    # Sample 1: correct class is 2.
    hidden = torch.tensor(
        [
            [1.50, -2.00],
            [-0.50, 1.00],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 2], dtype=torch.long)

    section("1. Batch input and labels")
    show("hidden batch", hidden)
    print("labels = [0, 2]\n")

    logits = linear(hidden)
    probabilities = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(labels, num_classes=3).float()

    section("2. Forward values")
    show("logits", logits)
    show("probabilities", probabilities)
    show("one_hot targets", one_hot)

    section("3. Per-sample responsibility error: probabilities - one_hot")
    responsibility_error = probabilities - one_hot
    show("responsibility_error", responsibility_error)

    print("Each row belongs to one sample.")
    print("Each column belongs to one class.")
    print("A negative value on the correct class means: push that class logit upward.")
    print("A positive value on a wrong class means: push that class logit downward.\n")

    section("4. CrossEntropyLoss uses reduction='mean' by default")
    loss = F.cross_entropy(logits, labels, reduction="mean")
    print(f"batch loss = {loss.item():.6f}\n")

    linear.zero_grad()
    loss.backward()

    batch_size = hidden.shape[0]
    manual_dlogits_for_mean_loss = responsibility_error / batch_size

    print("Because loss is the average of 2 sample losses:")
    print("dL_batch/dlogits = (probabilities - one_hot) / batch_size\n")
    show("manual dL_batch/dlogits", manual_dlogits_for_mean_loss)

    manual_weight_grad = manual_dlogits_for_mean_loss.T @ hidden
    manual_bias_grad = manual_dlogits_for_mean_loss.sum(dim=0)

    section("5. Batch gradient comparison")
    show("manual weight gradient", manual_weight_grad)
    show("PyTorch linear.weight.grad", linear.weight.grad)
    show("manual bias gradient", manual_bias_grad)
    show("PyTorch linear.bias.grad", linear.bias.grad)

    print("Conclusion:")
    print("A batch does not update parameters once per sample.")
    print("It first averages the loss, then backward produces one averaged gradient.")
    print("optimizer.step() then updates each parameter once.")


def main() -> None:
    torch.set_printoptions(precision=6, sci_mode=False)
    run_single_sample_demo()
    run_batch_mean_demo()


if __name__ == "__main__":
    main()
