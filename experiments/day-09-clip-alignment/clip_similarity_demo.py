"""Day 9: minimal CLIP similarity and contrastive loss shape demo.

This script does not load a real CLIP model and does not train anything.
It only makes the core CLIP alignment math visible:

image_embeddings [B, D]
text_embeddings  [B, D]
-> logits = image_embeddings @ text_embeddings.T  [B, B]
-> labels = [0, 1, ..., B - 1]
-> image-to-text CrossEntropyLoss
-> text-to-image CrossEntropyLoss
-> final CLIP loss

The tensors are tiny on purpose, so every printed number can be inspected.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def section(title: str) -> None:
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


def show_tensor(name: str, tensor: torch.Tensor, note: str | None = None) -> None:
    print(f"{name}.shape = {tuple(tensor.shape)}")
    print(f"{name}.dtype = {tensor.dtype}")
    if note:
        print(note)
    print(f"{name} =")
    print(tensor)


def format_vector(tensor: torch.Tensor) -> list[float]:
    return [round(float(x), 4) for x in tensor.detach().cpu()]


def main() -> None:
    torch.manual_seed(42)
    torch.set_printoptions(precision=4, sci_mode=False)

    batch_size = 3
    shared_dim = 4
    logit_scale = 10.0

    section("Step 01: create tiny paired image/text embeddings")
    print(f"batch_size B = {batch_size}")
    print(f"shared_dim D = {shared_dim}")
    print(
        "Each row is one whole-image or whole-text vector in the shared CLIP space, "
        "not a patch vector and not a token vector."
    )

    # These raw vectors stand in for outputs from image/text encoders plus projection heads.
    # In a real model, gradients would flow back into ViT/Text Transformer/projection weights.
    image_raw = torch.tensor(
        [
            [1.00, 0.20, 0.10, 0.00],
            [0.10, 1.00, 0.20, 0.10],
            [0.00, 0.10, 1.00, 0.30],
        ],
        requires_grad=True,
    )
    text_raw = torch.tensor(
        [
            [0.90, 0.25, 0.10, 0.00],
            [0.15, 0.95, 0.25, 0.10],
            [0.05, 0.10, 0.90, 0.35],
        ],
        requires_grad=True,
    )

    show_tensor(
        "image_raw",
        image_raw,
        "[B, D]: three whole-image vectors before L2 normalization.",
    )
    show_tensor(
        "text_raw",
        text_raw,
        "[B, D]: three whole-text vectors before L2 normalization.",
    )

    section("Step 02: L2-normalize embeddings")
    print(
        "F.normalize(x, dim=1) normalizes each row vector independently by its L2 norm.\n"
        "It does not subtract the row mean and does not divide by the row standard deviation."
    )
    image_embeddings = F.normalize(image_raw, dim=1)
    text_embeddings = F.normalize(text_raw, dim=1)
    image_embeddings.retain_grad()
    text_embeddings.retain_grad()

    show_tensor(
        "image_embeddings",
        image_embeddings,
        "[B, D]: each row is now a unit-length whole-image embedding.",
    )
    print(f"image row norms = {image_embeddings.norm(dim=1)}")
    show_tensor(
        "text_embeddings",
        text_embeddings,
        "[B, D]: each row is now a unit-length whole-text embedding.",
    )
    print(f"text row norms = {text_embeddings.norm(dim=1)}")

    section("Step 03: compute CLIP similarity logits")
    cosine_similarities = image_embeddings @ text_embeddings.T
    logits = logit_scale * cosine_similarities

    show_tensor(
        "cosine_similarities",
        cosine_similarities,
        "[B, B]: row i is image i against every text. Diagonal cells are matched pairs.",
    )
    print(f"logit_scale = {logit_scale}")
    show_tensor(
        "logits",
        logits,
        "[B, B]: scaled raw scores before softmax/CrossEntropyLoss.",
    )

    section("Step 04: create labels for diagonal matches")
    labels = torch.arange(batch_size)
    show_tensor(
        "labels",
        labels,
        "labels[i] = i means image i should match text i, so the diagonal is correct.",
    )

    section("Step 05: image-to-text direction")
    print(
        "Interpret logits as a classification table from images to texts:\n"
        "  - rows are images: image 0, image 1, image 2\n"
        "  - columns are candidate texts: text 0, text 1, text 2\n"
        "  - row i asks: given image i, which text is the correct match?\n"
        "  - labels[i] = i means the correct column for row i is the diagonal cell."
    )
    image_to_text_probs = logits.softmax(dim=1)
    per_image_loss = F.cross_entropy(logits, labels, reduction="none")
    loss_i2t = per_image_loss.mean()

    show_tensor(
        "image_to_text_probs",
        image_to_text_probs,
        "softmax(dim=1): each row sums to 1, so each image chooses among all text columns.",
    )
    print(f"row sums = {image_to_text_probs.sum(dim=1)}")
    print("Per-image loss details:")
    for image_index in range(batch_size):
        target_text_index = int(labels[image_index])
        p_correct = image_to_text_probs[image_index, target_text_index]
        manual_loss = -torch.log(p_correct)
        print(
            f"  image {image_index} -> target text {target_text_index} | "
            f"logits row = {format_vector(logits[image_index])} | "
            f"prob row = {format_vector(image_to_text_probs[image_index])} | "
            f"p_correct = {p_correct.item():.6f} | "
            f"-log(p_correct) = {manual_loss.item():.6f}"
        )
    print(f"per_image_loss = {per_image_loss}")
    print(f"loss_i2t = mean(per_image_loss) = {loss_i2t.item():.6f}")

    section("Step 06: text-to-image direction")
    print(
        "Now read the same logits table in the opposite direction.\n"
        "CrossEntropyLoss expects each row to be one sample and each column to be one class.\n"
        "For text-to-image, the samples are texts and the classes are images, so we use logits.T:\n"
        "  - rows of logits.T are texts: text 0, text 1, text 2\n"
        "  - columns of logits.T are candidate images: image 0, image 1, image 2\n"
        "  - row j asks: given text j, which image is the correct match?\n"
        "  - labels[j] = j means the correct image column is again the diagonal cell."
    )
    text_to_image_probs = logits.T.softmax(dim=1)
    per_text_loss = F.cross_entropy(logits.T, labels, reduction="none")
    loss_t2i = per_text_loss.mean()

    show_tensor(
        "text_to_image_probs",
        text_to_image_probs,
        "softmax(dim=1) after transpose: each row sums to 1, so each text chooses among all image columns.",
    )
    print(f"row sums = {text_to_image_probs.sum(dim=1)}")
    print("Per-text loss details:")
    for text_index in range(batch_size):
        target_image_index = int(labels[text_index])
        p_correct = text_to_image_probs[text_index, target_image_index]
        manual_loss = -torch.log(p_correct)
        print(
            f"  text {text_index} -> target image {target_image_index} | "
            f"logits.T row = {format_vector(logits.T[text_index])} | "
            f"prob row = {format_vector(text_to_image_probs[text_index])} | "
            f"p_correct = {p_correct.item():.6f} | "
            f"-log(p_correct) = {manual_loss.item():.6f}"
        )
    print(f"per_text_loss = {per_text_loss}")
    print(f"loss_t2i = mean(per_text_loss) = {loss_t2i.item():.6f}")

    section("Step 07: average the two contrastive losses")
    loss = (loss_i2t + loss_t2i) / 2
    print(f"loss = (loss_i2t + loss_t2i) / 2 = {loss.item():.6f}")

    section("Step 08: backpropagate once to prove gradients flow")
    loss.backward()

    show_tensor(
        "image_embeddings.grad",
        image_embeddings.grad,
        "Gradient on whole-image embeddings. In a real CLIP model this continues into image encoder weights.",
    )
    show_tensor(
        "text_embeddings.grad",
        text_embeddings.grad,
        "Gradient on whole-text embeddings. In a real CLIP model this continues into text encoder weights.",
    )
    show_tensor(
        "image_raw.grad",
        image_raw.grad,
        "Gradient reaches the raw image-side vectors before normalization.",
    )
    show_tensor(
        "text_raw.grad",
        text_raw.grad,
        "Gradient reaches the raw text-side vectors before normalization.",
    )

    section("Summary")
    print("1. image_embeddings and text_embeddings are both [B, D].")
    print("2. image_embeddings @ text_embeddings.T produces [B, B].")
    print("3. labels = [0, 1, 2] says the diagonal cells are the correct matches.")
    print("4. CLIP loss averages image-to-text and text-to-image CrossEntropyLoss.")
    print("5. backward() sends gradients back through both image and text sides.")


if __name__ == "__main__":
    main()
