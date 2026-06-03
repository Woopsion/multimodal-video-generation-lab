"""Day 10: minimal CLIP-style retrieval demo.

This script does not load a real CLIP model and does not train anything.
It uses tiny hand-written vectors to make the retrieval flow visible:

image embeddings [num_images, D]
text embeddings  [num_texts, D]
-> scores = image_embeddings @ text_embeddings.T
-> image-to-text top-k retrieval
-> text-to-image top-k retrieval

The goal is to understand ranking mechanics before using a real model.
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


def print_items(title: str, items: list[str]) -> None:
    print(title)
    for index, item in enumerate(items):
        print(f"  {index}: {item}")


def print_topk_for_rows(
    *,
    title: str,
    row_names: list[str],
    column_names: list[str],
    scores: torch.Tensor,
    top_k: int,
    expected_columns: dict[int, int] | None = None,
) -> None:
    section(title)
    values, indices = torch.topk(scores, k=top_k, dim=1)

    print(
        "torch.topk(scores, k, dim=1) sorts each row independently.\n"
        "Each row is one query. The returned indices point into the gallery columns."
    )
    show_tensor("topk_values", values, "Top-k similarity scores for each query row.")
    show_tensor("topk_indices", indices, "Top-k gallery indices for each query row.")

    for row_index, row_name in enumerate(row_names):
        print(f"\nQuery {row_index}: {row_name}")
        expected = None if expected_columns is None else expected_columns.get(row_index)
        for rank, (score, column_index) in enumerate(
            zip(values[row_index].tolist(), indices[row_index].tolist(), strict=True),
            start=1,
        ):
            marker = ""
            if expected is not None and column_index == expected:
                marker = "  <-- expected match"
            print(
                f"  rank {rank}: column {column_index} | "
                f"score = {score:.4f} | {column_names[column_index]}{marker}"
            )


def recall_at_k(
    scores: torch.Tensor,
    expected_columns: dict[int, int],
    top_k: int,
) -> float:
    _, indices = torch.topk(scores, k=top_k, dim=1)
    hits = 0
    for row_index, expected_column in expected_columns.items():
        if expected_column in indices[row_index].tolist():
            hits += 1
    return hits / len(expected_columns)


def main() -> None:
    torch.manual_seed(7)
    torch.set_printoptions(precision=4, sci_mode=False)

    top_k = 3

    image_names = [
        "image 0: red square",
        "image 1: blue circle",
        "image 2: dog on grass",
        "image 3: bowl of noodles",
    ]
    text_names = [
        'text 0: "a red square on a white background"',
        'text 1: "a blue circle on a white background"',
        'text 2: "a dog standing on grass"',
        'text 3: "a bowl of noodles on a table"',
        'text 4: "a red circle on a poster"',
        'text 5: "a dog near a blue ball"',
    ]

    # A tiny semantic space: [red, blue, dog, food, roundness].
    # In real CLIP these vectors come from image/text encoders plus projection heads.
    image_raw = torch.tensor(
        [
            [1.00, 0.05, 0.00, 0.00, 0.10],
            [0.05, 1.00, 0.00, 0.00, 0.95],
            [0.00, 0.15, 1.00, 0.00, 0.10],
            [0.05, 0.00, 0.00, 1.00, 0.35],
        ],
        dtype=torch.float32,
    )
    text_raw = torch.tensor(
        [
            [0.95, 0.05, 0.00, 0.00, 0.08],
            [0.05, 0.95, 0.00, 0.00, 0.90],
            [0.00, 0.10, 0.95, 0.00, 0.05],
            [0.05, 0.00, 0.00, 0.95, 0.30],
            [0.80, 0.05, 0.00, 0.00, 0.90],
            [0.00, 0.55, 0.80, 0.00, 0.45],
        ],
        dtype=torch.float32,
    )

    expected_text_for_image = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
    }
    expected_image_for_text = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
    }

    section("Step 01: define query/gallery items")
    print_items("Image gallery:", image_names)
    print_items("Text gallery:", text_names)
    print(
        "\nFor image-to-text retrieval, each image is a query and texts are the gallery.\n"
        "For text-to-image retrieval, each text is a query and images are the gallery."
    )

    section("Step 02: create and L2-normalize CLIP-style embeddings")
    show_tensor(
        "image_raw",
        image_raw,
        "[num_images, D]: hand-written whole-image vectors before normalization.",
    )
    show_tensor(
        "text_raw",
        text_raw,
        "[num_texts, D]: hand-written whole-text vectors before normalization.",
    )

    image_embeddings = F.normalize(image_raw, dim=1)
    text_embeddings = F.normalize(text_raw, dim=1)
    show_tensor(
        "image_embeddings",
        image_embeddings,
        "[num_images, D]: each row now has L2 norm 1.",
    )
    print(f"image row norms = {image_embeddings.norm(dim=1)}")
    show_tensor(
        "text_embeddings",
        text_embeddings,
        "[num_texts, D]: each row now has L2 norm 1.",
    )
    print(f"text row norms = {text_embeddings.norm(dim=1)}")

    section("Step 03: compute image-text similarity matrix")
    scores = image_embeddings @ text_embeddings.T
    show_tensor(
        "scores",
        scores,
        "[num_images, num_texts]: row i is image i against all text candidates.",
    )

    print_topk_for_rows(
        title="Step 04: image-to-text retrieval",
        row_names=image_names,
        column_names=text_names,
        scores=scores,
        top_k=top_k,
        expected_columns=expected_text_for_image,
    )

    print_topk_for_rows(
        title="Step 05: text-to-image retrieval",
        row_names=text_names,
        column_names=image_names,
        scores=scores.T,
        top_k=top_k,
        expected_columns=expected_image_for_text,
    )

    section("Step 06: tiny retrieval evaluation")
    image_to_text_recall_at_1 = recall_at_k(scores, expected_text_for_image, top_k=1)
    image_to_text_recall_at_3 = recall_at_k(scores, expected_text_for_image, top_k=3)
    text_to_image_recall_at_1 = recall_at_k(scores.T, expected_image_for_text, top_k=1)
    text_to_image_recall_at_3 = recall_at_k(scores.T, expected_image_for_text, top_k=3)

    print(
        "Recall@K asks: does the expected match appear in the first K retrieved items?\n"
        "This tiny demo only evaluates the four exact paired texts/images."
    )
    print(f"image-to-text Recall@1 = {image_to_text_recall_at_1:.3f}")
    print(f"image-to-text Recall@3 = {image_to_text_recall_at_3:.3f}")
    print(f"text-to-image Recall@1 = {text_to_image_recall_at_1:.3f}")
    print(f"text-to-image Recall@3 = {text_to_image_recall_at_3:.3f}")

    section("Summary")
    print("1. Retrieval starts from the same [num_images, num_texts] score matrix as Day9.")
    print("2. Image-to-text retrieval sorts each row of scores.")
    print("3. Text-to-image retrieval sorts each row of scores.T.")
    print("4. top-k returns both scores and gallery indices.")
    print("5. Recall@K checks whether the expected match is inside the first K results.")


if __name__ == "__main__":
    main()
