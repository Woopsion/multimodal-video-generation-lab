"""Day 10: real CLIP retrieval demo with Hugging Face Transformers.

This script loads a real CLIP model when it is available locally, or when
--allow-download is explicitly passed. It does not train anything.

Flow:

demo images + candidate texts
-> AutoProcessor
-> CLIPModel forward
-> logits_per_image [num_images, num_texts]
-> image-to-text top-k retrieval
-> text-to-image top-k retrieval

The default images are simple generated drawings, so the demo can run without
shipping binary image files in the repo. CLIP usually works better with natural
images, but synthetic images are enough to study retrieval shapes and ranking.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

# The local teaching run can leave a harmless resource_tracker warning when
# Transformers/Torch shuts down. Keep demo output focused on retrieval results.
os.environ.setdefault("PYTHONWARNINGS", "ignore:resource_tracker:UserWarning")

import torch
from PIL import Image, ImageDraw


DEFAULT_LOCAL_MODEL_DIR = (
    Path(__file__).resolve().parent / "pretrained" / "openai-clip-vit-base-patch32"
)
DEFAULT_MODEL_NAME = "openai/clip-vit-base-patch32"
DEFAULT_TEXTS = [
    "a red square on a white background",
    "a blue circle on a white background",
    "a green triangle on a white background",
    "a bowl of noodles on a table",
    "a photo of a dog on grass",
    "a city street at night",
    "a person riding a horse",
]
DEFAULT_EXPECTED_TEXT_FOR_IMAGE = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
}
DEFAULT_EXPECTED_IMAGE_FOR_TEXT = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
}
DEFAULT_NO_ANSWER_TEXT_ROWS = {4, 5, 6}


def default_model_reference() -> str:
    if DEFAULT_LOCAL_MODEL_DIR.exists():
        return str(DEFAULT_LOCAL_MODEL_DIR)
    return DEFAULT_MODEL_NAME


def section(title: str) -> None:
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


def line() -> None:
    print("-" * 96)


def tensor_shape(name: str, tensor: torch.Tensor, note: str | None = None) -> None:
    line()
    print(f"{name}.shape = {tuple(tensor.shape)}")
    print(f"{name}.dtype = {tensor.dtype}")
    print(f"{name}.device = {tensor.device}")
    if note:
        print(note)


def explain_missing_transformers() -> None:
    print(
        "The Python package 'transformers' is not installed in the current environment.\n"
        "\n"
        "This demo uses Hugging Face Transformers to load CLIPModel and AutoProcessor.\n"
        "\n"
        "Install step, when you are ready:\n"
        "  uv add transformers safetensors pillow\n"
        "\n"
        "After the dependency is installed, run this script again."
    )


def load_transformers_classes() -> tuple[Any, Any]:
    try:
        from transformers import AutoProcessor, CLIPModel
    except ModuleNotFoundError:
        explain_missing_transformers()
        raise SystemExit(1)
    return AutoProcessor, CLIPModel


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def draw_red_square() -> Image.Image:
    image = Image.new("RGB", (224, 224), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((52, 52, 172, 172), fill=(220, 30, 30), outline=(120, 0, 0), width=4)
    return image


def draw_blue_circle() -> Image.Image:
    image = Image.new("RGB", (224, 224), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((48, 48, 176, 176), fill=(40, 90, 220), outline=(10, 30, 130), width=4)
    return image


def draw_green_triangle() -> Image.Image:
    image = Image.new("RGB", (224, 224), "white")
    draw = ImageDraw.Draw(image)
    draw.polygon([(112, 42), (42, 176), (182, 176)], fill=(40, 170, 80), outline=(0, 90, 35))
    draw.line([(112, 42), (42, 176), (182, 176), (112, 42)], fill=(0, 90, 35), width=4)
    return image


def draw_noodle_bowl() -> Image.Image:
    image = Image.new("RGB", (224, 224), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((42, 114, 182, 188), fill=(230, 230, 235), outline=(90, 90, 100), width=4)
    draw.arc((54, 88, 170, 170), start=10, end=170, fill=(210, 150, 40), width=4)
    draw.arc((62, 94, 162, 166), start=10, end=170, fill=(210, 150, 40), width=4)
    draw.arc((70, 100, 154, 162), start=10, end=170, fill=(210, 150, 40), width=4)
    draw.line((60, 98, 178, 72), fill=(120, 80, 40), width=4)
    draw.line((66, 108, 184, 82), fill=(120, 80, 40), width=4)
    return image


def create_demo_images() -> tuple[list[str], list[Image.Image]]:
    names = [
        "image 0: generated red square",
        "image 1: generated blue circle",
        "image 2: generated green triangle",
        "image 3: generated noodle bowl drawing",
    ]
    images = [
        draw_red_square(),
        draw_blue_circle(),
        draw_green_triangle(),
        draw_noodle_bowl(),
    ]
    return names, images


def load_images(image_paths: list[str] | None) -> tuple[list[str], list[Image.Image]]:
    if not image_paths:
        return create_demo_images()

    names = []
    images = []
    for index, image_path in enumerate(image_paths):
        path = Path(image_path)
        names.append(f"image {index}: {path.name}")
        images.append(Image.open(path).convert("RGB"))
    return names, images


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
    no_answer_rows: set[int] | None = None,
    accept_threshold: float | None = None,
) -> None:
    section(title)
    k = min(top_k, scores.shape[1])
    values, indices = torch.topk(scores, k=k, dim=1)

    tensor_shape(
        "topk_values",
        values,
        f"[num_queries, {k}]: top-{k} CLIP scores for each query.",
    )
    tensor_shape(
        "topk_indices",
        indices,
        f"[num_queries, {k}]: gallery indices for each query.",
    )

    for row_index, row_name in enumerate(row_names):
        print(f"\nQuery {row_index}: {row_name}")
        expected_column = None if expected_columns is None else expected_columns.get(row_index)
        no_answer = no_answer_rows is not None and row_index in no_answer_rows
        for rank, (score, column_index) in enumerate(
            zip(values[row_index].tolist(), indices[row_index].tolist(), strict=True),
            start=1,
        ):
            marker = ""
            if expected_column is not None and column_index == expected_column:
                marker = "  <-- expected match"
            print(
                f"  rank {rank}: column {column_index} | "
                f"score = {score:.4f} | {column_names[column_index]}{marker}"
            )
        if accept_threshold is not None:
            top1_score = values[row_index, 0].item()
            top1_column = indices[row_index, 0].item()
            accepted = top1_score >= accept_threshold
            if no_answer and accepted:
                print(
                    f"  decision: FALSE ACCEPT | top1 score {top1_score:.4f} >= "
                    f"threshold {accept_threshold:.4f}, but this query has no matching gallery item."
                )
            elif no_answer:
                print(
                    f"  decision: REJECT | top1 score {top1_score:.4f} < "
                    f"threshold {accept_threshold:.4f}; no-answer query is rejected."
                )
            elif expected_column is not None and top1_column == expected_column:
                print(
                    f"  decision: HIT | expected column {expected_column} is rank 1 "
                    f"with score {top1_score:.4f}."
                )
            elif expected_column is not None:
                print(
                    f"  decision: MISS | expected column {expected_column}, "
                    f"but rank 1 is column {top1_column}."
                )


def recall_at_k(scores: torch.Tensor, expected_columns: dict[int, int], top_k: int) -> float:
    k = min(top_k, scores.shape[1])
    _, indices = torch.topk(scores, k=k, dim=1)
    hits = 0
    for row_index, expected_column in expected_columns.items():
        if row_index >= scores.shape[0]:
            continue
        if expected_column in indices[row_index].tolist():
            hits += 1
    return hits / len(expected_columns)


def print_threshold_report(
    *,
    scores: torch.Tensor,
    query_names: list[str],
    gallery_names: list[str],
    expected_columns: dict[int, int],
    no_answer_rows: set[int],
    accept_threshold: float,
) -> None:
    section("Step 07: success/failure cases and no-answer threshold")
    print(
        "This section is only meaningful for the default generated demo gallery.\n"
        "The threshold is a teaching heuristic for this run, not a universal CLIP rule."
    )

    values, indices = torch.topk(scores, k=1, dim=1)
    print(f"accept_threshold = {accept_threshold:.4f}")
    print("case | query | top1 score | top1 result | expected | decision")
    for row_index, query_name in enumerate(query_names):
        top1_score = values[row_index, 0].item()
        top1_column = indices[row_index, 0].item()
        top1_name = gallery_names[top1_column]
        expected = expected_columns.get(row_index)

        if row_index in no_answer_rows:
            expected_text = "NO_ANSWER"
            decision = "correct reject" if top1_score < accept_threshold else "false accept"
        elif expected is None:
            expected_text = "not evaluated"
            decision = "not evaluated"
        elif top1_column == expected:
            expected_text = f"column {expected}"
            decision = "success"
        else:
            expected_text = f"column {expected}"
            decision = "failure"

        print(
            f"{row_index:>4} | {query_name} | {top1_score:>10.4f} | "
            f"{top1_name} | {expected_text} | {decision}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Day 10: run real CLIP image-text retrieval with Transformers."
    )
    parser.add_argument(
        "--model",
        default=default_model_reference(),
        help=(
            "Model name or local model path. If a project-local pretrained copy exists, "
            "it is used by default; otherwise the default is openai/clip-vit-base-patch32."
        ),
    )
    parser.add_argument(
        "--images",
        nargs="*",
        default=None,
        help=(
            "Optional image paths. If omitted, the script creates four simple demo drawings."
        ),
    )
    parser.add_argument(
        "--texts",
        nargs="*",
        default=DEFAULT_TEXTS,
        help="Candidate text prompts. CLIP ViT-B/32 works best with English captions.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of retrieval results to print for each query.",
    )
    parser.add_argument(
        "--accept-threshold",
        type=float,
        default=28.0,
        help=(
            "Teaching threshold for accepting top-1 retrieval results in the default demo. "
            "Real systems must calibrate this per model, data, and score scale."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: auto, cpu, cuda, or mps.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help=(
            "Allow Transformers to download model files if they are not cached locally. "
            "By default this script uses local_files_only=True."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    AutoProcessor, CLIPModel = load_transformers_classes()

    local_files_only = not args.allow_download
    device = choose_device(args.device)

    section("Step 01: load real CLIP processor and model")
    print(f"model name/path = {args.model}")
    print(f"device = {device}")
    print(f"local_files_only = {local_files_only}")
    print(f"top_k = {args.top_k}")
    print(
        "\nThis script is offline/local-only by default. If the model is not already\n"
        "available locally or in the Hugging Face cache, re-run with --allow-download\n"
        "only when you are ready to download remote files."
    )

    try:
        processor = AutoProcessor.from_pretrained(
            args.model,
            local_files_only=local_files_only,
        )
        model = CLIPModel.from_pretrained(
            args.model,
            local_files_only=local_files_only,
        )
    except OSError as exc:
        print("\nModel/processor loading failed.")
        print(f"Reason: {exc}")
        print(
            "\nCommon fix:\n"
            f"  .venv/bin/python {sys.argv[0]} --model {DEFAULT_MODEL_NAME} --allow-download\n"
            "\nThe default is intentionally offline/local-only so this learning script does\n"
            "not download model files without your explicit choice."
        )
        raise SystemExit(1)

    model.to(device)
    model.eval()

    print(f"processor class = {processor.__class__.__name__}")
    print(f"model class = {model.__class__.__name__}")
    print(f"projection_dim = {model.config.projection_dim}")

    section("Step 02: prepare image gallery and text gallery")
    image_names, images = load_images(args.images)
    text_names = [f'text {index}: "{text}"' for index, text in enumerate(args.texts)]
    using_default_demo = args.images is None and args.texts == DEFAULT_TEXTS
    print_items("Image gallery:", image_names)
    print_items("Text gallery:", text_names)
    print(f"using_default_demo = {using_default_demo}")

    section("Step 03: processor(images, texts) -> tensors")
    encoded = processor(
        text=args.texts,
        images=images,
        return_tensors="pt",
        padding=True,
    )
    encoded = {name: tensor.to(device) for name, tensor in encoded.items()}

    tensor_shape(
        "pixel_values",
        encoded["pixel_values"],
        "[num_images, C, H, W]: normalized image tensor batch.",
    )
    tensor_shape(
        "input_ids",
        encoded["input_ids"],
        "[num_texts, L]: token IDs for all candidate text prompts.",
    )
    if "attention_mask" in encoded:
        tensor_shape(
            "attention_mask",
            encoded["attention_mask"],
            "[num_texts, L]: 1 means real token, 0 means padding token.",
        )

    section("Step 04: model forward -> embeddings and score matrices")
    with torch.inference_mode():
        outputs = model(**encoded)

    tensor_shape(
        "outputs.image_embeds",
        outputs.image_embeds,
        "[num_images, projection_dim]: one shared-space vector per image.",
    )
    tensor_shape(
        "outputs.text_embeds",
        outputs.text_embeds,
        "[num_texts, projection_dim]: one shared-space vector per text.",
    )
    tensor_shape(
        "outputs.logits_per_image",
        outputs.logits_per_image,
        "[num_images, num_texts]: image-to-text retrieval scores.",
    )
    tensor_shape(
        "outputs.logits_per_text",
        outputs.logits_per_text,
        "[num_texts, num_images]: text-to-image retrieval scores.",
    )

    scores_i2t = outputs.logits_per_image.detach().cpu()
    scores_t2i = outputs.logits_per_text.detach().cpu()

    print_topk_for_rows(
        title="Step 05: image-to-text top-k retrieval",
        row_names=image_names,
        column_names=text_names,
        scores=scores_i2t,
        top_k=args.top_k,
        expected_columns=DEFAULT_EXPECTED_TEXT_FOR_IMAGE if using_default_demo else None,
        accept_threshold=args.accept_threshold if using_default_demo else None,
    )
    print_topk_for_rows(
        title="Step 06: text-to-image top-k retrieval",
        row_names=text_names,
        column_names=image_names,
        scores=scores_t2i,
        top_k=args.top_k,
        expected_columns=DEFAULT_EXPECTED_IMAGE_FOR_TEXT if using_default_demo else None,
        no_answer_rows=DEFAULT_NO_ANSWER_TEXT_ROWS if using_default_demo else None,
        accept_threshold=args.accept_threshold if using_default_demo else None,
    )

    if using_default_demo:
        print_threshold_report(
            scores=scores_t2i,
            query_names=text_names,
            gallery_names=image_names,
            expected_columns=DEFAULT_EXPECTED_IMAGE_FOR_TEXT,
            no_answer_rows=DEFAULT_NO_ANSWER_TEXT_ROWS,
            accept_threshold=args.accept_threshold,
        )

        section("Step 08: retrieval metrics on known paired items")
        print(
            "Recall@K is computed only for the four queries that have known paired answers.\n"
            "The no-answer text queries are evaluated by the threshold report instead."
        )
        for k in (1, args.top_k):
            print(
                f"image-to-text Recall@{k} = "
                f"{recall_at_k(scores_i2t, DEFAULT_EXPECTED_TEXT_FOR_IMAGE, k):.3f}"
            )
            print(
                f"text-to-image Recall@{k} = "
                f"{recall_at_k(scores_t2i, DEFAULT_EXPECTED_IMAGE_FOR_TEXT, k):.3f}"
            )

    section("Summary")
    print("1. Real CLIP retrieval still starts from a [num_images, num_texts] score matrix.")
    print("2. logits_per_image rows answer: given this image, which texts rank highest?")
    print("3. logits_per_text rows answer: given this text, which images rank highest?")
    print("4. Retrieval uses ranking/top-k over CLIP scores; softmax is optional for display.")
    print("5. Recall@K evaluates known paired items; no-answer queries need threshold/reject logic.")
    print("6. The default generated drawings are for shape/ranking study, not benchmark quality.")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
