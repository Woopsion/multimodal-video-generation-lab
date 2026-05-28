"""Day 9: real CLIP inference demo with Hugging Face Transformers.

This script loads a real CLIP model when it is available locally, or when
--allow-download is explicitly passed. It does not train anything.

Flow:

demo image + candidate texts
-> AutoProcessor
-> pixel_values / input_ids / attention_mask
-> CLIPModel forward
-> image_embeds / text_embeds
-> logits_per_image [num_images, num_texts]
-> softmax probabilities over text candidates

The default image is a simple generated red square, so the demo can run without
shipping image files in the repo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from sympy.physics import pring

DEFAULT_LOCAL_MODEL_DIR = (
    Path(__file__).resolve().parent / "pretrained" / "openai-clip-vit-base-patch32"
)
DEFAULT_MODEL_NAME = "openai/clip-vit-base-patch32"
DEFAULT_TEXTS = [
    "a red square on a white background",
    "a blue circle on a white background",
    "a photo of a dog on grass",
    "a bowl of noodles on a table",
]


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


def create_demo_image() -> Image.Image:
    image = Image.new("RGB", (224, 224), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((52, 52, 172, 172), fill=(220, 30, 30), outline=(120, 0, 0), width=4)
    return image


def load_image(image_path: str | None) -> Image.Image:
    if image_path is None:
        return create_demo_image()
    return Image.open(image_path).convert("RGB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Day 9: run real CLIP image-text matching with Transformers."
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
        "--image",
        default=None,
        help="Optional image path. If omitted, the script creates a simple red-square image.",
    )
    parser.add_argument(
        "--texts",
        nargs="*",
        default=DEFAULT_TEXTS,
        help="Candidate text prompts. CLIP ViT-B/32 works best with English captions.",
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
    print(f"vision hidden_size = {model.config.vision_config.hidden_size}")
    print(f"text hidden_size = {model.config.text_config.hidden_size}")
    print(f"model config shape = {model.config}")
    print(
        "dimension flow:\n"
        f"  vision tower hidden states use {model.config.vision_config.hidden_size} dims, "
        f"then visual_projection maps them to {model.config.projection_dim} dims.\n"
        f"  text tower hidden states use {model.config.text_config.hidden_size} dims, "
        f"then text_projection maps them to {model.config.projection_dim} dims.\n"
        "  only outputs.image_embeds and outputs.text_embeds must share projection_dim."
    )

    section("Step 02: prepare one image and candidate texts")
    image = load_image(args.image)
    texts = args.texts
    print(f"image source = {args.image if args.image else 'generated red-square image'}")
    print(f"image.size = {image.size}")
    print("candidate texts:")
    for index, text in enumerate(texts):
        print(f"  {index}: {text!r}")

    section("Step 03: processor(image, texts) -> tensors")
    encoded = processor(
        text=texts,
        images=image,
        return_tensors="pt",
        padding=True,
    )
    encoded = {name: tensor.to(device) for name, tensor in encoded.items()}

    tensor_shape(
        "pixel_values",
        encoded["pixel_values"],
        "[num_images, C, H, W]: normalized image tensor ready for the vision encoder.",
    )
    tensor_shape(
        "input_ids",
        encoded["input_ids"],
        "[num_texts, L]: integer token IDs for candidate text prompts.",
    )
    if "attention_mask" in encoded:
        tensor_shape(
            "attention_mask",
            encoded["attention_mask"],
            "[num_texts, L]: 1 means real token, 0 means padding token.",
        )

    section("Step 04: model forward -> image/text embeddings and logits")
    with torch.inference_mode():
        outputs = model(**encoded)
    print(f"outputs = {outputs}")
    tensor_shape(
        "outputs.image_embeds",
        outputs.image_embeds,
        "[num_images, projection_dim]: whole-image embeddings in CLIP shared space.",
    )
    tensor_shape(
        "outputs.text_embeds",
        outputs.text_embeds,
        "[num_texts, projection_dim]: whole-text embeddings in CLIP shared space.",
    )
    tensor_shape(
        "outputs.logits_per_image",
        outputs.logits_per_image,
        "[num_images, num_texts]: image-to-text matching scores before softmax.",
    )
    tensor_shape(
        "outputs.logits_per_text",
        outputs.logits_per_text,
        "[num_texts, num_images]: text-to-image matching scores before softmax.",
    )

    section("Step 05: softmax over text candidates")
    probabilities = outputs.logits_per_image.softmax(dim=1)
    tensor_shape(
        "probabilities",
        probabilities,
        "[num_images, num_texts]: each row sums to 1 over candidate texts.",
    )

    print("\nRanking for image 0:")
    probs = probabilities[0].detach().cpu()
    order = torch.argsort(probs, descending=True)
    print("rank | text_index | probability | text")
    for rank, text_index in enumerate(order.tolist(), start=1):
        print(
            f"{rank:>4} | {text_index:>10} | {probs[text_index].item():>11.6f} | "
            f"{texts[text_index]!r}"
        )

    section("Summary")
    print("1. The processor creates pixel_values for the image and input_ids for texts.")
    print("2. CLIPModel returns whole-image and whole-text embeddings in the same projection_dim.")
    print("3. logits_per_image has shape [num_images, num_texts].")
    print("4. softmax(logits_per_image, dim=1) turns text scores into probabilities.")
    print("5. The highest-probability text is the model's best match for the image.")


if __name__ == "__main__":
    main()
