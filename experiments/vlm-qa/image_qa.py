"""Day 12: minimal real VLM image question-answering demo.

Flow:

image path + question text
-> AutoProcessor
-> pixel_values / input_ids / attention_mask
-> AutoModelForImageTextToText.generate
-> processor.batch_decode
-> natural-language answer

The script is offline/local-only by default. Pass --allow-download only when you
are ready to download the model files from Hugging Face.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


DEFAULT_MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"
DEFAULT_LOCAL_MODEL_DIR = (
    Path(__file__).resolve().parent
    / "pretrained"
    / "HuggingFaceTB-SmolVLM-256M-Instruct"
)
DEFAULT_ASSET_DIR = Path(__file__).resolve().parent / "assets"
DEFAULT_TASKS_PATH = Path(__file__).resolve().parent / "sample_tasks.jsonl"


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


def choose_device(requested: str) -> Any:
    import torch

    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def tensor_shape(name: str, value: Any, note: str | None = None) -> None:
    if not hasattr(value, "shape"):
        return
    line()
    print(f"{name}.shape = {tuple(value.shape)}")
    print(f"{name}.dtype = {value.dtype}")
    print(f"{name}.device = {value.device}")
    if note:
        print(note)


def load_transformers_classes() -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ModuleNotFoundError:
        print(
            "The Python package 'transformers' is not installed.\n"
            "Install it with: uv add transformers safetensors pillow"
        )
        raise SystemExit(1)
    return AutoProcessor, AutoModelForImageTextToText


def load_image(path: str | Path) -> Image.Image:
    image_path = Path(path)
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        raise SystemExit(1)
    return Image.open(image_path).convert("RGB")


def load_tasks(path: str | Path) -> list[dict[str, str]]:
    task_path = Path(path)
    if not task_path.exists():
        print(f"Task file not found: {task_path}")
        raise SystemExit(1)
    tasks: list[dict[str, str]] = []
    for line_no, raw_line in enumerate(task_path.read_text(encoding="utf-8").splitlines(), start=1):
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            task = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            print(f"Invalid JSON on line {line_no}: {exc}")
            raise SystemExit(1) from exc
        for field in ("id", "image", "question", "expected"):
            if field not in task:
                print(f"Task line {line_no} misses field: {field}")
                raise SystemExit(1)
        tasks.append(task)
    return tasks


def move_inputs_to_device(inputs: Any, device: Any) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {name: value.to(device) if hasattr(value, "to") else value for name, value in inputs.items()}


def build_prompt(processor: Any, question: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]
    if hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(messages, add_generation_prompt=True)
    return f"User: <image>\n{question}\nAssistant:"


def encode_inputs(processor: Any, image: Image.Image, prompt: str) -> Any:
    try:
        return processor(images=[image], text=prompt, return_tensors="pt")
    except Exception:
        return processor(images=image, text=prompt, return_tensors="pt")


def decode_answer(processor: Any, generated_ids: Any, input_token_count: int | None) -> str:
    if input_token_count is not None and generated_ids.ndim == 2:
        new_ids = generated_ids[:, input_token_count:]
    else:
        new_ids = generated_ids
    return processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()


def maybe_cuda_memory_mb(device: Any) -> float | None:
    import torch

    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024**2)


def reset_cuda_memory(device: Any) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def load_processor_and_model(args: argparse.Namespace) -> tuple[Any, Any, Any, bool]:
    import torch

    AutoProcessor, AutoModelForImageTextToText = load_transformers_classes()
    device = choose_device(args.device)
    local_files_only = not args.allow_download

    section("Step 01: load VLM processor and model")
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
            trust_remote_code=args.trust_remote_code,
        )
        dtype = torch.bfloat16 if device.type == "cuda" and args.dtype == "auto" else None
        if args.dtype == "float16":
            dtype = torch.float16
        elif args.dtype == "bfloat16":
            dtype = torch.bfloat16
        elif args.dtype == "float32":
            dtype = torch.float32

        model_kwargs: dict[str, Any] = {
            "local_files_only": local_files_only,
            "trust_remote_code": args.trust_remote_code,
        }
        if dtype is not None:
            model_kwargs["dtype"] = dtype

        model = AutoModelForImageTextToText.from_pretrained(args.model, **model_kwargs)
    except OSError as exc:
        print("\nModel/processor loading failed.")
        print(f"Reason: {exc}")
        print(
            "\nCommon fix when you allow a first download:\n"
            f"  uv run python {sys.argv[0]} --model {DEFAULT_MODEL_NAME} --allow-download\n"
            "\nThe default is intentionally offline/local-only so this learning script does\n"
            "not download model files without your explicit choice."
        )
        raise SystemExit(1) from exc

    model.to(device)
    model.eval()

    print(f"processor class = {processor.__class__.__name__}")
    print(f"model class = {model.__class__.__name__}")
    print(f"model dtype parameter sample = {next(model.parameters()).dtype}")
    print(f"model config type = {model.config.__class__.__name__}")
    return processor, model, device, local_files_only


def run_one_question(
    *,
    processor: Any,
    model: Any,
    device: Any,
    image_path: str | Path,
    question: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    import torch

    image = load_image(image_path)
    prompt = build_prompt(processor, question)

    section("Step 02: processor(image, question) -> model tensors")
    print(f"image path = {image_path}")
    print(f"image.size = {image.size}")
    print(f"question = {question!r}")
    print("prompt from chat template:")
    print(prompt)

    encoded = encode_inputs(processor, image, prompt)
    encoded = move_inputs_to_device(encoded, device)

    input_token_count = None
    if "input_ids" in encoded:
        input_token_count = int(encoded["input_ids"].shape[-1])
        tensor_shape("input_ids", encoded["input_ids"], "[B, T]: text token IDs plus image placeholders.")
    if "attention_mask" in encoded:
        tensor_shape("attention_mask", encoded["attention_mask"], "[B, T]: 1 means visible token, 0 means padding.")
    if "pixel_values" in encoded:
        tensor_shape("pixel_values", encoded["pixel_values"], "Image tensor prepared for the vision encoder.")
    if "pixel_attention_mask" in encoded:
        tensor_shape("pixel_attention_mask", encoded["pixel_attention_mask"], "Image patch mask prepared by the processor.")

    section("Step 03: model.generate -> output token ids")
    reset_cuda_memory(device)
    start = time.perf_counter()
    with torch.inference_mode():
        generated_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    elapsed = time.perf_counter() - start

    tensor_shape("generated_ids", generated_ids, "[B, input_tokens + answer_tokens].")
    answer = decode_answer(processor, generated_ids, input_token_count)
    peak_memory_mb = maybe_cuda_memory_mb(device)

    section("Step 04: decode answer and record boundary")
    print(f"answer = {answer!r}")
    print(f"elapsed_seconds = {elapsed:.3f}")
    print(f"peak_cuda_memory_mb = {peak_memory_mb if peak_memory_mb is not None else 'n/a'}")

    return {
        "image": str(image_path),
        "question": question,
        "answer": answer,
        "elapsed_seconds": round(elapsed, 3),
        "peak_cuda_memory_mb": None if peak_memory_mb is None else round(peak_memory_mb, 1),
    }


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def create_sample_images(asset_dir: Path = DEFAULT_ASSET_DIR, tasks_path: Path = DEFAULT_TASKS_PATH) -> None:
    asset_dir.mkdir(parents=True, exist_ok=True)
    font_large = load_font(34)
    font_mid = load_font(24)
    font_small = load_font(18)

    scene_path = asset_dir / "scene_desk.png"
    image = Image.new("RGB", (720, 480), "#f8f4ea")
    draw = ImageDraw.Draw(image)
    draw.rectangle((60, 300, 660, 410), fill="#c9a26a", outline="#6b4a24", width=4)
    draw.rectangle((250, 150, 470, 295), fill="#2f3a3a", outline="#111", width=4)
    draw.rectangle((270, 170, 450, 270), fill="#bfe7ff")
    draw.rectangle((220, 295, 500, 320), fill="#242424")
    draw.ellipse((520, 235, 610, 325), fill="#ffffff", outline="#6b4a24", width=4)
    draw.ellipse((544, 258, 586, 294), fill="#8b4a2f")
    draw.arc((598, 258, 650, 306), start=270, end=90, fill="#6b4a24", width=5)
    draw.rectangle((105, 220, 220, 330), fill="#f6f0c4", outline="#6b6b35", width=3)
    draw.line((120, 248, 205, 248), fill="#9d8b45", width=3)
    draw.line((120, 276, 205, 276), fill="#9d8b45", width=3)
    draw.line((135, 325, 230, 255), fill="#c62828", width=8)
    draw.text((70, 38), "Desk scene", fill="#202427", font=font_large)
    draw.text((70, 82), "Laptop, notebook, pen, and coffee cup", fill="#555", font=font_mid)
    image.save(scene_path)

    ocr_path = asset_dir / "ocr_ui.png"
    image = Image.new("RGB", (720, 480), "#edf4ef")
    draw = ImageDraw.Draw(image)
    draw.rectangle((95, 65, 625, 415), fill="#fffaf0", outline="#2f3836", width=5)
    draw.text((150, 100), "MODEL LAB", fill="#0f766e", font=font_large)
    draw.text((150, 160), "Status: READY", fill="#202427", font=font_mid)
    draw.text((150, 210), "Input image loaded", fill="#65706c", font=font_mid)
    draw.rectangle((150, 285, 390, 350), fill="#b4532a", outline="#7a2f12", width=3)
    draw.text((184, 302), "RUN DEMO", fill="#ffffff", font=font_mid)
    draw.text((150, 378), "Footer: Day12 VLM QA", fill="#65706c", font=font_small)
    image.save(ocr_path)

    chart_path = asset_dir / "chart_sales.png"
    image = Image.new("RGB", (720, 480), "#fffaf0")
    draw = ImageDraw.Draw(image)
    draw.text((70, 40), "Quarterly Demo Scores", fill="#202427", font=font_large)
    origin = (100, 390)
    draw.line((origin[0], 90, origin[0], origin[1]), fill="#202427", width=4)
    draw.line((origin[0], origin[1], 650, origin[1]), fill="#202427", width=4)
    values = [("Q1", 3), ("Q2", 5), ("Q3", 2), ("Q4", 7)]
    colors = ["#0f766e", "#5b4b8a", "#b4532a", "#2f3836"]
    max_value = 8
    bar_width = 70
    gap = 62
    for index, ((label, value), color) in enumerate(zip(values, colors, strict=True)):
        x0 = 145 + index * (bar_width + gap)
        bar_height = int((value / max_value) * 260)
        y0 = origin[1] - bar_height
        draw.rectangle((x0, y0, x0 + bar_width, origin[1]), fill=color)
        draw.text((x0 + 18, origin[1] + 16), label, fill="#202427", font=font_mid)
        draw.text((x0 + 22, y0 - 32), str(value), fill="#202427", font=font_mid)
    draw.text((70, 424), "Q4 is tallest; Q3 is lowest.", fill="#65706c", font=font_mid)
    image.save(chart_path)

    tasks = [
        {
            "id": "scene_desk",
            "type": "scene",
            "image": str(scene_path),
            "question": "What objects are on the desk, and where is the coffee cup relative to the laptop?",
            "expected": "Laptop in the center, notebook and pen on the left, coffee cup to the right of the laptop.",
        },
        {
            "id": "ocr_ui",
            "type": "ocr",
            "image": str(ocr_path),
            "question": "What is the button label and what is the status text?",
            "expected": "Button label: RUN DEMO. Status text: READY.",
        },
        {
            "id": "chart_sales",
            "type": "chart",
            "image": str(chart_path),
            "question": "Which quarter has the highest score, and which quarter has the lowest score?",
            "expected": "Q4 has the highest score. Q3 has the lowest score.",
        },
    ]
    tasks_path.write_text("\n".join(json.dumps(task, ensure_ascii=False) for task in tasks) + "\n", encoding="utf-8")

    section("Created Day12 sample images and task file")
    print(f"asset_dir = {asset_dir}")
    print(f"tasks_path = {tasks_path}")
    for task in tasks:
        print(f"- {task['id']}: {task['image']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Day 12: run minimal real VLM image QA.")
    parser.add_argument("--model", default=default_model_reference(), help="Model name or local model path.")
    parser.add_argument("--image", default=None, help="Image path for a single question.")
    parser.add_argument("--question", default=None, help="Question text for a single image.")
    parser.add_argument("--tasks", default=None, help="JSONL task file with id, image, question, expected.")
    parser.add_argument("--output", default=None, help="Optional JSONL path for saved results.")
    parser.add_argument("--create-samples", action="store_true", help="Create three local sample images and sample_tasks.jsonl.")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, or mps.")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"], help="Model dtype.")
    parser.add_argument("--max-new-tokens", type=int, default=96, help="Maximum new answer tokens.")
    parser.add_argument("--allow-download", action="store_true", help="Allow downloading model files if not cached locally.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to Transformers.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.create_samples:
        create_sample_images()
        if args.image is None and args.tasks is None:
            return

    if args.tasks is None and (args.image is None or args.question is None):
        print(
            "Provide either:\n"
            "  --image PATH --question TEXT\n"
            "or:\n"
            "  --tasks experiments/vlm-qa/sample_tasks.jsonl\n"
            "\nTo create sample images first:\n"
            "  uv run python experiments/vlm-qa/image_qa.py --create-samples"
        )
        raise SystemExit(1)

    processor, model, device, _ = load_processor_and_model(args)

    results: list[dict[str, Any]] = []
    if args.tasks:
        tasks = load_tasks(args.tasks)
        for index, task in enumerate(tasks, start=1):
            section(f"Task {index}/{len(tasks)}: {task['id']} ({task.get('type', 'unknown')})")
            print(f"expected = {task['expected']}")
            result = run_one_question(
                processor=processor,
                model=model,
                device=device,
                image_path=task["image"],
                question=task["question"],
                max_new_tokens=args.max_new_tokens,
            )
            result.update({"id": task["id"], "type": task.get("type"), "expected": task["expected"]})
            results.append(result)
    else:
        results.append(
            run_one_question(
                processor=processor,
                model=model,
                device=device,
                image_path=args.image,
                question=args.question,
                max_new_tokens=args.max_new_tokens,
            )
        )

    section("Summary")
    for result in results:
        print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            "\n".join(json.dumps(result, ensure_ascii=False) for result in results) + "\n",
            encoding="utf-8",
        )
        print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
