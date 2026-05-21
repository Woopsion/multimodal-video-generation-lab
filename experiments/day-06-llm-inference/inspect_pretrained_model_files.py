from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

from safetensors import safe_open


DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "pretrained" / "sshleifer-tiny-gpt2"


def section(title: str) -> None:
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


def file_size(path: Path) -> str:
    size = path.stat().st_size
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / 1024 / 1024:.1f} MB"


def parse_args() -> Path:
    parser = ArgumentParser(description="Inspect a local pretrained tiny GPT-2 folder.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Local pretrained model directory to inspect.",
    )
    return parser.parse_args().model_dir


def main() -> None:
    model_dir = parse_args()

    section("Step 01: local pretrained model directory")
    print(model_dir)
    if not model_dir.exists():
        print(
            "\nThe local pretrained model directory does not exist.\n"
            "Model files are intentionally not committed to the repository.\n"
            "Download them explicitly when needed, for example:\n"
            f"  .venv/bin/python {Path(__file__).with_name('transformers_real_generation_demo.py')} "
            "--model sshleifer/tiny-gpt2 --allow-download\n"
        )
        raise SystemExit(1)

    print("\nFiles:")
    for path in sorted(model_dir.iterdir()):
        if path.is_file():
            print(f"- {path.name:<28} {file_size(path)}")

    section("Step 02: config.json - model architecture settings")
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    for key in [
        "model_type",
        "vocab_size",
        "n_embd",
        "hidden_size",
        "n_layer",
        "num_hidden_layers",
        "n_head",
        "num_attention_heads",
        "n_positions",
    ]:
        if key in config:
            print(f"{key:<24} = {config[key]}")

    section("Step 03: vocab.json - token string to token_id")
    vocab = json.loads((model_dir / "vocab.json").read_text(encoding="utf-8"))
    print(f"vocab size = {len(vocab)}")
    print("\nSome tokens used by the default prompt:")
    for token in ["Deep", "Ġlearning", "Ġis"]:
        print(f"{token!r:<16} -> {vocab.get(token)}")
    print("\nFirst 12 vocab entries:")
    for index, (token, token_id) in enumerate(vocab.items()):
        if index >= 12:
            break
        print(f"{index:>2}. {token!r:<12} -> {token_id}")

    section("Step 04: merges.txt - BPE merge rules")
    merge_lines = (model_dir / "merges.txt").read_text(encoding="utf-8").splitlines()
    print("First 12 lines:")
    for line in merge_lines[:12]:
        print(line)

    section("Step 05: model.safetensors - pretrained tensor names and shapes")
    weights_path = model_dir / "model.safetensors"
    with safe_open(weights_path, framework="pt", device="cpu") as weights:
        keys = list(weights.keys())
        print(f"number of tensors = {len(keys)}")
        print("\nFirst 24 tensors:")
        for key in keys[:24]:
            tensor = weights.get_tensor(key)
            print(f"{key:<52} shape={tuple(tensor.shape)} dtype={tensor.dtype}")

    section("Summary")
    print(
        "config.json tells Transformers how to build the model architecture.\n"
        "vocab.json and merges.txt tell the tokenizer how to convert text to token IDs.\n"
        "model.safetensors stores the pretrained parameter tensors used by the model."
    )


if __name__ == "__main__":
    main()
