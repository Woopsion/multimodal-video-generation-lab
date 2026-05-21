from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


DEFAULT_LOCAL_MODEL_DIR = Path(__file__).resolve().parent / "pretrained" / "sshleifer-tiny-gpt2"
DEFAULT_MODEL_NAME = "sshleifer/tiny-gpt2"


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


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Day 6 Part 6: compare real generation controls on a tiny GPT-2 model."
    )
    parser.add_argument(
        "--model",
        default=default_model_reference(),
        help=(
            "Local model path or Hugging Face model name. If the project-local "
            "pretrained copy exists, it is used by default; otherwise the default "
            "is sshleifer/tiny-gpt2."
        ),
    )
    parser.add_argument(
        "--prompt",
        default="Deep learning is",
        help="Prompt text. The tiny GPT-2 test model works best with English prompts.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: auto, cpu, cuda, or mps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used by sampling cases.",
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


def decode_token_ids(tokenizer: Any, token_ids: torch.Tensor) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def explain_stop_reason(
    *,
    new_token_ids: torch.Tensor,
    eos_token_id: int | None,
    requested_new_tokens: int,
) -> str:
    if eos_token_id is not None and eos_token_id in new_token_ids.tolist():
        return "stopped because eos_token_id appeared"
    if new_token_ids.numel() >= requested_new_tokens:
        return "stopped because max_new_tokens was reached"
    return "stopped by a generation rule before max_new_tokens"


def run_case(
    *,
    case_name: str,
    tokenizer: Any,
    model: Any,
    encoded: dict[str, torch.Tensor],
    generation_kwargs: dict[str, Any],
) -> None:
    line()
    print(case_name)
    print("generation kwargs:")
    for key, value in generation_kwargs.items():
        print(f"  {key}: {value}")

    with torch.inference_mode():
        generated_ids = model.generate(
            **encoded,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            **generation_kwargs,
        )

    input_len = encoded["input_ids"].shape[1]
    new_token_ids = generated_ids[0, input_len:]
    stop_reason = explain_stop_reason(
        new_token_ids=new_token_ids,
        eos_token_id=tokenizer.eos_token_id,
        requested_new_tokens=generation_kwargs["max_new_tokens"],
    )

    print(f"generated_ids.shape = {tuple(generated_ids.shape)}")
    print(f"new_token_count = {new_token_ids.numel()}")
    print(f"new_token_ids = {new_token_ids.tolist()}")
    print(f"new tokens = {tokenizer.convert_ids_to_tokens(new_token_ids.tolist())}")
    print(f"stop reason = {stop_reason}")
    print(f"decoded text = {decode_token_ids(tokenizer, generated_ids[0])!r}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    local_files_only = not args.allow_download

    section("Step 01: load local model and encode prompt")
    print(f"model path = {args.model}")
    print(f"prompt = {args.prompt!r}")
    print(f"device = {device}")
    print(f"seed = {args.seed}")
    print(f"local_files_only = {local_files_only}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=local_files_only)
        model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=local_files_only)
    except OSError as exc:
        print("\nModel/tokenizer loading failed.")
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

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded = tokenizer(args.prompt, return_tensors="pt")
    encoded = {name: tensor.to(device) for name, tensor in encoded.items()}

    print(f"input_ids = {encoded['input_ids'][0].tolist()}")
    print(f"tokens = {tokenizer.convert_ids_to_tokens(encoded['input_ids'][0].tolist())}")
    print(f"eos_token_id = {tokenizer.eos_token_id}")
    print(f"pad_token_id = {tokenizer.pad_token_id}")

    section("Step 02: compare generation controls")
    print(
        "The model is intentionally tiny, so output quality is not the point.\n"
        "The goal is to observe which parameter changes the generation procedure."
    )

    cases: list[tuple[str, dict[str, Any]]] = [
        (
            "Case A - greedy baseline: deterministic, always take the strongest next token",
            {
                "max_new_tokens": 12,
                "do_sample": False,
            },
        ),
        (
            "Case B - shorter limit: same greedy rule, but max_new_tokens stops earlier",
            {
                "max_new_tokens": 4,
                "do_sample": False,
            },
        ),
        (
            "Case C - sampling with low temperature: random, but still relatively conservative",
            {
                "max_new_tokens": 12,
                "do_sample": True,
                "temperature": 0.7,
                "top_k": 20,
            },
        ),
        (
            "Case D - sampling with high temperature: random and flatter probability distribution",
            {
                "max_new_tokens": 12,
                "do_sample": True,
                "temperature": 1.4,
                "top_k": 20,
            },
        ),
        (
            "Case E - top_k: sample only from the top 5 candidates at each step",
            {
                "max_new_tokens": 12,
                "do_sample": True,
                "top_k": 5,
                "temperature": 1.0,
            },
        ),
        (
            "Case F - top_p: keep the smallest candidate set whose probability mass reaches 0.8",
            {
                "max_new_tokens": 12,
                "do_sample": True,
                "top_p": 0.8,
                "temperature": 1.0,
            },
        ),
        (
            "Case G - repetition penalty: reduce the score of tokens that already appeared",
            {
                "max_new_tokens": 12,
                "do_sample": False,
                "repetition_penalty": 1.25,
            },
        ),
    ]

    for index, (case_name, generation_kwargs) in enumerate(cases):
        set_seed(args.seed + index)
        run_case(
            case_name=case_name,
            tokenizer=tokenizer,
            model=model,
            encoded=encoded,
            generation_kwargs=generation_kwargs,
        )

    section("Summary")
    print(
        "max_new_tokens controls the upper bound of newly generated tokens.\n"
        "do_sample=False makes generation deterministic and greedy-like.\n"
        "temperature changes how sharp or flat the probability distribution is.\n"
        "top_k limits sampling to a fixed number of high-probability candidates.\n"
        "top_p limits sampling by cumulative probability mass.\n"
        "repetition_penalty discourages repeating tokens, but it cannot make a tiny model smart.\n"
        "eos_token_id can stop generation early if the model emits the end-of-sequence token."
    )


if __name__ == "__main__":
    main()
