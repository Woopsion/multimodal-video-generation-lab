from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch


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


def explain_missing_transformers() -> None:
    print(
        "The Python package 'transformers' is not installed in the current environment.\n"
        "\n"
        "This Part 5 demo is intentionally a real Hugging Face Transformers demo, so it\n"
        "needs that package before it can load AutoTokenizer and AutoModelForCausalLM.\n"
        "\n"
        "Install step, when you are ready:\n"
        "  uv add transformers safetensors\n"
        "\n"
        "After the dependency is installed, run this script again."
    )


def load_transformers_classes() -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError:
        explain_missing_transformers()
        raise SystemExit(1)
    return AutoModelForCausalLM, AutoTokenizer


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def tensor_shape(name: str, tensor: torch.Tensor, note: str | None = None) -> None:
    line()
    print(f"{name}.shape = {tuple(tensor.shape)}")
    print(f"{name}.dtype = {tensor.dtype}")
    print(f"{name}.device = {tensor.device}")
    if note:
        print(note)


def show_token_table(tokenizer: Any, input_ids: torch.Tensor) -> None:
    line()
    ids = input_ids[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids)
    print("Token table for the first batch item:")
    print("index | token_id | token")
    for index, (token_id, token) in enumerate(zip(ids, tokens, strict=True)):
        print(f"{index:>5} | {token_id:>8} | {token}")


def show_top_candidates(
    tokenizer: Any,
    next_token_logits: torch.Tensor,
    *,
    top_k: int,
) -> int:
    line()
    probabilities = torch.softmax(next_token_logits, dim=-1)
    k = min(top_k, probabilities.shape[-1])
    top_probs, top_ids = torch.topk(probabilities[0], k=k)

    print(f"Top {k} candidates from next_token_logits:")
    print("rank | token_id | probability | decoded token")
    for rank, (token_id, probability) in enumerate(
        zip(top_ids.tolist(), top_probs.tolist(), strict=True),
        start=1,
    ):
        decoded = tokenizer.decode([token_id])
        print(f"{rank:>4} | {token_id:>8} | {probability:>11.6f} | {decoded!r}")

    greedy_token_id = int(top_ids[0].item())
    print(f"\nGreedy next_token_id = {greedy_token_id}")
    print(f"Greedy decoded token = {tokenizer.decode([greedy_token_id])!r}")
    return greedy_token_id


def show_past_key_values(past_key_values: Any) -> None:
    section("Step 05: inspect past_key_values, the real model's KV Cache")
    if past_key_values is None:
        print("outputs.past_key_values is None. This model call did not return a KV Cache.")
        return

    print(f"type(outputs.past_key_values) = {type(past_key_values)}")

    try:
        layer_count = len(past_key_values)
    except TypeError:
        layer_count = None

    if layer_count is not None:
        print(f"number of cached layers = {layer_count}")

    if hasattr(past_key_values, "layers"):
        first_layer = past_key_values.layers[0]
        key_cache = first_layer.keys
        value_cache = first_layer.values
    else:
        try:
            first_layer = past_key_values[0]
            key_cache, value_cache = first_layer[:2]
        except Exception as exc:  # noqa: BLE001 - learning script should explain odd cache types.
            print("Could not inspect the first layer cache.")
            print(f"Reason: {exc}")
            return

    tensor_shape(
        "first_layer_key_cache",
        key_cache,
        note=(
            "[B, H, T, d] in many decoder-only LMs: one cached Key tensor per layer.\n"
            "B=batch size, H=head count, T=visible token count, d=head dimension."
        ),
    )
    tensor_shape(
        "first_layer_value_cache",
        value_cache,
        note=(
            "[B, H, T, d] in many decoder-only LMs: one cached Value tensor per layer.\n"
            "This is the real-library version of the KV Cache shape demo."
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Day 6 Part 5: load a real causal LM with Hugging Face Transformers."
    )
    parser.add_argument(
        "--model",
        default=default_model_reference(),
        help=(
            "Model name or local model path. If the project-local pretrained copy "
            "exists, it is used by default; otherwise the default is sshleifer/tiny-gpt2."
        ),
    )
    parser.add_argument(
        "--prompt",
        default="Deep learning is",
        help="Prompt text. GPT-2-style tiny models work better with English prompts.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=12,
        help="How many new tokens generate() may append.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="How many next-token candidates to print from the first forward pass.",
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
    AutoModelForCausalLM, AutoTokenizer = load_transformers_classes()

    local_files_only = not args.allow_download
    device = choose_device(args.device)

    section("Step 01: load tokenizer and causal language model")
    print(f"model name/path = {args.model}")
    print(f"prompt = {args.prompt!r}")
    print(f"device = {device}")
    print(f"local_files_only = {local_files_only}")
    print(
        "\nThis script is offline/local-only by default. If the model is not already\n"
        "available locally or in the Hugging Face cache, re-run with --allow-download\n"
        "only when you are ready to download remote files."
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            local_files_only=local_files_only,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=local_files_only,
        )
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

    print(f"tokenizer class = {tokenizer.__class__.__name__}")
    print(f"model class = {model.__class__.__name__}")
    print(f"vocab_size from tokenizer = {len(tokenizer)}")
    print(f"vocab_size from model config = {model.config.vocab_size}")
    print(f"hidden_size from model config = {getattr(model.config, 'hidden_size', 'unknown')}")
    print(f"num_layers from model config = {getattr(model.config, 'num_hidden_layers', 'unknown')}")
    print(f"num_heads from model config = {getattr(model.config, 'num_attention_heads', 'unknown')}")

    section("Step 02: tokenizer(prompt) -> input_ids and attention_mask")
    encoded = tokenizer(args.prompt, return_tensors="pt")
    encoded = {name: tensor.to(device) for name, tensor in encoded.items()}

    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")

    tensor_shape(
        "input_ids",
        input_ids,
        note="[B, T]: integer token IDs. This is not an embedding vector yet.",
    )
    if attention_mask is not None:
        tensor_shape(
            "attention_mask",
            attention_mask,
            note="[B, T]: 1 means real token, 0 means padding token.",
        )
    show_token_table(tokenizer, input_ids)

    section("Step 03: model forward -> logits")
    with torch.inference_mode():
        outputs = model(**encoded, use_cache=True)

    logits = outputs.logits
    tensor_shape(
        "logits",
        logits,
        note="[B, T, V]: every position has one raw score for every vocabulary token.",
    )

    next_token_logits = logits[:, -1, :]
    tensor_shape(
        "next_token_logits",
        next_token_logits,
        note="[B, V]: the final position's vocabulary scores for choosing the next token.",
    )

    section("Step 04: inspect next-token candidates")
    greedy_token_id = show_top_candidates(
        tokenizer,
        next_token_logits,
        top_k=args.top_k,
    )

    manually_appended = torch.cat(
        [input_ids, torch.tensor([[greedy_token_id]], device=device, dtype=input_ids.dtype)],
        dim=-1,
    )
    print(f"\ninput_ids after manual greedy append = {manually_appended[0].tolist()}")
    print(f"decoded after manual greedy append = {tokenizer.decode(manually_appended[0])!r}")

    show_past_key_values(getattr(outputs, "past_key_values", None))

    section("Step 06: model.generate() runs the same loop for multiple tokens")
    with torch.inference_mode():
        generated_ids = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    tensor_shape(
        "generated_ids",
        generated_ids,
        note="[B, T + new_tokens]: original prompt IDs plus generated token IDs.",
    )
    print(f"generated_ids[0] = {generated_ids[0].tolist()}")
    print(f"decoded generated text = {tokenizer.decode(generated_ids[0], skip_special_tokens=True)!r}")

    section("Summary")
    print(
        "Real Transformers inference follows the same Day 6 chain:\n"
        "prompt text -> tokenizer -> input_ids -> model forward -> logits -> next token\n"
        "-> append -> repeat -> decode.\n"
        "\n"
        "The main difference from the toy demos is that the tokenizer, Transformer\n"
        "layers, KV Cache behavior, and lm_head weights all come from real saved files."
    )


if __name__ == "__main__":
    main()
