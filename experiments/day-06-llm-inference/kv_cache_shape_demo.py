from __future__ import annotations

import math

import torch
from torch import nn


torch.manual_seed(42)
torch.set_printoptions(precision=4, sci_mode=False)


def section(title: str) -> None:
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


def line() -> None:
    print("-" * 96)


def show_tensor(name: str, tensor: torch.Tensor, note: str | None = None) -> None:
    line()
    print(name)
    print(tensor)
    print(f"{name}.shape = {tuple(tensor.shape)}")
    if note:
        print(note)


def project_to_qkv(
    hidden_states: torch.Tensor,
    qkv_projection: nn.Linear,
    *,
    num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, seq_len, embed_dim = hidden_states.shape
    qkv_packed = qkv_projection(hidden_states)
    qkv = qkv_packed.view(batch_size, seq_len, 3, num_heads, head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    query, key, value = qkv.unbind(dim=0)
    return query, key, value


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    head_dim = query.shape[-1]
    scores = query @ key.transpose(-2, -1) / math.sqrt(head_dim)
    weights = torch.softmax(scores, dim=-1)
    context = weights @ value
    return scores, weights, context


def main() -> None:
    batch_size = 1
    prompt_len = 4
    max_new_tokens = 3
    embed_dim = 8
    num_heads = 2
    head_dim = embed_dim // num_heads

    qkv_projection = nn.Linear(in_features=embed_dim, out_features=3 * embed_dim)
    prompt_hidden = torch.randn(batch_size, prompt_len, embed_dim)
    new_token_hiddens = torch.randn(batch_size, max_new_tokens, embed_dim)

    section("Step 01: symbols and the problem KV Cache solves")
    print(
        f"B = batch_size = {batch_size}\n"
        f"T_prompt = prompt_len = {prompt_len}\n"
        f"N = max_new_tokens = {max_new_tokens}\n"
        f"D = embed_dim = {embed_dim}\n"
        f"H = num_heads = {num_heads}\n"
        f"d = head_dim = D / H = {head_dim}\n"
    )
    print(
        "During autoregressive generation, every new token can attend to all previous tokens.\n"
        "The key observation:\n"
        "- Old tokens' K and V do not change after they are computed.\n"
        "- Without KV Cache, we recompute old K/V at every generation step.\n"
        "- With KV Cache, we store old K/V and only compute K/V for the newest token."
    )
    show_tensor(
        "prompt_hidden",
        prompt_hidden,
        note="[B, T_prompt, D]: hidden vectors for the prompt tokens.",
    )

    section("Step 02: no KV Cache - recompute the whole sequence every step")
    no_cache_projected_tokens = 0
    no_cache_attention_cells = 0
    no_cache_hidden = prompt_hidden.clone()

    for step in range(1, max_new_tokens + 1):
        current_len = no_cache_hidden.shape[1]
        query, key, value = project_to_qkv(
            no_cache_hidden,
            qkv_projection,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        scores, weights, context = scaled_dot_product_attention(query, key, value)

        no_cache_projected_tokens += current_len
        no_cache_attention_cells += batch_size * num_heads * current_len * current_len

        print(f"\nGeneration step {step} without KV Cache")
        print(f"current sequence length S = {current_len}")
        print(f"query.shape = {tuple(query.shape)} = [B, H, S, d]")
        print(f"key.shape   = {tuple(key.shape)} = [B, H, S, d]")
        print(f"value.shape = {tuple(value.shape)} = [B, H, S, d]")
        print(f"scores.shape = {tuple(scores.shape)} = [B, H, S, S]")
        print(f"context.shape = {tuple(context.shape)} = [B, H, S, d]")
        print(
            "Meaning: even though generation only needs the last position's next-token logits,\n"
            "this path recomputes Q/K/V and attention for the entire sequence."
        )

        if step < max_new_tokens:
            next_hidden = new_token_hiddens[:, step - 1 : step, :]
            no_cache_hidden = torch.cat([no_cache_hidden, next_hidden], dim=1)

    section("Step 03: with KV Cache - prefill prompt once")
    cache_query, cache_key, cache_value = project_to_qkv(
        prompt_hidden,
        qkv_projection,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    prefill_scores, prefill_weights, prefill_context = scaled_dot_product_attention(
        cache_query,
        cache_key,
        cache_value,
    )
    key_cache = cache_key
    value_cache = cache_value

    cache_projected_tokens = prompt_len
    cache_attention_cells = batch_size * num_heads * prompt_len * prompt_len

    show_tensor(
        "key_cache after prefill",
        key_cache,
        note="[B, H, T_prompt, d]: K vectors for all prompt tokens are stored.",
    )
    show_tensor(
        "value_cache after prefill",
        value_cache,
        note="[B, H, T_prompt, d]: V vectors for all prompt tokens are stored.",
    )
    print(f"prefill_scores.shape = {tuple(prefill_scores.shape)} = [B, H, T_prompt, T_prompt]")
    print(
        "Prefill means: process the prompt once, store its K/V, and keep those tensors\n"
        "for later generation steps."
    )

    section("Step 04: with KV Cache - decode one new token at a time")
    for step in range(1, max_new_tokens + 1):
        new_hidden = new_token_hiddens[:, step - 1 : step, :]
        new_query, new_key, new_value = project_to_qkv(
            new_hidden,
            qkv_projection,
            num_heads=num_heads,
            head_dim=head_dim,
        )

        key_cache = torch.cat([key_cache, new_key], dim=2)
        value_cache = torch.cat([value_cache, new_value], dim=2)

        scores, weights, context = scaled_dot_product_attention(
            new_query,
            key_cache,
            value_cache,
        )

        cache_projected_tokens += 1
        total_cache_len = key_cache.shape[2]
        cache_attention_cells += batch_size * num_heads * 1 * total_cache_len

        print(f"\nGeneration step {step} with KV Cache")
        print("new_hidden.shape = [B, 1, D] because only the newest token is projected")
        print(f"new_query.shape = {tuple(new_query.shape)} = [B, H, 1, d]")
        print(f"new_key.shape   = {tuple(new_key.shape)} = [B, H, 1, d]")
        print(f"new_value.shape = {tuple(new_value.shape)} = [B, H, 1, d]")
        print(f"key_cache.shape = {tuple(key_cache.shape)} = [B, H, T_total, d]")
        print(f"value_cache.shape = {tuple(value_cache.shape)} = [B, H, T_total, d]")
        print(f"scores.shape = {tuple(scores.shape)} = [B, H, 1, T_total]")
        print(f"context.shape = {tuple(context.shape)} = [B, H, 1, d]")
        print(
            "Meaning: the new token's Query attends to all cached Keys,\n"
            "but old tokens' K/V are reused instead of recomputed."
        )

    section("Step 05: compare rough work")
    print(
        "This demo counts two simple proxies for work:\n"
        "1. projected token count: how many token vectors go through Q/K/V projection.\n"
        "2. attention score cells: how many scalar attention scores are produced.\n"
    )
    print(f"without cache projected tokens = {no_cache_projected_tokens}")
    print(f"with cache projected tokens    = {cache_projected_tokens}")
    print(f"without cache attention cells  = {no_cache_attention_cells}")
    print(f"with cache attention cells     = {cache_attention_cells}")
    print(
        "\nConclusion:\n"
        "- KV Cache does not change model parameters.\n"
        "- KV Cache does not change the mathematical meaning of attention.\n"
        "- KV Cache stores previous K/V tensors so generation avoids recomputing old tokens.\n"
        "- Memory usage increases because cached K/V tensors must be kept for every layer."
    )


if __name__ == "__main__":
    main()
