r"""Day 5 Transformer Encoder shape demo.

Run:
    .\.venv\Scripts\python.exe experiments/day-05-transformer-encoder/encoder_shape_demo.py

macOS / Linux:
    .venv/bin/python experiments/day-05-transformer-encoder/encoder_shape_demo.py

This script intentionally does not train anything. It prints tensors and shapes
so the Transformer Encoder can be understood step by step.
"""

from __future__ import annotations

import torch
from torch import nn


def print_title(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def print_section(title: str) -> None:
    print("\n" + "-" * 88)
    print(title)
    print("-" * 88)


def show_tensor(name: str, tensor: torch.Tensor) -> None:
    print(f"{name}:")
    print(tensor)
    print(f"{name}.shape = {tuple(tensor.shape)}\n")


def main() -> None:
    torch.manual_seed(42)
    torch.set_printoptions(precision=4, sci_mode=False)

    print_title("Step 01: token ids -> token embedding + position embedding")

    # B = batch size: how many sequences are processed together.
    # T = sequence length: how many tokens each sequence has.
    # D = embedding dimension: how many numbers represent each token.
    batch_size = 2
    seq_len = 5
    vocab_size = 20
    embed_dim = 4
    max_seq_len = 8
    num_heads = 2
    head_dim = embed_dim // num_heads

    print("Symbols:")
    print(f"B = batch_size = {batch_size}")
    print(f"T = seq_len = {seq_len}")
    print(f"D = embed_dim = {embed_dim}")
    print(f"vocab_size = {vocab_size}")
    print(f"max_seq_len = {max_seq_len}")
    print(f"H = num_heads = {num_heads}")
    print(f"d = head_dim = D / H = {embed_dim} / {num_heads} = {head_dim}")

    if embed_dim % num_heads != 0:
        raise ValueError("embed_dim must be divisible by num_heads")

    # token_ids are integer ids, not meaningful numeric values.
    # Shape [B, T] means:
    #   B sequences
    #   T token ids per sequence
    token_ids = torch.tensor(
        [
            [2, 5, 7, 1, 9],
            [4, 4, 3, 8, 0],
        ],
        dtype=torch.long,
    )

    print_section("1. Input token ids")
    show_tensor("token_ids", token_ids)
    print("Meaning:")
    print("- token_ids[0] is the first sequence.")
    print("- token_ids[0, 2] is the token id at sequence 0, position 2.")
    print("- token id is only a lookup index; it is not treated as a numeric feature.")

    # nn.Embedding is a learnable lookup table:
    #   token_embedding.weight.shape = [vocab_size, embed_dim]
    token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)

    print_section("2. Token embedding table")
    show_tensor("token_embedding.weight", token_embedding.weight)
    print("Meaning:")
    print("- token_embedding.weight has one row per token id.")
    print("- token id 5 uses row token_embedding.weight[5].")
    print("- The rows are randomly initialized here; no training happens in this demo.")

    token_vectors = token_embedding(token_ids)

    print_section("3. token ids -> token vectors")
    show_tensor("token_vectors = token_embedding(token_ids)", token_vectors)
    print("Shape explanation:")
    print("[B, T] token ids become [B, T, D] token vectors.")
    print(f"[{batch_size}, {seq_len}] -> [{batch_size}, {seq_len}, {embed_dim}]")

    # Position ids describe where each token is in the sequence.
    # They are not token ids.
    position_ids = torch.arange(seq_len, dtype=torch.long)
    position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)

    print_section("4. Position ids")
    show_tensor("position_ids", position_ids)
    print("Meaning:")
    print("- position id 0 means the first position in each sequence.")
    print("- position id 1 means the second position in each sequence.")
    print("- These ids answer 'where is this token?', not 'which token is this?'.")

    # Position embedding is also a learnable lookup table:
    #   position_embedding.weight.shape = [max_seq_len, embed_dim]
    position_embedding = nn.Embedding(num_embeddings=max_seq_len, embedding_dim=embed_dim)

    print_section("5. Position embedding table")
    show_tensor("position_embedding.weight", position_embedding.weight)
    print("Meaning:")
    print("- position_embedding.weight has one row per possible position.")
    print("- position id 3 uses row position_embedding.weight[3].")
    print("- This table is also randomly initialized here and would be learned during training.")

    position_vectors = position_embedding(position_ids)

    print_section("6. position ids -> position vectors")
    show_tensor("position_vectors = position_embedding(position_ids)", position_vectors)
    print("Shape explanation:")
    print("[B, T] position ids become [B, T, D] position vectors.")
    print(f"[{batch_size}, {seq_len}] -> [{batch_size}, {seq_len}, {embed_dim}]")

    # Transformer Encoder input combines token identity and position.
    encoder_input = token_vectors + position_vectors

    print_section("7. Encoder input")
    show_tensor("encoder_input = token_vectors + position_vectors", encoder_input)
    print("Meaning:")
    print("encoder_input[b, t] = token_embedding[token_ids[b, t]] + position_embedding[t]")
    print("In plain words:")
    print("encoder_input = 'which token is this?' + 'where is this token?'")

    print_title("Step 02: encoder_input -> Q/K/V projection -> split heads")

    # One Linear layer can generate Q, K, and V at the same time.
    # Input last dimension is D.
    # Output last dimension is 3D because it contains Q, K, and V.
    qkv_projection = nn.Linear(in_features=embed_dim, out_features=3 * embed_dim)


    print_section("8. Q/K/V projection layer")
    show_tensor("qkv_projection.weight", qkv_projection.weight)
    show_tensor("qkv_projection.bias", qkv_projection.bias)
    print("Meaning:")
    print("- This Linear layer maps each token vector from D dimensions to 3D dimensions.")
    print("- The 3D output contains Q, K, and V packed together.")
    print("- These parameters are randomly initialized here; no training happens in this demo.")

    qkv_packed = qkv_projection(encoder_input)

    print_section("9. encoder_input -> packed Q/K/V")
    show_tensor("qkv_packed = qkv_projection(encoder_input)", qkv_packed)
    print("Shape explanation:")
    print("[B, T, D] becomes [B, T, 3D].")
    print(
        f"[{batch_size}, {seq_len}, {embed_dim}] -> "
        f"[{batch_size}, {seq_len}, {3 * embed_dim}]"
    )
    print("Content explanation:")
    print("- qkv_packed[b, t] is one token's packed Q/K/V vector.")
    print("- Its length is 3D.")
    print("- The first D numbers are Q.")
    print("- The next D numbers are K.")
    print("- The last D numbers are V.")

    first_token_qkv = qkv_packed[0, 0]
    first_token_q = first_token_qkv[:embed_dim]
    first_token_k = first_token_qkv[embed_dim : 2 * embed_dim]
    first_token_v = first_token_qkv[2 * embed_dim :]

    print_section("9.1 Inspect one token inside packed Q/K/V")
    show_tensor("qkv_packed[0, 0]", first_token_qkv)
    show_tensor("Q part = qkv_packed[0, 0, 0:D]", first_token_q)
    show_tensor("K part = qkv_packed[0, 0, D:2D]", first_token_k)
    show_tensor("V part = qkv_packed[0, 0, 2D:3D]", first_token_v)
    print("Meaning:")
    print("- qkv_packed[0, 0] belongs to batch item 0, token position 0.")
    print("- In this demo D = 4, so qkv_packed[0, 0] has 12 numbers.")
    print("- numbers 0..3 are Q, numbers 4..7 are K, numbers 8..11 are V.")

    # Reshape the packed Q/K/V tensor so the Q/K/V axis and head axis are explicit.
    # [B, T, 3D] -> [B, T, 3, H, d]
    qkv_reshaped = qkv_packed.view(batch_size, seq_len, 3, num_heads, head_dim)

    print_section("10. Reshape packed Q/K/V")
    show_tensor("qkv_reshaped", qkv_reshaped)
    print("Shape explanation:")
    print("[B, T, 3D] becomes [B, T, 3, H, d].")
    print("The dimension with size 3 means: Q, K, V.")
    print("The dimension H means: number of heads.")
    print("The dimension d means: head_dim.")
    print("For the same token qkv_packed[0, 0], reshape turns:")
    print("[12] -> [3, 2, 2]")
    print("That means:")
    print("- 3 groups: Q, K, V")
    print("- each group has 2 heads")
    print("- each head has 2 numbers")

    print_section("10.1 Same token after reshape")
    show_tensor("qkv_reshaped[0, 0]", qkv_reshaped[0, 0])
    show_tensor("Q for token [0,0], split into heads", qkv_reshaped[0, 0, 0])
    show_tensor("K for token [0,0], split into heads", qkv_reshaped[0, 0, 1])
    show_tensor("V for token [0,0], split into heads", qkv_reshaped[0, 0, 2])

    # Move the Q/K/V dimension to the front and put heads before tokens.
    # [B, T, 3, H, d] -> [3, B, H, T, d]
    qkv_for_attention = qkv_reshaped.permute(2, 0, 3, 1, 4)

    print_section("11. Permute for attention")
    show_tensor("qkv_for_attention", qkv_for_attention)
    print("Shape explanation:")
    print("[B, T, 3, H, d] becomes [3, B, H, T, d].")
    print("Now qkv_for_attention[0] is Q, [1] is K, [2] is V.")

    query, key, value = qkv_for_attention.unbind(dim=0)

    print_section("12. Split into Q, K, V")
    show_tensor("query", query)
    show_tensor("key", key)
    show_tensor("value", value)
    print("Shape explanation:")
    print("Q, K, V all have shape [B, H, T, d].")
    print(
        f"In this demo: [{batch_size}, {num_heads}, {seq_len}, {head_dim}]"
    )
    print("Meaning:")
    print("- B: which sequence in the batch.")
    print("- H: which attention head.")
    print("- T: which token position.")
    print("- d: the vector dimension inside one head.")

    print_title("Step 02 complete")
    print("Current state:")
    print(f"encoder_input.shape = [{batch_size}, {seq_len}, {embed_dim}] = [B, T, D]")
    print(f"query.shape = key.shape = value.shape = [{batch_size}, {num_heads}, {seq_len}, {head_dim}] = [B, H, T, d]")
    print("\nNext step will be attention scores: Q @ K^T, still without training.")


if __name__ == "__main__":
    main()
