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
    ffn_hidden_dim = 4 * embed_dim

    print("Symbols:")
    print(f"B = batch_size = {batch_size}")
    print(f"T = seq_len = {seq_len}")
    print(f"D = embed_dim = {embed_dim}")
    print(f"vocab_size = {vocab_size}")
    print(f"max_seq_len = {max_seq_len}")
    print(f"H = num_heads = {num_heads}")
    print(f"d = head_dim = D / H = {embed_dim} / {num_heads} = {head_dim}")
    print(f"FFN hidden dimension = 4 * D = {ffn_hidden_dim}")

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

    print_title("Step 03: attention scores = Q @ K^T")

    # Attention scores compare every token's Q with every token's K inside
    # each batch item and each head.
    #
    # query.shape = [B, H, T, d]
    # key.shape   = [B, H, T, d]
    #
    # To calculate [T, d] @ [d, T] for each [B, H] group, K must be
    # transposed on its last two dimensions:
    #
    # key_transposed.shape = [B, H, d, T]
    key_transposed = key.transpose(-2, -1)

    print_section("13. Transpose K for Q @ K^T")
    show_tensor("key", key)
    show_tensor("key_transposed = key.transpose(-2, -1)", key_transposed)
    print("Shape explanation:")
    print("[B, H, T, d] becomes [B, H, d, T].")
    print(
        f"[{batch_size}, {num_heads}, {seq_len}, {head_dim}] -> "
        f"[{batch_size}, {num_heads}, {head_dim}, {seq_len}]"
    )
    print("Meaning:")
    print("- transpose(-2, -1) swaps only the last two dimensions: T and d.")
    print("- It does not mix different batch items.")
    print("- It does not mix different heads.")
    print("- It prepares K so each head can calculate [T, d] @ [d, T].")

    print_section("13.1 Inspect one head before and after K transpose")
    show_tensor("key[0, 0]", key[0, 0])
    show_tensor("key_transposed[0, 0]", key_transposed[0, 0])
    print("Meaning:")
    print("- key[0, 0] contains all 5 token key vectors in batch 0, head 0.")
    print("- Before transpose, each row is one token key vector: [T, d] = [5, 2].")
    print("- After transpose, each column corresponds to one token key vector: [d, T] = [2, 5].")

    attention_scores = query @ key_transposed

    print_section("14. Raw attention scores")
    show_tensor("attention_scores = query @ key_transposed", attention_scores)
    print("Shape explanation:")
    print("[B, H, T, d] @ [B, H, d, T] becomes [B, H, T, T].")
    print(
        f"[{batch_size}, {num_heads}, {seq_len}, {head_dim}] @ "
        f"[{batch_size}, {num_heads}, {head_dim}, {seq_len}] -> "
        f"[{batch_size}, {num_heads}, {seq_len}, {seq_len}]"
    )
    print("Meaning:")
    print("- attention_scores[b, h] is one token-by-token score table.")
    print("- Rows are tokens being updated.")
    print("- Columns are tokens being looked at.")
    print("- These are raw scores, not probabilities.")

    print_section("14.1 Inspect one score table: attention_scores[0, 0]")
    show_tensor("query[0, 0]", query[0, 0])
    show_tensor("key_transposed[0, 0]", key_transposed[0, 0])
    show_tensor("attention_scores[0, 0]", attention_scores[0, 0])
    show_tensor("attention_scores[0, 0, 0]", attention_scores[0, 0, 0])
    print("Meaning:")
    print("- attention_scores[0, 0] is the score table for batch 0, head 0.")
    print("- attention_scores[0, 0, 0] is row 0 of that table.")
    print("- Row 0 means token 0 looking at token 0, 1, 2, 3, 4.")

    q_token0 = query[0, 0, 0]
    k_token3 = key[0, 0, 3]
    score_token0_to_token3 = attention_scores[0, 0, 0, 3]

    print_section("14.2 One cell: attention_scores[0, 0, 0, 3]")
    show_tensor("query[0, 0, 0]", q_token0)
    show_tensor("key[0, 0, 3]", k_token3)
    show_tensor("attention_scores[0, 0, 0, 3]", score_token0_to_token3)
    print("Manual calculation:")
    print(
        "attention_scores[0, 0, 0, 3] = "
        "query[0, 0, 0] dot key[0, 0, 3]"
    )
    print(
        f"= ({q_token0[0].item():.4f} * {k_token3[0].item():.4f}) "
        f"+ ({q_token0[1].item():.4f} * {k_token3[1].item():.4f})"
    )
    print(f"= {score_token0_to_token3.item():.4f}")
    print("Meaning:")
    print("- This one number is token 0's raw score for looking at token 3.")
    print("- It is inside batch 0 and head 0.")
    print("- It is a Q dot K score, not a Q dot V score.")

    scale = head_dim ** 0.5
    scaled_attention_scores = attention_scores / scale

    print_section("15. Scale attention scores by sqrt(d)")
    print(f"d = head_dim = {head_dim}")
    print(f"sqrt(d) = sqrt({head_dim}) = {scale:.4f}")
    show_tensor("scaled_attention_scores = attention_scores / sqrt(d)", scaled_attention_scores)
    show_tensor("scaled_attention_scores[0, 0]", scaled_attention_scores[0, 0])
    show_tensor("scaled_attention_scores[0, 0, 0]", scaled_attention_scores[0, 0, 0])
    print("Meaning:")
    print("- Scaling keeps Q dot K scores from becoming too large when d grows.")
    print("- The relative order of scores stays similar, but the values are milder.")
    print("- These scaled scores are still not probabilities.")
    print("- The next step will be softmax over the last dimension.")

    print_title("Step 04: attention weights = softmax(scaled scores)")

    # Softmax turns each row of scaled scores into positive weights that sum to 1.
    #
    # scaled_attention_scores.shape = [B, H, T, T]
    #
    # dim=-1 means softmax is applied over the last T dimension: the tokens being
    # looked at. For each fixed [b, h, t], it normalizes the scores over all j.
    attention_weights = torch.softmax(scaled_attention_scores, dim=-1)

    print_section("16. Softmax over the last dimension")
    show_tensor(
        "attention_weights = torch.softmax(scaled_attention_scores, dim=-1)",
        attention_weights,
    )
    print("Shape explanation:")
    print("softmax keeps the same shape.")
    print("[B, H, T, T] stays [B, H, T, T].")
    print(
        f"[{batch_size}, {num_heads}, {seq_len}, {seq_len}] -> "
        f"[{batch_size}, {num_heads}, {seq_len}, {seq_len}]"
    )
    print("Meaning:")
    print("- dim=-1 means softmax is applied across the last dimension.")
    print("- The last dimension is the token being looked at.")
    print("- For each fixed [b, h, t], the weights across all looked-at tokens sum to 1.")

    print_section("16.1 Compare one row before and after softmax")
    show_tensor("scaled_attention_scores[0, 0, 0]", scaled_attention_scores[0, 0, 0])
    show_tensor("attention_weights[0, 0, 0]", attention_weights[0, 0, 0])
    show_tensor(
        "attention_weights[0, 0, 0].sum()",
        attention_weights[0, 0, 0].sum(),
    )
    print("Meaning:")
    print("- This row belongs to batch 0, head 0, token 0.")
    print("- It says how much token 0 looks at token 0, 1, 2, 3, 4.")
    print("- Before softmax: raw scaled scores, can be negative and do not sum to 1.")
    print("- After softmax: positive weights, and this row sums to 1.")

    print_section("16.2 Inspect one full attention weight table")
    show_tensor("attention_weights[0, 0]", attention_weights[0, 0])
    show_tensor("attention_weights[0, 0].sum(dim=-1)", attention_weights[0, 0].sum(dim=-1))
    print("Meaning:")
    print("- attention_weights[0, 0] is the weight table for batch 0, head 0.")
    print("- Every row corresponds to one token being updated.")
    print("- sum(dim=-1) checks each row; every row should sum to 1.")

    print_title("Step 05: context = attention_weights @ value")

    # Attention weights decide how much each token should look at every token.
    # Value provides the actual content vectors to be mixed.
    #
    # attention_weights.shape = [B, H, T, T]
    # value.shape             = [B, H, T, d]
    #
    # For each [B, H] group:
    #   [T, T] @ [T, d] = [T, d]
    context = attention_weights @ value

    print_section("17. Weighted sum of V")
    show_tensor("value", value)
    show_tensor("context = attention_weights @ value", context)
    print("Shape explanation:")
    print("[B, H, T, T] @ [B, H, T, d] becomes [B, H, T, d].")
    print(
        f"[{batch_size}, {num_heads}, {seq_len}, {seq_len}] @ "
        f"[{batch_size}, {num_heads}, {seq_len}, {head_dim}] -> "
        f"[{batch_size}, {num_heads}, {seq_len}, {head_dim}]"
    )
    print("Meaning:")
    print("- attention_weights tells each token how much to look at every token.")
    print("- value provides the content vectors being mixed.")
    print("- context[b, h, t] is the new vector for token t inside batch b and head h.")

    print_section("17.1 One token context vector: context[0, 0, 0]")
    show_tensor("attention_weights[0, 0, 0]", attention_weights[0, 0, 0])
    show_tensor("value[0, 0]", value[0, 0])
    weighted_value_terms = attention_weights[0, 0, 0].unsqueeze(-1) * value[0, 0]
    manual_context_token0 = weighted_value_terms.sum(dim=0)
    show_tensor(
        "weighted terms = attention_weights[0, 0, 0].unsqueeze(-1) * value[0, 0]",
        weighted_value_terms,
    )
    show_tensor("manual sum of weighted terms", manual_context_token0)
    show_tensor("context[0, 0, 0]", context[0, 0, 0])
    print("Meaning:")
    print("- This is token 0 in batch 0, head 0.")
    print("- Its attention row has 5 weights, one for each token.")
    print("- Each weight multiplies the corresponding V vector.")
    print("- The weighted V vectors are summed into one d-dimensional context vector.")

    print_section("17.2 Inspect one full context table")
    show_tensor("context[0, 0]", context[0, 0])
    print("Meaning:")
    print("- context[0, 0] contains the new vectors for all 5 tokens in batch 0, head 0.")
    print("- Each row is one token after it has mixed V using its attention weights.")

    print_title("Step 06: merge heads [B, H, T, d] -> [B, T, D]")

    # context is currently grouped by head:
    #   context.shape = [B, H, T, d]
    #
    # For the next Transformer step, each token should get one full D-dimensional
    # vector again. So we first move T before H:
    #   [B, H, T, d] -> [B, T, H, d]
    #
    # Then we flatten H and d together:
    #   [B, T, H, d] -> [B, T, H * d] = [B, T, D]
    context_for_merge = context.permute(0, 2, 1, 3)
    context_merged = context_for_merge.contiguous().view(batch_size, seq_len, embed_dim)

    print_section("18. Move token dimension before head dimension")
    show_tensor("context", context)
    show_tensor("context_for_merge = context.permute(0, 2, 1, 3)", context_for_merge)
    print("Shape explanation:")
    print("[B, H, T, d] becomes [B, T, H, d].")
    print(
        f"[{batch_size}, {num_heads}, {seq_len}, {head_dim}] -> "
        f"[{batch_size}, {seq_len}, {num_heads}, {head_dim}]"
    )
    print("Meaning:")
    print("- Before permute, heads are before tokens: [B, H, T, d].")
    print("- After permute, tokens are before heads: [B, T, H, d].")
    print("- Now context_for_merge[b, t] contains all heads for token t.")

    print_section("18.1 One token before merge")
    show_tensor("context[0, :, 0, :]", context[0, :, 0, :])
    show_tensor("context_for_merge[0, 0]", context_for_merge[0, 0])
    print("Meaning:")
    print("- context[0, :, 0, :] selects batch 0, all heads, token 0, all head dimensions.")
    print("- context_for_merge[0, 0] is the same data arranged as token 0's head table.")
    print("- Its shape is [H, d], so each row is one head's context vector for token 0.")

    print_section("19. Flatten H and d back to D")
    show_tensor(
        "context_merged = context_for_merge.contiguous().view(B, T, D)",
        context_merged,
    )
    print("Shape explanation:")
    print("[B, T, H, d] becomes [B, T, H * d].")
    print(
        f"[{batch_size}, {seq_len}, {num_heads}, {head_dim}] -> "
        f"[{batch_size}, {seq_len}, {num_heads * head_dim}]"
    )
    print("Because H * d = D:")
    print(f"{num_heads} * {head_dim} = {embed_dim}")
    print("Meaning:")
    print("- context_merged[b, t] is one full D-dimensional vector for token t.")
    print("- It is made by concatenating the small vectors from all heads.")

    print_section("19.1 One token after merge")
    head0_token0 = context_for_merge[0, 0, 0]
    head1_token0 = context_for_merge[0, 0, 1]
    manual_merged_token0 = torch.cat([head0_token0, head1_token0], dim=0)
    show_tensor("head 0 vector for token 0", head0_token0)
    show_tensor("head 1 vector for token 0", head1_token0)
    show_tensor("manual concat of head 0 and head 1", manual_merged_token0)
    show_tensor("context_merged[0, 0]", context_merged[0, 0])
    print("Meaning:")
    print("- Each head gives token 0 a d-dimensional vector.")
    print("- Merge heads means putting those vectors next to each other.")
    print("- In this demo: [2] from head 0 + [2] from head 1 = [4].")

    print_title("Step 07: output projection [B, T, D] -> [B, T, D]")

    # After heads are concatenated, Transformer attention usually applies one
    # more Linear layer. It keeps the same outer shape [B, T, D], but it lets the
    # model mix information across the concatenated head dimensions.
    output_projection = nn.Linear(in_features=embed_dim, out_features=embed_dim)
    attention_output = output_projection(context_merged)

    print_section("20. Output projection layer")
    show_tensor("output_projection.weight", output_projection.weight)
    show_tensor("output_projection.bias", output_projection.bias)
    print("Meaning:")
    print("- This Linear layer is often called W_O in Transformer explanations.")
    print("- It maps each merged token vector from D dimensions back to D dimensions.")
    print("- The shape stays [B, T, D], but the values are linearly mixed.")
    print("- These parameters are randomly initialized here; no training happens in this demo.")

    print_section("21. Apply output projection")
    show_tensor("context_merged", context_merged)
    show_tensor("attention_output = output_projection(context_merged)", attention_output)
    print("Shape explanation:")
    print("[B, T, D] becomes [B, T, D].")
    print(
        f"[{batch_size}, {seq_len}, {embed_dim}] -> "
        f"[{batch_size}, {seq_len}, {embed_dim}]"
    )
    print("Meaning:")
    print("- Every token vector is processed independently by the same Linear layer.")
    print("- Batch dimension B is not mixed.")
    print("- Token dimension T is not mixed by this Linear layer.")
    print("- Only the D numbers inside each token vector are mixed.")

    print_section("21.1 One token through output projection")
    token0_before_output = context_merged[0, 0]
    first_output_weight_row = output_projection.weight[0]
    first_output_bias = output_projection.bias[0]
    manual_first_output_number = (
        token0_before_output * first_output_weight_row
    ).sum() + first_output_bias
    manual_token0_after_output = (
        token0_before_output @ output_projection.weight.T
    ) + output_projection.bias
    show_tensor(
        "token0 before output projection = context_merged[0, 0]",
        token0_before_output,
    )
    show_tensor(
        "weight row for output dim 0 = output_projection.weight[0]",
        first_output_weight_row,
    )
    show_tensor("bias for output dim 0 = output_projection.bias[0]", first_output_bias)
    show_tensor("manual first output number", manual_first_output_number)
    show_tensor("manual full output token", manual_token0_after_output)
    show_tensor("attention_output[0, 0]", attention_output[0, 0])
    print("Manual calculation for the first output number:")
    print("attention_output[0, 0, 0] = context_merged[0, 0] dot weight[0] + bias[0]")
    print(
        f"= ({token0_before_output[0].item():.4f} * {first_output_weight_row[0].item():.4f}) "
        f"+ ({token0_before_output[1].item():.4f} * {first_output_weight_row[1].item():.4f}) "
        f"+ ({token0_before_output[2].item():.4f} * {first_output_weight_row[2].item():.4f}) "
        f"+ ({token0_before_output[3].item():.4f} * {first_output_weight_row[3].item():.4f}) "
        f"+ {first_output_bias.item():.4f}"
    )
    print(f"= {manual_first_output_number.item():.4f}")
    print("Meaning:")
    print("- The merged token vector has D numbers.")
    print("- Output projection computes each new output dimension from all D input numbers.")
    print("- This is how information from different heads can be mixed after concatenation.")

    print_title("Step 08: first residual connection")

    # Residual connection keeps the original encoder input and adds the attention
    # sublayer output to it. The two tensors must have the same shape.
    attention_residual = encoder_input + attention_output

    print_section("22. Add original input back to attention output")
    show_tensor("encoder_input", encoder_input)
    show_tensor("attention_output", attention_output)
    show_tensor("attention_residual = encoder_input + attention_output", attention_residual)
    print("Shape explanation:")
    print("[B, T, D] + [B, T, D] becomes [B, T, D].")
    print(
        f"[{batch_size}, {seq_len}, {embed_dim}] + "
        f"[{batch_size}, {seq_len}, {embed_dim}] -> "
        f"[{batch_size}, {seq_len}, {embed_dim}]"
    )
    print("Meaning:")
    print("- attention_output is what the attention sublayer newly computed.")
    print("- encoder_input is the original token representation before attention.")
    print("- The residual connection keeps the original path and adds the new attention information.")

    print_section("22.1 One token residual addition")
    manual_attention_residual_token0 = encoder_input[0, 0] + attention_output[0, 0]
    show_tensor("encoder_input[0, 0]", encoder_input[0, 0])
    show_tensor("attention_output[0, 0]", attention_output[0, 0])
    show_tensor("manual encoder_input[0, 0] + attention_output[0, 0]", manual_attention_residual_token0)
    show_tensor("attention_residual[0, 0]", attention_residual[0, 0])
    print("Meaning:")
    print("- Residual addition is element-wise.")
    print("- The first number adds to the first number, the second to the second, and so on.")

    print_title("Step 09: first LayerNorm over D")

    # LayerNorm normalizes each token vector over its last dimension D.
    # It does not normalize across batch items or across token positions.
    attention_layer_norm = nn.LayerNorm(normalized_shape=embed_dim)
    attention_normed = attention_layer_norm(attention_residual)

    print_section("23. First LayerNorm layer")
    show_tensor("attention_layer_norm.weight", attention_layer_norm.weight)
    show_tensor("attention_layer_norm.bias", attention_layer_norm.bias)
    print("Meaning:")
    print("- LayerNorm has one learnable scale and one learnable bias for each D dimension.")
    print("- In this demo D = 4, so weight.shape and bias.shape are both [4].")
    print("- The default initial weight is all 1 and bias is all 0.")

    print_section("24. Apply first LayerNorm")
    show_tensor("attention_normed = attention_layer_norm(attention_residual)", attention_normed)
    print("Shape explanation:")
    print("LayerNorm keeps the same shape.")
    print(
        f"[{batch_size}, {seq_len}, {embed_dim}] -> "
        f"[{batch_size}, {seq_len}, {embed_dim}]"
    )
    print("Meaning:")
    print("- For each token vector, LayerNorm computes mean and variance across D only.")
    print("- It does not mix different tokens.")
    print("- It does not mix different batch items.")

    print_section("24.1 One token LayerNorm calculation")
    token0_before_norm = attention_residual[0, 0]
    token0_mean = token0_before_norm.mean(dim=-1)
    token0_variance = ((token0_before_norm - token0_mean) ** 2).mean(dim=-1)
    token0_normalized_manual = (token0_before_norm - token0_mean) / torch.sqrt(
        token0_variance + attention_layer_norm.eps
    )
    show_tensor("attention_residual[0, 0]", token0_before_norm)
    show_tensor("mean of attention_residual[0, 0]", token0_mean)
    show_tensor("variance of attention_residual[0, 0]", token0_variance)
    show_tensor("manual normalized token", token0_normalized_manual)
    show_tensor("attention_normed[0, 0]", attention_normed[0, 0])
    print("Formula:")
    print("normalized = (x - mean(x)) / sqrt(variance(x) + eps)")
    print("Meaning:")
    print("- x is one token's D-dimensional vector.")
    print("- mean and variance are calculated only inside this one token vector.")
    print("- eps is a tiny number that avoids division by zero.")

    print_title("Step 10: Feed Forward Network [B, T, D] -> [B, T, D]")

    # Position-wise FFN:
    #   Linear1: D -> hidden dimension
    #   ReLU: non-linear activation
    #   Linear2: hidden dimension -> D
    #
    # It is applied independently to every token position.
    ffn_linear1 = nn.Linear(in_features=embed_dim, out_features=ffn_hidden_dim)
    ffn_activation = nn.ReLU()
    ffn_linear2 = nn.Linear(in_features=ffn_hidden_dim, out_features=embed_dim)

    ffn_hidden = ffn_linear1(attention_normed)
    ffn_activated = ffn_activation(ffn_hidden)
    ffn_output = ffn_linear2(ffn_activated)

    print_section("25. FFN layers")
    show_tensor("ffn_linear1.weight", ffn_linear1.weight)
    show_tensor("ffn_linear1.bias", ffn_linear1.bias)
    show_tensor("ffn_linear2.weight", ffn_linear2.weight)
    show_tensor("ffn_linear2.bias", ffn_linear2.bias)
    print("Meaning:")
    print("- FFN means Feed Forward Network.")
    print("- ffn_linear1 expands each token from D to 4D.")
    print("- ReLU keeps positive numbers and turns negative numbers into 0.")
    print("- ffn_linear2 compresses each token from 4D back to D.")

    print_section("26. Apply FFN")
    show_tensor("attention_normed", attention_normed)
    show_tensor("ffn_hidden = ffn_linear1(attention_normed)", ffn_hidden)
    show_tensor("ffn_activated = ReLU(ffn_hidden)", ffn_activated)
    show_tensor("ffn_output = ffn_linear2(ffn_activated)", ffn_output)
    print("Shape explanation:")
    print("[B, T, D] -> [B, T, 4D] -> [B, T, 4D] -> [B, T, D].")
    print(
        f"[{batch_size}, {seq_len}, {embed_dim}] -> "
        f"[{batch_size}, {seq_len}, {ffn_hidden_dim}] -> "
        f"[{batch_size}, {seq_len}, {ffn_hidden_dim}] -> "
        f"[{batch_size}, {seq_len}, {embed_dim}]"
    )
    print("Meaning:")
    print("- FFN processes each token independently.")
    print("- It does not calculate token-to-token attention.")
    print("- Token-to-token information exchange has already happened in self-attention.")

    print_section("26.1 One token through FFN")
    show_tensor("attention_normed[0, 0]", attention_normed[0, 0])
    show_tensor("ffn_hidden[0, 0]", ffn_hidden[0, 0])
    show_tensor("ffn_activated[0, 0]", ffn_activated[0, 0])
    show_tensor("ffn_output[0, 0]", ffn_output[0, 0])
    print("Meaning:")
    print("- This is one token going through D -> 4D -> ReLU -> D.")
    print("- ReLU is the non-linear part of this FFN.")

    print_title("Step 11: second residual connection + second LayerNorm")

    # The FFN sublayer also uses residual connection and LayerNorm.
    ffn_residual = attention_normed + ffn_output
    ffn_layer_norm = nn.LayerNorm(normalized_shape=embed_dim)
    encoder_output = ffn_layer_norm(ffn_residual)

    print_section("27. Add FFN output back to attention_normed")
    show_tensor("attention_normed", attention_normed)
    show_tensor("ffn_output", ffn_output)
    show_tensor("ffn_residual = attention_normed + ffn_output", ffn_residual)
    print("Shape explanation:")
    print("[B, T, D] + [B, T, D] becomes [B, T, D].")
    print("Meaning:")
    print("- This is the second residual connection inside the Encoder block.")
    print("- It keeps the path from attention_normed and adds the FFN transformation.")

    print_section("28. Apply second LayerNorm")
    show_tensor("ffn_layer_norm.weight", ffn_layer_norm.weight)
    show_tensor("ffn_layer_norm.bias", ffn_layer_norm.bias)
    show_tensor("encoder_output = ffn_layer_norm(ffn_residual)", encoder_output)
    print("Shape explanation:")
    print("The final Encoder block output stays [B, T, D].")
    print(
        f"[{batch_size}, {seq_len}, {embed_dim}] -> "
        f"[{batch_size}, {seq_len}, {embed_dim}]"
    )
    print("Meaning:")
    print("- encoder_output can be passed into the next Encoder block.")
    print("- Because input and output are both [B, T, D], many Encoder blocks can be stacked.")

    print_section("28.1 One token final output")
    show_tensor("ffn_residual[0, 0]", ffn_residual[0, 0])
    show_tensor("encoder_output[0, 0]", encoder_output[0, 0])

    print_title("Step 11 complete")
    print("Current state:")
    print(f"encoder_input.shape = [{batch_size}, {seq_len}, {embed_dim}] = [B, T, D]")
    print(f"query.shape = key.shape = value.shape = [{batch_size}, {num_heads}, {seq_len}, {head_dim}] = [B, H, T, d]")
    print(f"attention_scores.shape = [{batch_size}, {num_heads}, {seq_len}, {seq_len}] = [B, H, T, T]")
    print(f"scaled_attention_scores.shape = [{batch_size}, {num_heads}, {seq_len}, {seq_len}] = [B, H, T, T]")
    print(f"attention_weights.shape = [{batch_size}, {num_heads}, {seq_len}, {seq_len}] = [B, H, T, T]")
    print(f"context.shape = [{batch_size}, {num_heads}, {seq_len}, {head_dim}] = [B, H, T, d]")
    print(f"context_merged.shape = [{batch_size}, {seq_len}, {embed_dim}] = [B, T, D]")
    print(f"attention_output.shape = [{batch_size}, {seq_len}, {embed_dim}] = [B, T, D]")
    print(f"attention_residual.shape = [{batch_size}, {seq_len}, {embed_dim}] = [B, T, D]")
    print(f"attention_normed.shape = [{batch_size}, {seq_len}, {embed_dim}] = [B, T, D]")
    print(f"ffn_hidden.shape = [{batch_size}, {seq_len}, {ffn_hidden_dim}] = [B, T, 4D]")
    print(f"ffn_output.shape = [{batch_size}, {seq_len}, {embed_dim}] = [B, T, D]")
    print(f"ffn_residual.shape = [{batch_size}, {seq_len}, {embed_dim}] = [B, T, D]")
    print(f"encoder_output.shape = [{batch_size}, {seq_len}, {embed_dim}] = [B, T, D]")
    print("\nNext step can be wrapping these operations into a reusable EncoderBlock class.")


if __name__ == "__main__":
    main()
