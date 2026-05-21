from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


torch.manual_seed(42)
torch.set_printoptions(precision=4, sci_mode=False)


@dataclass(frozen=True)
class ToyTokenizer:
    """A tiny tokenizer for understanding the LLM inference data flow."""

    token_to_id: dict[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "id_to_token",
            {token_id: token for token, token_id in self.token_to_id.items()},
        )
        normal_tokens = [
            token
            for token in self.token_to_id
            if token not in {"<pad>", "<unk>"}
        ]
        object.__setattr__(
            self,
            "tokens_by_length",
            sorted(normal_tokens, key=len, reverse=True),
        )

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str) -> tuple[list[str], list[int]]:
        """Use longest-match tokenization so "深度学习" stays one token."""

        tokens: list[str] = []
        cursor = 0
        while cursor < len(text):
            matched_token = None
            for token in self.tokens_by_length:
                if text.startswith(token, cursor):
                    matched_token = token
                    break

            if matched_token is None:
                matched_token = "<unk>"
                cursor += 1
            else:
                cursor += len(matched_token)

            tokens.append(matched_token)

        input_ids = [self.token_to_id.get(token, self.token_to_id["<unk>"]) for token in tokens]
        return tokens, input_ids

    def decode(self, token_ids: list[int]) -> str:
        tokens = [self.id_to_token.get(token_id, "<unk>") for token_id in token_ids]
        return "".join(token for token in tokens if token not in {"<pad>", "<unk>"})


def line() -> None:
    print("-" * 88)


def section(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def show_tensor(name: str, tensor: torch.Tensor, note: str | None = None) -> None:
    line()
    print(name)
    print(tensor)
    print(f"{name}.shape = {tuple(tensor.shape)}")
    if note:
        print(note)


def show_token_table(tokenizer: ToyTokenizer) -> None:
    print("Toy vocab:")
    for token, token_id in sorted(tokenizer.token_to_id.items(), key=lambda item: item[1]):
        print(f"  id {token_id:>2} -> {token}")


def main() -> None:
    tokenizer = ToyTokenizer(
        token_to_id={
            "<pad>": 0,
            "<unk>": 1,
            "我": 2,
            "喜欢": 3,
            "深度学习": 4,
            "AI": 5,
            "。": 6,
            "电影": 7,
            "很": 8,
            "有趣": 9,
        }
    )

    prompt = "我喜欢"
    max_seq_len = 8
    embed_dim = 6

    section("Step 01: prompt text -> tokens -> input_ids")
    show_token_table(tokenizer)
    print(f"\nprompt = {prompt!r}")

    tokens, input_ids_list = tokenizer.encode(prompt)
    print(f"tokens = {tokens}")
    print(f"input_ids_list = {input_ids_list}")
    print(
        "\nMeaning:\n"
        "- token 是 tokenizer 切出来的文本单位。\n"
        "- token id 是 token 在词表里的整数编号。\n"
        "- input_ids 还不是向量，只是查 embedding 表用的索引。"
    )

    input_ids = torch.tensor([input_ids_list], dtype=torch.long)
    batch_size, seq_len = input_ids.shape
    vocab_size = tokenizer.vocab_size

    show_tensor(
        "input_ids",
        input_ids,
        note=(
            f"B = batch_size = {batch_size}\n"
            f"T = seq_len = {seq_len}\n"
            "[B, T] means: batch dimension first, token-position dimension second."
        ),
    )

    section("Step 02: input_ids -> token vectors + position vectors")
    print(
        "Symbols:\n"
        f"V = vocab_size = {vocab_size}\n"
        f"C = embed_dim = {embed_dim}\n"
        f"max_seq_len = {max_seq_len}"
    )

    token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
    position_embedding = nn.Embedding(num_embeddings=max_seq_len, embedding_dim=embed_dim)

    show_tensor(
        "token_embedding.weight",
        token_embedding.weight,
        note="[V, C]: one row per token id, one vector per token.",
    )

    token_vectors = token_embedding(input_ids)
    show_tensor(
        "token_vectors = token_embedding(input_ids)",
        token_vectors,
        note="[B, T] -> [B, T, C]. Each integer id has become a C-dimensional vector.",
    )

    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    show_tensor(
        "position_ids",
        position_ids,
        note="position_ids say where each token is in the sequence: 0, 1, 2, ...",
    )

    position_vectors = position_embedding(position_ids)
    show_tensor(
        "position_vectors = position_embedding(position_ids)",
        position_vectors,
        note="[B, T] -> [B, T, C]. Each position id also becomes a vector.",
    )

    model_input = token_vectors + position_vectors
    show_tensor(
        "model_input = token_vectors + position_vectors",
        model_input,
        note=(
            "Each token position now contains both:\n"
            "- which token it is\n"
            "- where it is in the sequence"
        ),
    )

    section("Step 03: model_input -> hidden_states")
    hidden_states = model_input
    show_tensor(
        "hidden_states",
        hidden_states,
        note=(
            "This demo skips real causal Transformer blocks for now.\n"
            "In a real LLM, hidden_states would be produced by many causal Transformer layers.\n"
            "We keep hidden_states = model_input only to isolate the logits path."
        ),
    )

    section("Step 04: hidden_states -> logits over the vocabulary")
    lm_head = nn.Linear(in_features=embed_dim, out_features=vocab_size, bias=False)
    print(
        "lm_head = nn.Linear(in_features=C, out_features=V, bias=False)\n"
        "Meaning:\n"
        "- C is the hidden vector dimension of one token position.\n"
        "- V is the vocabulary size.\n"
        "- lm_head turns each C-dimensional hidden vector into V raw scores.\n"
        "- One output score is calculated by a dot product:\n"
        "  logits[b, t, i] = sum(hidden_states[b, t, k] * lm_head.weight[i, k])\n"
        "  where i is a candidate token id and k is a hidden-vector dimension index."
    )

    # Make this demo readable: manually bias the last position toward "深度学习".
    # A real language model learns lm_head through training; we are not training here.
    with torch.no_grad():
        lm_head.weight.normal_(mean=0.0, std=0.03)
        last_hidden = hidden_states[0, -1]
        last_direction = last_hidden / last_hidden.norm().clamp_min(1e-6)
        lm_head.weight[tokenizer.token_to_id["深度学习"]] = 2.0 * last_direction
        lm_head.weight[tokenizer.token_to_id["电影"]] = 0.8 * last_direction
        lm_head.weight[tokenizer.token_to_id["。"]] = -0.4 * last_direction

    show_tensor(
        "lm_head.weight",
        lm_head.weight,
        note=(
            "[V, C]: one output row per vocab token.\n"
            "Row 4 corresponds to token id 4, which is '深度学习' in this toy vocab."
        ),
    )

    logits = lm_head(hidden_states)
    show_tensor(
        "logits = lm_head(hidden_states)",
        logits,
        note=(
            "[B, T, C] -> [B, T, V].\n"
            "For every token position, the model gives one score for every vocab token."
        ),
    )

    section("Step 04.1: manually calculate one logit")
    candidate_token = "深度学习"
    candidate_token_id = tokenizer.token_to_id[candidate_token]
    last_position_hidden = hidden_states[0, -1, :]
    candidate_output_weight = lm_head.weight[candidate_token_id]
    elementwise_product = last_position_hidden * candidate_output_weight
    manual_logit = elementwise_product.sum()
    pytorch_logit = logits[0, -1, candidate_token_id]

    print(
        "We now calculate this one number by hand:\n"
        f"logits[0, -1, {candidate_token_id}] = score for next token {candidate_token!r}\n"
        "\nIndex meaning:\n"
        "- 0 means batch item 0.\n"
        "- -1 means the last position in the current prompt.\n"
        f"- {candidate_token_id} means vocab token id {candidate_token_id}, which is {candidate_token!r}."
    )
    show_tensor(
        "last_position_hidden = hidden_states[0, -1, :]",
        last_position_hidden,
        note="[C]: the final prompt position's hidden vector.",
    )
    show_tensor(
        f"candidate_output_weight = lm_head.weight[{candidate_token_id}]",
        candidate_output_weight,
        note="[C]: the output weight row for candidate token '深度学习'.",
    )
    show_tensor(
        "elementwise_product = last_position_hidden * candidate_output_weight",
        elementwise_product,
        note=(
            "Multiply dimension by dimension.\n"
            "Then sum all C numbers to get this candidate token's logit."
        ),
    )
    print(f"manual_logit = elementwise_product.sum() = {manual_logit.item():.4f}")
    print(f"pytorch_logit = logits[0, -1, {candidate_token_id}] = {pytorch_logit.item():.4f}")
    print(
        "\nConclusion:\n"
        "lm_head(hidden_states) is just doing this dot-product calculation for:\n"
        "- every batch item\n"
        "- every token position\n"
        "- every candidate token id in the vocabulary"
    )

    next_token_logits = logits[0, -1, :]
    show_tensor(
        "next_token_logits = logits[0, -1, :]",
        next_token_logits,
        note=(
            "0 means the first batch item.\n"
            "-1 means the last token position in the current prompt.\n"
            ": means all vocab-token scores."
        ),
    )

    section("Step 05: next_token_logits -> probability -> next_token_id")
    probabilities = torch.softmax(next_token_logits, dim=-1)
    show_tensor(
        "probabilities = softmax(next_token_logits)",
        probabilities,
        note="Softmax turns raw logits into probabilities. All probabilities sum to 1.",
    )
    print(f"probabilities.sum() = {probabilities.sum().item():.4f}")

    top_values, top_indices = torch.topk(probabilities, k=5)
    print("\nTop-5 candidate next tokens:")
    for rank, (probability, token_id_tensor) in enumerate(zip(top_values, top_indices), start=1):
        token_id = int(token_id_tensor.item())
        token = tokenizer.id_to_token[token_id]
        logit = next_token_logits[token_id].item()
        print(
            f"  rank {rank}: token={token!r:>8} "
            f"id={token_id:>2} logit={logit:>7.4f} probability={probability.item():.4f}"
        )

    next_token_id = int(torch.argmax(next_token_logits).item())
    next_token = tokenizer.id_to_token[next_token_id]
    print("\nGreedy choice:")
    print(f"next_token_id = {next_token_id}")
    print(f"next_token = {next_token!r}")
    print(
        "\nMeaning:\n"
        "- Greedy means choosing the token id with the largest logit.\n"
        "- Later we will compare greedy with temperature and top_p sampling."
    )

    section("Step 06: append next_token_id -> decode")
    new_input_ids = torch.cat(
        [input_ids, torch.tensor([[next_token_id]], dtype=torch.long)],
        dim=1,
    )
    show_tensor(
        "new_input_ids",
        new_input_ids,
        note="[B, T] becomes [B, T + 1] after appending the predicted token id.",
    )

    decoded_text = tokenizer.decode(new_input_ids[0].tolist())
    print(f"decoded_text = {decoded_text!r}")
    print(
        "\nOne generation step summary:\n"
        "prompt text -> tokens -> input_ids -> embeddings -> hidden_states -> logits\n"
        "-> choose next_token_id -> append -> decode"
    )


if __name__ == "__main__":
    main()
