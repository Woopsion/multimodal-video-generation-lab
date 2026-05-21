from __future__ import annotations

from dataclasses import dataclass

import torch


torch.manual_seed(7)
torch.set_printoptions(precision=4, sci_mode=False)


@dataclass(frozen=True)
class ToyVocab:
    id_to_token: dict[int, str]

    @property
    def token_to_id(self) -> dict[str, int]:
        return {token: token_id for token_id, token in self.id_to_token.items()}

    @property
    def size(self) -> int:
        return len(self.id_to_token)

    def decode(self, token_ids: list[int]) -> str:
        return "".join(
            self.id_to_token[token_id]
            for token_id in token_ids
            if self.id_to_token[token_id] != "<eos>"
        )


def section(title: str) -> None:
    print("\n" + "=" * 92)
    print(title)
    print("=" * 92)


def line() -> None:
    print("-" * 92)


def softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)


def show_distribution(
    title: str,
    vocab: ToyVocab,
    logits: torch.Tensor,
    probabilities: torch.Tensor,
) -> None:
    line()
    print(title)
    print(f"{'id':>2}  {'token':<8}  {'logit':>8}  {'probability':>11}")
    for token_id in torch.argsort(probabilities, descending=True).tolist():
        print(
            f"{token_id:>2}  {vocab.id_to_token[token_id]:<8}  "
            f"{logits[token_id].item():>8.4f}  {probabilities[token_id].item():>11.4f}"
        )
    print(f"probability sum = {probabilities.sum().item():.4f}")


def sample_from_probabilities(probabilities: torch.Tensor) -> int:
    """Draw one token id according to a probability distribution."""

    return int(torch.multinomial(probabilities, num_samples=1).item())


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")
    return logits / temperature


def top_p_filter(probabilities: torch.Tensor, top_p: float) -> tuple[torch.Tensor, list[int]]:
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    keep_sorted = cumulative_probs <= top_p
    keep_sorted[0] = True

    first_over_limit = torch.nonzero(cumulative_probs > top_p, as_tuple=False)
    if len(first_over_limit) > 0:
        keep_sorted[first_over_limit[0].item()] = True

    kept_indices = sorted_indices[keep_sorted].tolist()
    filtered = torch.zeros_like(probabilities)
    filtered[kept_indices] = probabilities[kept_indices]
    filtered = filtered / filtered.sum()
    return filtered, kept_indices


def toy_next_logits(vocab: ToyVocab, generated_ids: list[int]) -> torch.Tensor:
    """A hand-written fake model so max_new_tokens can be demonstrated."""

    token_to_id = vocab.token_to_id
    last_token = vocab.id_to_token[generated_ids[-1]]
    logits = torch.full((vocab.size,), -3.0)

    if last_token == "喜欢":
        logits[token_to_id["深度学习"]] = 4.2
        logits[token_to_id["AI"]] = 3.4
        logits[token_to_id["电影"]] = 2.2
        logits[token_to_id["音乐"]] = 1.8
    elif last_token in {"深度学习", "AI"}:
        logits[token_to_id["很"]] = 3.8
        logits[token_to_id["模型"]] = 2.5
        logits[token_to_id["训练"]] = 2.0
        logits[token_to_id["。"]] = 1.2
    elif last_token == "很":
        logits[token_to_id["有趣"]] = 4.0
        logits[token_to_id["难"]] = 2.4
        logits[token_to_id["。"]] = 1.5
    elif last_token in {"有趣", "难", "电影", "音乐", "模型", "训练"}:
        logits[token_to_id["。"]] = 4.2
        logits[token_to_id["<eos>"]] = 2.2
    elif last_token == "。":
        logits[token_to_id["<eos>"]] = 5.0
    else:
        logits[token_to_id["<eos>"]] = 4.0

    return logits


def choose_next_token(
    logits: torch.Tensor,
    *,
    strategy: str,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> tuple[int, torch.Tensor]:
    if strategy == "greedy":
        probabilities = softmax(logits)
        return int(torch.argmax(logits).item()), probabilities

    if strategy != "sample":
        raise ValueError("strategy must be 'greedy' or 'sample'")

    adjusted_logits = apply_temperature(logits, temperature)
    probabilities = softmax(adjusted_logits)
    if top_p < 1.0:
        probabilities, _ = top_p_filter(probabilities, top_p)
    return sample_from_probabilities(probabilities), probabilities


def generate(
    vocab: ToyVocab,
    prompt_ids: list[int],
    *,
    max_new_tokens: int,
    strategy: str,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> list[int]:
    generated_ids = prompt_ids.copy()
    eos_id = vocab.token_to_id["<eos>"]

    for step in range(max_new_tokens):
        logits = toy_next_logits(vocab, generated_ids)
        next_token_id, probabilities = choose_next_token(
            logits,
            strategy=strategy,
            temperature=temperature,
            top_p=top_p,
        )
        print(
            f"step {step + 1}: last={vocab.id_to_token[generated_ids[-1]]!r:<6} "
            f"next={vocab.id_to_token[next_token_id]!r:<8} "
            f"prob={probabilities[next_token_id].item():.4f}"
        )
        generated_ids.append(next_token_id)

        if next_token_id == eos_id:
            print("stop reason: generated <eos>")
            break
    else:
        print(f"stop reason: reached max_new_tokens={max_new_tokens}")

    return generated_ids


def main() -> None:
    vocab = ToyVocab(
        id_to_token={
            0: "<eos>",
            1: "深度学习",
            2: "AI",
            3: "电影",
            4: "音乐",
            5: "很",
            6: "有趣",
            7: "难",
            8: "模型",
            9: "训练",
            10: "。",
        }
    )
    token_to_id = vocab.token_to_id

    section("Step 01: one next_token_logits vector")
    print(
        "Assume the prompt is: 我喜欢\n"
        "The model has already produced next_token_logits for the last prompt position.\n"
        "Now the only question is: how do we choose the next token?"
    )
    next_token_logits = torch.tensor(
        [
            -2.0,  # <eos>
            4.2,  # 深度学习
            3.4,  # AI
            2.2,  # 电影
            1.8,  # 音乐
            0.2,  # 很
            -0.4,  # 有趣
            -1.0,  # 难
            0.8,  # 模型
            0.5,  # 训练
            1.1,  # 。
        ],
        dtype=torch.float32,
    )
    base_probabilities = softmax(next_token_logits)
    show_distribution("Base distribution: softmax(next_token_logits)", vocab, next_token_logits, base_probabilities)

    section("Step 02: greedy decoding")
    greedy_id = int(torch.argmax(next_token_logits).item())
    print("Greedy means: choose the token id with the largest logit.")
    print(f"largest logit token id = {greedy_id}")
    print(f"greedy next token = {vocab.id_to_token[greedy_id]!r}")
    print(
        "Greedy is deterministic: the same logits always produce the same next token.\n"
        "It does not use randomness and does not care about the rest of the probability mass."
    )

    section("Step 03: temperature")
    print(
        "Temperature changes the sharpness of the probability distribution before sampling.\n"
        "Formula: adjusted_logits = logits / temperature\n"
        "- temperature < 1: sharper, top token becomes more dominant.\n"
        "- temperature = 1: original distribution.\n"
        "- temperature > 1: flatter, lower-score tokens get more chance."
    )
    for temperature in [0.5, 1.0, 1.8]:
        adjusted_logits = apply_temperature(next_token_logits, temperature)
        probabilities = softmax(adjusted_logits)
        sampled_id = sample_from_probabilities(probabilities)
        show_distribution(
            f"temperature = {temperature}",
            vocab,
            adjusted_logits,
            probabilities,
        )
        print(f"sampled token = {vocab.id_to_token[sampled_id]!r}")

    section("Step 04: top_p nucleus sampling")
    print(
        "top_p keeps the smallest high-probability token set whose cumulative probability reaches top_p.\n"
        "Then it sets all other token probabilities to 0 and renormalizes the kept probabilities.\n"
        "top_p does not directly mean 'keep p percent of tokens'. It means 'keep p cumulative probability mass'."
    )
    probabilities = softmax(next_token_logits)
    for top_p in [0.60, 0.85, 0.95]:
        filtered_probabilities, kept_indices = top_p_filter(probabilities, top_p)
        kept_tokens = [vocab.id_to_token[token_id] for token_id in kept_indices]
        sampled_id = sample_from_probabilities(filtered_probabilities)
        show_distribution(
            f"top_p = {top_p}, kept tokens = {kept_tokens}",
            vocab,
            next_token_logits,
            filtered_probabilities,
        )
        print(f"sampled token = {vocab.id_to_token[sampled_id]!r}")

    section("Step 05: combine temperature and top_p")
    temperature = 1.4
    top_p = 0.85
    adjusted_logits = apply_temperature(next_token_logits, temperature)
    probabilities = softmax(adjusted_logits)
    filtered_probabilities, kept_indices = top_p_filter(probabilities, top_p)
    print(
        f"Common sampling order in this demo:\n"
        f"1. logits / temperature, temperature = {temperature}\n"
        f"2. softmax -> probabilities\n"
        f"3. top_p filter, top_p = {top_p}\n"
        f"4. multinomial sample from remaining probabilities"
    )
    show_distribution(
        f"temperature = {temperature}, top_p = {top_p}, kept ids = {kept_indices}",
        vocab,
        adjusted_logits,
        filtered_probabilities,
    )
    sampled_id = sample_from_probabilities(filtered_probabilities)
    print(f"sampled token = {vocab.id_to_token[sampled_id]!r}")

    section("Step 06: max_new_tokens controls the generation loop length")
    prompt_ids = [token_to_id["深度学习"], token_to_id["很"]]
    print(f"prompt tokens = {[vocab.id_to_token[token_id] for token_id in prompt_ids]}")
    print(
        "max_new_tokens is a hard upper bound on how many new token ids may be appended.\n"
        "It does not make the model smarter. It only limits the loop."
    )

    print("\nGreedy generation with max_new_tokens=2")
    greedy_ids = generate(
        vocab,
        prompt_ids,
        max_new_tokens=2,
        strategy="greedy",
    )
    print(f"decoded = {vocab.decode(greedy_ids)!r}")

    print("\nSampling generation with max_new_tokens=5, temperature=1.2, top_p=0.9")
    sampled_ids = generate(
        vocab,
        prompt_ids,
        max_new_tokens=5,
        strategy="sample",
        temperature=1.2,
        top_p=0.9,
    )
    print(f"decoded = {vocab.decode(sampled_ids)!r}")


if __name__ == "__main__":
    main()
