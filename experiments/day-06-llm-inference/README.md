# Day 6 - LLM Inference Demos

This folder contains the Day 6 code demos for LLM inference.

The learning path starts with toy shape demos, then moves to a real tiny GPT-2 model through Hugging Face Transformers. The main goal is to make the inference data flow visible:

```text
prompt
-> tokens
-> input_ids
-> token vectors
-> position vectors
-> hidden_states
-> logits
-> next_token_id
-> append
-> decode
```

Run:

```bash
.venv/bin/python experiments/day-06-llm-inference/llm_inference_shape_demo.py
```

Part 3 sampling demo:

```bash
.venv/bin/python experiments/day-06-llm-inference/sampling_strategy_demo.py
```

Part 4 KV Cache shape demo:

```bash
.venv/bin/python experiments/day-06-llm-inference/kv_cache_shape_demo.py
```

Part 5 real Transformers generation demo:

```bash
.venv/bin/python experiments/day-06-llm-inference/transformers_real_generation_demo.py
```

If the model is not already available locally or in the Hugging Face cache, explicitly allow the download:

```bash
.venv/bin/python experiments/day-06-llm-inference/transformers_real_generation_demo.py --model sshleifer/tiny-gpt2 --allow-download
```

Part 6 real generation controls demo:

```bash
.venv/bin/python experiments/day-06-llm-inference/generation_controls_demo.py
```

If the model is not already available locally or in the Hugging Face cache, explicitly allow the download:

```bash
.venv/bin/python experiments/day-06-llm-inference/generation_controls_demo.py --model sshleifer/tiny-gpt2 --allow-download
```

When present, the real-model scripts prefer this optional project-local pretrained model copy:

```text
experiments/day-06-llm-inference/pretrained/sshleifer-tiny-gpt2/
```

The `pretrained/` directory is intentionally ignored by git. Do not commit downloaded model weights or tokenizer files to the repository. If this directory is absent, the scripts can still use the Hugging Face cache, or download explicitly when `--allow-download` is passed.

Inspect the optional project-local pretrained files:

```bash
.venv/bin/python experiments/day-06-llm-inference/inspect_pretrained_model_files.py
```

Important constraints:

- The toy shape demo uses a tiny hand-written tokenizer for learning only.
- The toy model is not trained.
- The toy demo's `lm_head` is manually arranged so it produces a readable next token.
- Real LLM inference will use a real tokenizer, many causal Transformer layers, and learned model weights.
- The sampling demo uses fixed toy logits so `greedy`, `temperature`, `top_p`, and `max_new_tokens` can be studied without a real model.
- The KV Cache demo compares generation with and without cached K/V tensors using shape prints and rough work counts.
- The Transformers demo needs the optional `transformers` package. It is offline/local-only by default and requires `--allow-download` before remote downloads.
- The generation controls demo uses the real tiny GPT-2 model to compare `do_sample`, `temperature`, `top_k`, `top_p`, `repetition_penalty`, and stop reasons.
