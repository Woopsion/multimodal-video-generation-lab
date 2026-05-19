# Day 5 - Transformer Encoder Shape Demo

This experiment is a no-training Transformer Encoder forward-pass shape demo.

It prints tensors and shapes step by step so the Encoder Block data flow can be inspected directly.

## Run

macOS / Linux:

```bash
.venv/bin/python experiments/day-05-transformer-encoder/encoder_shape_demo.py
```

Windows:

```powershell
.\.venv\Scripts\python.exe experiments/day-05-transformer-encoder/encoder_shape_demo.py
```

## What It Covers

```text
token ids
-> token embedding + position embedding
-> Q/K/V projection
-> split heads
-> attention scores
-> scaled attention scores
-> attention weights
-> context = weights @ V
-> merge heads
-> output projection
-> residual + LayerNorm
-> FFN
-> residual + LayerNorm
-> encoder_output
```

## Final Shape Flow

```text
encoder_input      [B,T,D] = [2,5,4]
attention_output   [B,T,D] = [2,5,4]
attention_residual [B,T,D] = [2,5,4]
attention_normed   [B,T,D] = [2,5,4]
ffn_hidden         [B,T,4D] = [2,5,16]
ffn_output         [B,T,D] = [2,5,4]
ffn_residual       [B,T,D] = [2,5,4]
encoder_output     [B,T,D] = [2,5,4]
```

## Final Notes

The final Day 5 learning note is:

```text
docs/day-05-transformer-encoder/index.html
```

Open it in a browser and use the final section, `Day5 正式架构图：Transformer Encoder Block`, as the review entry point.
