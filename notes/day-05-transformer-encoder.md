# Day 5：Transformer Encoder 学习状态

> 日期：2026-05-19
> 最终笔记：`docs/day-05-transformer-encoder/index.html`
> 对应实验：`experiments/day-05-transformer-encoder/encoder_shape_demo.py`

## 当前状态

Day 5 已完成第一轮学习收口。最终学习笔记以 HTML 为准：

```text
docs/day-05-transformer-encoder/index.html
```

这个 HTML 已经整理为完整 Day5 笔记，包含：

- tokenizer、token id、vocab、embedding 的关系。
- token embedding 与 position embedding。
- Q / K / V 的含义、shape 和计算方式。
- scaled dot-product attention：
  `Q @ K^T / sqrt(d)`、softmax、attention weights、weighted sum of V。
- Multi-Head Attention 的拆 head、各 head 独立计算、合并 heads。
- output projection。
- residual connection、LayerNorm、Feed Forward Network。
- 最终 Transformer Encoder Block 正式架构图。

## 当前代码完成范围

`encoder_shape_demo.py` 是不训练的 shape demo，目标是把 Encoder Block 的 forward 主线逐步打印清楚。

当前脚本已经覆盖：

```text
token_ids
-> token embedding + position embedding
-> encoder_input
-> Q/K/V projection
-> split heads
-> attention scores
-> scaled attention scores
-> attention weights
-> context = weights @ V
-> merge heads
-> output projection
-> first residual connection
-> first LayerNorm
-> Feed Forward Network
-> second residual connection
-> second LayerNorm
-> encoder_output
```

最终 shape 主线：

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

## 验收结论

Day5 的验收目标已经达到：

- 能解释 Transformer Encoder Block 的数据流。
- 能解释每个关键 shape 的含义。
- 能说明 Self-Attention、Multi-Head、Residual、LayerNorm、FFN 分别解决什么问题。
- 能对照代码理解一个 token 如何从输入向量流到最终 `encoder_output`。
- 能画出当前 demo 对应的 Post-LN Encoder Block 架构。

## 后续建议

Day5 到这里不再继续扩写概念。下一步如果继续推进，应进入工程封装：

```text
把当前散开的 forward 操作封装成 EncoderBlock 类。
```

也就是从“shape demo”进入“可复用模块实现”。
