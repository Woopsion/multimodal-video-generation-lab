# Day 5：Transformer Encoder、Embedding 与 Self-Attention

> 日期：2026-05-15  
> 对应实验：`experiments/day-05-transformer-encoder/encoder_shape_demo.py`  
> 可视化说明：`docs/day-05-transformer-encoder/index.html`

## 1. 当前阶段目标

Day 5 的目标不是马上训练 Transformer，而是先把 Transformer Encoder 中的数据形状和核心对象看清楚：

```text
token id
-> token embedding
-> position embedding
-> encoder input
-> Q / K / V
-> attention score
-> softmax 权重
-> weighted sum
-> updated token representation
```

## 2. 当前仍需重点澄清的问题

这些问题来自当前学习过程，后续需要逐个拆开验证：

1. `embedding` 到底是什么？
2. 一个 token 是如何和其他 token 计算，得到自己的新向量表达的？
3. Self-Attention 到底是什么？
4. `position_embedding` 到底是什么？
5. 位置 id `[0, 1, 2, 3, 4]` 是什么意思？
6. 位置 id 如何通过 embedding 表变成向量？
7. 位置 embedding 表为什么也是可学习参数？
8. 每个 token 是否除了自己的 embedding 向量，还会通过 `W_Q`、`W_K`、`W_V` 三个可学习矩阵生成 Q/K/V？
9. Query / Key / Value 中说的“信息”和“内容”到底是什么？它们在代码里是什么形态？
10. softmax 中的 `exp(score)` 是不是 `e^score`？
11. softmax 是否是为了把分数变成正数，并归一化为加起来等于 1 的权重？
12. `output_t = Σ attention_weight(t, j) * v_j` 这个公式具体如何计算？
13. 为什么通过 attention 的加权求和可以得到 token 的新向量？
14. 这个新向量是如何通过 loss 和梯度影响 embedding 表、QKV 矩阵等参数的？

## 3. 先建立的基本理解

### 3.1 Token embedding

Token embedding 可以先理解成：

```text
把离散的 token id，查表变成一行连续数字向量。
```

例如 token id 是 `5`，embedding 表中第 5 行是：

```text
[0.12, -0.35, 0.88, 0.04]
```

那么 token id `5` 就会被表示成这个向量。

这个表是可学习参数，训练时会被 loss 和梯度不断调整。

### 3.2 Position embedding

Position embedding 可以先理解成：

```text
把“第几个位置”也查表变成一行向量。
```

位置 id：

```text
[0, 1, 2, 3, 4]
```

表示：

```text
第 0 个位置、第 1 个位置、第 2 个位置、第 3 个位置、第 4 个位置
```

它不是 token 内容，而是 token 在序列中的位置。

Transformer 通常把两种信息相加：

```text
encoder_input = token_embedding + position_embedding
```

含义是：

```text
这个位置上是什么 token + 这个 token 在哪里
```

### 3.3 Q / K / V

当前先用通俗版本记：

```text
Query：我想找什么
Key：我有什么可被匹配
Value：我真正提供给别人的内容
```

在代码里，它们不是自然语言，而是向量。

每个 token 的输入向量 `x_t` 会经过可学习线性层得到：

```text
q_t = x_t * W_Q
k_t = x_t * W_K
v_t = x_t * W_V
```

`W_Q`、`W_K`、`W_V` 是可学习参数矩阵，不是每个 token 单独拥有一套，而是这一层共享的参数。

### 3.4 Softmax 权重

Attention score 会经过 softmax：

```text
attention_weight(t, j) =
exp(score(t, j)) / 所有 j 的 exp(score(t, j)) 总和
```

其中 `exp(x)` 就是：

```text
e^x
```

softmax 的作用：

```text
把一组任意分数变成一组非负权重，并且所有权重加起来等于 1。
```

### 3.5 加权求和

公式：

```text
output_t = Σ attention_weight(t, j) * v_j
```

可以先理解成：

```text
当前 token t 按照注意力权重，从所有 token 的 value 向量中混合出一个新向量。
```

如果某个 token 的权重大，它的 value 向量对 `output_t` 的贡献就大。

这个 `output_t` 就是 token t 结合上下文后的新表示。

## 4. 当前实验文件的作用

`encoder_shape_demo.py` 不训练模型，只打印张量和 shape。

当前脚本已经适合用来观察：

```text
token_ids:        [B, T]
token_vectors:    [B, T, D]
position_vectors: [B, T, D]
qkv_packed:       [B, T, 3D]
Q/K/V:            [B, H, T, d]
```

后续还需要继续补充：

```text
attention_scores: [B, H, T, T]
attention_weights:[B, H, T, T]
attention_output: [B, T, D]
```

下一步学习时，应该逐段运行并对照打印结果，而不是只看公式。
