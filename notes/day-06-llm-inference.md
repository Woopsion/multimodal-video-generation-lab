# Day 6 - LLM 基础与推理流程

## 当前阶段

- 日期：2026-05-21
- 学习状态：Day6 推理主线已完成；Instruction Tuning / LoRA / QLoRA 延后到微调专题。
- 主笔记：`docs/day-06-llm-inference/index.html`
- 当前 demo：`experiments/day-06-llm-inference/llm_inference_shape_demo.py`
- 采样 demo：`experiments/day-06-llm-inference/sampling_strategy_demo.py`
- KV Cache demo：`experiments/day-06-llm-inference/kv_cache_shape_demo.py`
- 真实 Transformers demo：`experiments/day-06-llm-inference/transformers_real_generation_demo.py`
- 真实生成参数 demo：`experiments/day-06-llm-inference/generation_controls_demo.py`

## Part 1 目标

先建立 LLM 推理流程的主线：

```text
prompt text
-> tokenizer.encode
-> input_ids
-> token embedding + position embedding
-> causal transformer blocks
-> logits
-> choose next_token_id
-> append next_token_id
-> repeat
-> tokenizer.decode
```

## 当前需要掌握

- `prompt` 是原始输入文本。
- `tokenizer` 负责把文本切成 token，并把 token 映射成整数 `token id`。
- `input_ids` 只是一串整数 id，不是 token 向量，也不是语义详情。
- `nn.Embedding` 根据 token id 查表，把 `[B, T]` 形状的 id 张量变成 `[B, T, C]` 形状的向量张量。
- Causal LM 生成时只能看左边已有 token，不能偷看未来 token。
- LLM 的 `logits` 类似 Day3 MNIST 的分类 logits，但类别不再是 10 个数字，而是整个词表里的 token。
- 生成下一个 token 时，通常取最后一个位置的 logits，再根据 greedy、temperature、top_p 等策略选择 `next_token_id`。

## Part 2 - shape demo

已新增一个不加载真实大模型的最小 shape demo：

```text
experiments/day-06-llm-inference/llm_inference_shape_demo.py
```

运行：

```bash
.venv/bin/python experiments/day-06-llm-inference/llm_inference_shape_demo.py
```

目标是打印并理解：

- prompt
- tokens
- input_ids
- token vectors
- position vectors
- hidden_states
- logits
- next_token_logits
- probabilities
- next_token_id
- decoded text

重要边界：

- 这个 demo 使用手写小词表和 toy tokenizer。
- 这个 demo 不训练模型。
- 这个 demo 暂时跳过真实 causal Transformer blocks，只保留 logits 路径。
- `lm_head` 被手动安排为更容易读的输出，真实 LLM 中这部分参数来自训练。

## 阶段复盘：LLM 核心流程

### 1. Tokenizer 与词表

Tokenizer 是把文本切成 token，并把 token 映射成 token id 的程序。

常见 tokenizer 构建方法包括 BPE、WordPiece、Unigram。以 BPE 为例：

1. 初始化时，词表包含基础字符或字节。
2. 在大量语料中统计相邻 token 对的频率。
3. 把最高频的一对合并成新 token，加入词表。
4. 重复这个过程，直到达到目标词表大小。

高频组合可能变成一个 token；低频词会被拆成更小的子词。

编码文本时，使用已经构建好的词表和规则。一个容易理解的实现是最长匹配：

```python
for token in self.tokens_by_length:
    if text.startswith(token, cursor):
        matched_token = token
        break
```

`text.startswith(token, cursor)` 的意思是：从 `cursor` 位置开始，看文本是否以当前 `token` 开头。

需要区分：

- tokenizer 是预处理工具，负责文本到 token id。
- embedding 是模型参数，负责 token id 到向量。
- 主模型训练时通常不通过梯度更新 tokenizer 的切分规则。

### 2. Embedding：token id 到向量

Embedding 矩阵形状：

```text
token_embedding.weight.shape = [vocab_size, embedding_dim]
```

每个 token id 对应 embedding 矩阵中的一行。

```text
input_ids.shape = [B, T]
token_vectors.shape = [B, T, C]
```

含义：

- `B` 是 batch size。
- `T` 是当前序列 token 数量。
- `C` 是每个 token 向量的维度。

Embedding 通常随机初始化。训练后，它会通过反向传播学习到更有用的表示。

### 3. Token Embedding + Position Embedding

模型输入通常来自两部分相加：

```text
model_input = token_vectors + position_vectors
```

- `token_vectors` 表示这个位置是什么 token。
- `position_vectors` 表示这个 token 在第几个位置。

相加后，每个位置的向量同时包含“内容信息”和“位置信息”。

### 4. Causal Transformer Block

Causal Decoder 的输入通常是：

```text
hidden_states.shape = [B, T, C]
```

单层大致流程：

1. LayerNorm。
2. 线性投影得到 Q、K、V。
3. Causal Self-Attention。
4. 残差连接。
5. LayerNorm。
6. Feed Forward Network。
7. 残差连接。

多层堆叠后，每个位置的 hidden state 会融合它自己和左侧上下文的信息。

### 5. Causal Self-Attention

对某个位置 `t`：

```text
q_t = x_t W_Q
k_t = x_t W_K
v_t = x_t W_V
```

注意力分数：

```text
score[t, j] = q_t · k_j
```

其中：

- `t` 是正在更新的 token 位置。
- `j` 是被看的 token 位置。
- `j > t` 表示未来位置。

Causal Mask 会把未来位置的分数改成 `-inf`：

```text
if j > t:
    score[t, j] = -inf
```

经过 softmax 后，未来位置权重变成 0：

```text
alpha[t, :] = softmax(score[t, :] / sqrt(d_k))
output_t = sum(alpha[t, j] * v_j, j = 0 到 t)
```

这样当前位置只能从自己和左侧 token 聚合信息。

为什么训练时必须 mask：

- 训练输入通常是完整序列。
- 如果没有 mask，模型预测某个位置时会看到未来正确答案。
- 这会让模型学会作弊，推理时因为未来 token 不存在，能力会崩。

### 6. Hidden States 到 Logits

Transformer 输出的是隐藏向量，不是文字。最后要通过 `lm_head` 把隐藏向量映射到词表分数。

```python
lm_head = nn.Linear(in_features=C, out_features=V, bias=False)
```

形状：

```text
hidden_states.shape = [B, T, C]
lm_head.weight.shape = [V, C]
logits.shape = [B, T, V]
```

矩阵形式：

```text
logits = hidden_states @ lm_head.weight.T
```

单个 logit 的公式：

```text
logits[b, t, i] =
sum(hidden_states[b, t, k] * lm_head.weight[i, k], k = 0 到 C - 1)
```

也就是：

```text
logits[i] = h · w_i
```

其中：

- `h` 是最后位置的 hidden state。
- `w_i` 是候选 token `i` 在 `lm_head.weight` 里的权重向量。
- 点积越大，表示模型越倾向于选择这个候选 token。

一些 LLM 会使用 weight tying，让输出层权重和 token embedding 权重共享或绑定。当前阶段只需要先理解：`lm_head` 的每一行对应词表中的一个候选 token。

### 7. 推理生成循环

推理时，模型参数固定，不再更新。生成是逐 token 进行的：

```text
当前 input_ids
-> token embedding + position embedding
-> 多层 causal Transformer
-> hidden_states
-> 取最后位置 hidden vector
-> lm_head 得到 next_token_logits
-> softmax / sampling 选择 next_token_id
-> append 到 input_ids
-> 重复直到结束
```

第一个生成 token 不是凭空产生的。它至少基于用户 prompt，或者基于特殊起始 token，例如 `<BOS>`。

### 8. 关键概念辨析

| 概念 | 说明 |
| --- | --- |
| Tokenizer | 将文本切分并映射成 token id 的程序，使用预先构建的词表和规则。 |
| Embedding | 模型参数矩阵，把 token id 查表变成连续向量，训练时会被优化。 |
| Causal Mask | 强制每个位置只能看自己和左侧 token，保证自回归性质。 |
| Hidden State | 经过 Transformer 层处理后的 token 向量，编码了当前位置可见上下文。 |
| Logits | Hidden state 通过 `lm_head` 得到的原始分数，每个值对应一个候选 token。 |
| 自回归生成 | 基于已生成序列预测下一个 token，拼接后继续预测。 |

## Part 3 - Sampling 采样策略

当模型输出 `next_token_logits` 后，还没有真正生成下一个 token。Sampling 要解决的问题是：

```text
从 next_token_logits 这组词表分数里，选出一个 next_token_id。
```

运行 demo：

```bash
.venv/bin/python experiments/day-06-llm-inference/sampling_strategy_demo.py
```

### 1. Greedy Decoding

Greedy decoding 永远选择 logit 最大的 token。

```text
next_token_id = argmax(next_token_logits)
```

特点：

- 同一组 logits 每次输出一样。
- 不随机。
- 低分 token 永远没有机会被选中。
- 长文本生成时可能更容易重复或进入固定套路。

### 2. Temperature

Temperature 在 softmax 前调整 logits：

```text
adjusted_logits = logits / temperature
probabilities = softmax(adjusted_logits)
```

含义：

- `temperature < 1`：分布更尖锐，高分 token 更占优势，输出更保守。
- `temperature = 1`：不改变 logits。
- `temperature > 1`：分布更平，低分 token 机会增加，输出更发散。

注意：

- temperature 不是“创造力本身”。
- 它只是改变候选 token 概率的集中程度。
- 模型不知道的知识，不会因为 temperature 变高就突然知道。

### 3. Top-p / Nucleus Sampling

Top-p 保留的是“累计概率达到 `p` 的最小候选集合”。

示例：

```text
深度学习 0.55
AI       0.25
电影     0.07
音乐     0.05
...
```

如果 `top_p = 0.85`：

```text
0.55 + 0.25 = 0.80，还没到 0.85
0.80 + 0.07 = 0.87，达到 0.85
```

所以保留：

```text
深度学习、AI、电影
```

其他 token 概率置为 0，再重新归一化。

重点：

- `top_p` 不是保留固定数量的 token。
- `top_p` 不是保留 p 百分比的 token。
- 它保留的是累计概率质量。

### 4. Temperature 和 top_p 的组合

常见顺序：

```text
adjusted_logits = logits / temperature
probabilities = softmax(adjusted_logits)
filtered_probabilities = top_p_filter(probabilities, top_p)
next_token_id = multinomial_sample(filtered_probabilities)
```

其中 `multinomial_sample` 表示按概率抽样。概率越高越容易抽到，但不是最高概率一定被抽到。

### 5. max_new_tokens

`max_new_tokens` 控制最多生成多少个新 token。

```text
for step in range(max_new_tokens):
    logits = model(input_ids)
    next_token_id = choose_from_logits(logits)
    input_ids.append(next_token_id)
    if next_token_id == eos_token_id:
        break
```

它控制的是循环上限，不是模型质量。

停止原因通常有两种：

- 达到 `max_new_tokens`。
- 生成 `<eos>` 结束 token。

### 6. 参数辨析

| 参数 / 策略 | 控制什么 | 不控制什么 |
| --- | --- | --- |
| greedy | 永远选最高分 token。 | 不探索其他候选。 |
| temperature | 概率分布尖锐还是平坦。 | 不限制候选集合大小。 |
| top_p | 保留多少累计概率质量。 | 不表示保留几个 token。 |
| max_new_tokens | 最多追加多少个新 token。 | 不保证完整句子，也不提高质量。 |

## Part 4 - KV Cache

KV Cache 解决的是生成阶段的重复计算问题。

运行 demo：

```bash
.venv/bin/python experiments/day-06-llm-inference/kv_cache_shape_demo.py
```

### 1. 核心结论

KV Cache 不改变模型参数，不改变 attention 的数学含义，也不提高模型知识能力。

它只做一件事：

```text
把历史 token 在每一层 attention 中已经算好的 K/V 保存起来，
后续生成新 token 时直接复用。
```

这样可以避免每生成一个 token，就重新计算整段历史 token 的 K/V。

### 2. 为什么能缓存 K/V

Self-Attention 中，新 token 要用自己的 Query 去看历史 Key，再混合历史 Value：

```text
score_new,j = q_new · k_j
attention_weight_new = softmax(score_new)
output_new = sum(attention_weight_new,j * v_j)
```

历史 token 的 `k_j` 和 `v_j` 在当前层已经算出来后，后续不会变。因此可以存起来。

### 3. Prefill 和 Decode

KV Cache 推理通常分两段：

```text
prefill: 一次性处理 prompt，计算并保存所有 prompt token 的 K/V。
decode: 每次只处理最新 token，计算新 token 的 Q/K/V，并把新 K/V 追加进 cache。
```

Prefill 后：

```text
key_cache.shape = [B, H, T_prompt, d]
value_cache.shape = [B, H, T_prompt, d]
```

Decode 某一步：

```text
new_query.shape = [B, H, 1, d]
key_cache.shape = [B, H, T_total, d]
scores.shape = [B, H, 1, T_total]
```

### 4. 不用 cache vs 使用 cache

不用 cache：

```text
query.shape = [B, H, S, d]
key.shape   = [B, H, S, d]
value.shape = [B, H, S, d]
scores.shape = [B, H, S, S]
```

含义：当前完整序列长度是 `S`，每一步都重算整个序列。

使用 cache：

```text
new_query.shape = [B, H, 1, d]
key_cache.shape = [B, H, T_total, d]
value_cache.shape = [B, H, T_total, d]
scores.shape = [B, H, 1, T_total]
```

含义：只计算新 token 的 Query，让它看缓存中的全部历史 Key。

### 5. 计算量直觉

假设 prompt 长度是 4，要生成 3 个 token。

不用 cache 时，每一步重算当前完整序列：

```text
4 + 5 + 6 = 15 个 token 参与 Q/K/V 投影
```

使用 cache：

```text
4 个 prompt token prefill
+ 1 个新 token
+ 1 个新 token
+ 1 个新 token
= 7 个 token 参与 Q/K/V 投影
```

demo 中粗略统计：

```text
without cache projected tokens = 15
with cache projected tokens    = 7
without cache attention cells  = 154
with cache attention cells     = 68
```

### 6. 代价

KV Cache 是用空间换时间。

每一层都要保存：

```text
key_cache.shape = [B, H, T_total, d]
value_cache.shape = [B, H, T_total, d]
```

序列越长，缓存越大；batch 越大，缓存也越大。所以 KV Cache 会加速生成，但会增加显存占用。

### 7. 重要误区：V 不是更新后的 hidden state

K 和 V 不是 attention 之后更新出来的结果，而是 attention 之前由当前层输入线性投影出来的中间张量：

```text
当前层输入 hidden_states
-> Linear_Q 得到 Q
-> Linear_K 得到 K
-> Linear_V 得到 V

attention_scores = Q @ K^T
attention_weights = softmax(mask(attention_scores) / sqrt(d))
context = attention_weights @ V

updated_hidden_states
= OutputProjection(context)
-> residual
-> LayerNorm
-> FFN
-> residual / LayerNorm
```

所以：

- `Q/K` 决定注意力分数，也就是“看谁、看多少”。
- `V` 是被权重读取和混合的内容材料。
- 真正被更新并传给下一层的是 `hidden_states`。
- KV Cache 保存的是每层历史 token 的 `K/V`，不是完整的历史 `hidden_states`。

新 token decode 时：

```text
只为新 token 计算 q_new, k_new, v_new
把 k_new, v_new 追加到当前层 cache
q_new @ key_cache.T 得到分数
softmax 后得到权重
weights @ value_cache 得到 context
context 再经过输出投影、残差、FFN，形成新 hidden state
```

旧 token 不需要因为新 token 出现而重新更新，因为 causal mask 规定旧 token 不能看右边未来 token。

## Part 5 - Transformers 真实模型推理

Part 5 开始把前面的 toy demo 换成真实 Hugging Face Transformers 推理，但学习目标仍然是看懂流程，不追求输出质量。

运行 demo：

```bash
.venv/bin/python experiments/day-06-llm-inference/transformers_real_generation_demo.py
```

当前脚本默认离线 / local-only。若你手动放置了项目内可选 pretrained 目录，会优先从这里加载：

```text
experiments/day-06-llm-inference/pretrained/sshleifer-tiny-gpt2/
```

这个目录不提交到 git。若本机没有模型文件或 Hugging Face 缓存，需要显式允许下载：

```bash
.venv/bin/python experiments/day-06-llm-inference/transformers_real_generation_demo.py --model sshleifer/tiny-gpt2 --allow-download
```

如果只想查看本地 pretrained 文件和权重 tensor shape，可以运行：

```bash
.venv/bin/python experiments/day-06-llm-inference/inspect_pretrained_model_files.py
```

### 1. 真实模型由哪些文件组成

| 文件 / 对象 | 作用 |
| --- | --- |
| `tokenizer.json` / `vocab.json` / `merges.txt` | 文本和 token id 的转换规则。 |
| `config.json` | 模型结构超参数，例如 hidden size、层数、head 数、词表大小。 |
| `model.safetensors` / `pytorch_model.bin` | 训练好的参数矩阵，例如 embedding、Q/K/V、FFN、LM Head。 |

### 2. 真实代码与 Day6 主线

```text
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

encoded = tokenizer(prompt, return_tensors="pt")
input_ids = encoded["input_ids"]              # [B, T]
attention_mask = encoded["attention_mask"]    # [B, T]

outputs = model(**encoded, use_cache=True)
logits = outputs.logits                       # [B, T, V]
past_key_values = outputs.past_key_values     # 每层缓存的 K/V

next_token_logits = logits[:, -1, :]          # [B, V]
next_token_id = argmax(next_token_logits)
```

这和前面的手写 demo 是同一条链路：

```text
prompt text
-> tokenizer
-> input_ids
-> model forward
-> logits
-> 取最后位置 next_token_logits
-> 选择 next_token_id
-> append
-> decode
```

### 3. attention_mask 和 causal mask 的区别

`attention_mask` 主要解决 padding 问题：

```text
[1, 1, 1, 0, 0]
```

表示前三个是真 token，后两个只是补齐长度的 padding。

`causal mask` 解决未来泄露问题：当前位置只能看自己和左侧 token，不能看右侧未来 token。

在 `AutoModelForCausalLM` 中，causal mask 通常由模型内部处理；`attention_mask` 通常由 tokenizer 输出并传给模型。

### 4. model.eval 和 inference_mode

`model.eval()` 把模型切换到推理模式，关闭 dropout 等训练行为。

`torch.inference_mode()` 告诉 PyTorch 不需要梯度和反向传播图，推理时更省内存、更快。

### 5. generate 不是魔法

`model.generate()` 内部仍然是循环：

```text
outputs = model(input_ids, past_key_values=cache, use_cache=True)
next_token_logits = outputs.logits[:, -1, :]
next_token_id = greedy / temperature / top_p 选出来
input_ids = concat(input_ids, next_token_id)
cache = outputs.past_key_values
```

它只是把前向计算、KV Cache、sampling、停止条件封装到一个成熟接口里。

## Part 6 - 真实模型生成参数、停止条件、输出质量

运行 demo：

```bash
.venv/bin/python experiments/day-06-llm-inference/generation_controls_demo.py
```

Part 6 的核心结论：

- 生成参数不是训练参数，不会改变模型权重。
- 模型 forward 仍然输出 `logits`。
- 生成参数主要控制如何从最后位置的 logits 中选择 `next_token_id`。

关键参数：

| 参数 | 作用 |
| --- | --- |
| `max_new_tokens` | 最多追加多少个新 token。 |
| `do_sample=False` | 不随机采样，通常每一步选择最高分 token。 |
| `do_sample=True` | 按概率分布随机采样。 |
| `temperature` | 调节概率分布尖锐或平坦。 |
| `top_k` | 只从概率最高的 k 个候选中采样。 |
| `top_p` | 保留累计概率达到 p 的候选集合。 |
| `repetition_penalty` | 降低已出现 token 再次出现的倾向。 |
| `eos_token_id` | 生成结束符时提前停止。 |

生成参数在循环里的位置：

```text
outputs = model(input_ids, past_key_values=cache, use_cache=True)
next_token_logits = outputs.logits[:, -1, :]

adjusted_logits = repetition_penalty(next_token_logits)
probabilities = softmax(adjusted_logits / temperature)
candidate_set = top_k_or_top_p_filter(probabilities)
next_token_id = greedy_or_sample(candidate_set)
```

注意：tiny-gpt2 是测试模型，输出质量差是正常的。参数只能改变选择策略，不能创造模型权重里没有的语言能力。

### 本次真实运行报告

运行命令：

```bash
.venv/bin/python experiments/day-06-llm-inference/generation_controls_demo.py --device cpu --prompt "Deep learning is"
```

输入：

```text
input_ids = [29744, 4673, 318]
tokens = ['Deep', 'Ġlearning', 'Ġis']
eos_token_id = 50256
pad_token_id = 50256
```

结果摘要：

| case | 关键参数 | 观察结果 | 停止原因 |
| --- | --- | --- | --- |
| A | `do_sample=False`, `max_new_tokens=12` | `factors` 重复 12 次 | 达到 `max_new_tokens` |
| B | `do_sample=False`, `max_new_tokens=4` | `factors` 重复 4 次 | 达到 `max_new_tokens` |
| C | `temperature=0.7`, `top_k=20` | 输出开始随机，但仍受 top-k 限制 | 达到 `max_new_tokens` |
| D | `temperature=1.4`, `top_k=20` | 输出更发散，出现更多奇怪 token | 达到 `max_new_tokens` |
| E | `top_k=5` | 每步只从前 5 个候选采样 | 达到 `max_new_tokens` |
| F | `top_p=0.8` | 候选集合按累计概率动态变化 | 达到 `max_new_tokens` |
| G | `repetition_penalty=1.25` | greedy 不再一直重复 `factors` | 达到 `max_new_tokens` |

关键结论：

- 本次所有 case 都是由 `max_new_tokens` 停止，不是由 `eos_token_id` 提前停止。
- `do_sample=False` 更稳定，但弱模型更容易重复。
- `temperature`、`top_k`、`top_p` 改变的是采样池和概率分布。
- `repetition_penalty` 能降低重复倾向，但不能提高模型知识能力。
- tiny-gpt2 的价值是观察真实 Transformers API 和推理流程，不是获得高质量文本。

## Day6 收口判断

已完成：

- Tokenizer、词表、`input_ids`、decode。
- token embedding、position embedding、hidden states、logits。
- Causal LM 和 Causal Mask。
- KV Cache 的 prefill / decode 过程。
- 使用本机 tiny-gpt2 文件加载真实 Transformers 模型做生成。
- 对比 `max_new_tokens`、`temperature`、`top_k`、`top_p`、`repetition_penalty` 对输出的影响。

延后：

- chat template、聊天模型输入格式、真实 instruction model 差异。
- Instruction Tuning、LoRA、QLoRA。它们属于训练 / 微调主题，后续单独学习更清晰。

## 下一步

- 进入 Week 1 复盘，整理 Day1 到 Day6 的主线、代码路径、关键概念和遗留问题。
