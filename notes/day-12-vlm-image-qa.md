# Day 12 - VLM 图像问答入门

> 当前阶段：Day12 已收口。已完成 VLM 架构理解、三类测试任务设计和 SmolVLM-256M 最小本地图像问答实验。

## 今日目标

- 从 Day11 的 CLIP 图文匹配过渡到 VLM 图像问答。
- 理解 VLM 的通用结构：vision encoder、projector、LLM、processor / prompt template。
- 掌握图片如何变成 LLM 可消费的 token-like 表示，并跑通轻量 VLM 本地推理。

## 主学习入口

- HTML 教学文档：`docs/day-12-vlm-image-qa/index.html`

## 当前最短主线

```text
image
-> vision encoder
-> visual tokens [B, N, Dv]
-> projector
-> projected visual tokens [B, N, Dl]
-> LLM context with text question
-> generated answer tokens
```

## Step 01 验收问题

1. CLIP 和 VLM 的输出目标有什么不同？
2. 为什么图片不能直接以像素形式进入 LLM？
3. projector 为什么要把视觉特征映射到 LLM hidden size？
4. 图像问答时，LLM 是依据什么生成答案？

## 收口边界

- 已下载并缓存 `HuggingFaceTB/SmolVLM-256M-Instruct` 用于 Day12 最小实验。
- 已跑通场景描述、OCR / 截图、图表理解三类样例，但只做教学级边界观察，不做正式 benchmark。
- 暂不进入 LLaVA 论文，Day13 再读。

## Vision encoder 复习

Vision encoder 是“图片特征提取器”的总称。它不直接生成答案，而是把图片编码成 visual tokens。

- `ViT` 是一种视觉 Transformer 架构：图片切 patch，patch 变 embedding，再进入 Transformer Encoder。
- `CLIP vision` 是 CLIP 模型里的视觉塔，通常可以是 ViT 架构，但重点是它经过图文对比学习训练，视觉特征更容易和文本语义对齐。
- 在 VLM 里，vision encoder 输出一般记作 `[B, N, Dv]`：`B` 是 batch，`N` 是视觉 token 数，`Dv` 是视觉编码器宽度。
- projector 会把 `[B, N, Dv]` 转成 `[B, N, Dl]`，其中 `Dl` 是 LLM 的 hidden size。

最短记忆：`ViT` 偏结构名，`CLIP vision` 偏训练好的视觉塔来源；VLM 可以使用 CLIP 训练过的 ViT 作为 vision encoder。

## Step 02：四个部件

当前把 VLM 拆成四个部件理解：

| 部件 | 作用 | 输出/结果 |
| --- | --- | --- |
| Vision encoder | 把图片编码成视觉 token | `[B, N, Dv]` |
| Projector | 把视觉 token 宽度对齐到 LLM hidden size | `[B, N, Dl]` |
| Prompt template | 组织图片占位符、用户问题和回答位置 | 模型需要的对话格式 |
| LLM | 根据图片 token 和问题 token 生成答案 | answer tokens |

最短形状链路：

```text
image_features = vision_encoder(image)   # [B, N, Dv]
image_tokens = projector(image_features) # [B, N, Dl]
question_tokens                         # [B, T, Dl]
LLM input                               # [B, N + T, Dl]
```

Step 02 的关键结论：projector 不负责看图，也不负责回答；它负责把 vision encoder 的输出变成 LLM 能接收的输入宽度。

## Step 03：image tokens 进入上下文

当前理解校准：

- projector 的作用是把 `[B, N, Dv]` 转成 `[B, N, Dl]`，让图片 token 的宽度和 LLM token embedding 的宽度一致。
- projector 不负责看图；看图由 vision encoder 完成。
- projector 不负责回答；回答由 LLM 基于上下文做 next-token prediction 完成。
- prompt template 提供文字结构和图片插槽，让模型知道哪里是图片、哪里是用户问题、哪里开始生成回答。

更精确的链路：

```text
prompt text / template tokens
+ projected image tokens [B, N, Dl]
+ question tokens [B, T, Dl]
-> LLM context
-> predict next token
-> repeat until answer complete
```

关键区别：`<image>` 只是模板里的占位符，不是图片内容本身。真正进入 LLM 计算的是 projector 输出的 image token embeddings。

## Step 04：多模态 self-attention

当前要理解：projected image tokens 和 text tokens 都已经是 `[*, Dl]` 宽度的向量，所以它们可以进入同一个 Transformer 上下文。

```text
context = image_tokens + question_tokens
Q = context @ Wq
K = context @ Wk
V = context @ Wv
attention = softmax(Q @ K.T) @ V
```

直观理解：

- 问题 token 可以通过 attention 关注相关图片 token。
- 回答 token 可以同时关注问题 token 和图片 token。
- 下一 token 预测仍然是 LLM 的核心，只是上下文里多了视觉 token。

关键结论：VLM 不是必须先把图片翻译成完整文字再回答；它可以在 embedding 层把视觉 token 放进上下文，让 Transformer 的 attention 自己学习如何利用这些视觉信息。
## Step 05：三类测试任务设计

当前不要只问“请描述这张图”。这种问题太宽泛，模型即使回答流畅，也不容易判断它到底有没有看对关键视觉证据。

三类最小测试任务：

| 类型 | 考察能力 | 问题设计方式 |
| --- | --- | --- |
| 场景描述 | 物体、属性、动作、空间关系 | 问主体是谁、在做什么、彼此是什么关系 |
| OCR / 截图 | 可见文字识别、界面布局理解 | 问标题、按钮文字、提示信息、界面元素位置 |
| 图表理解 | 坐标轴、图例、数值比较、趋势推理 | 问最大值、变化趋势、A/B 对比 |

最小测试集写法：

```text
sample_01_scene:
  image: 一张有主体和动作关系的普通照片
  question: 这张图的主要内容是什么？主体之间是什么关系？
  expected: 人工写出可核验答案

sample_02_ocr:
  image: 一张有清晰文字的截图或海报
  question: 图中标题是什么？按钮文字是什么？
  expected: 人工抄写关键文字和位置

sample_03_chart:
  image: 一张简单柱状图或折线图
  question: 哪个类别最高？整体趋势是什么？
  expected: 人工写出最大值、趋势、对比关系
```

Step 05 的关键结论：测试 VLM 不是随便找图聊天，而是用不同图片类型拆分能力。先写人工期望答案，后面模型输出时才知道它是看对了、猜对了，还是幻觉式回答。
## Step 06：真实轻量 VLM 实验

已创建脚本：`experiments/vlm-qa/image_qa.py`

最小输入输出：

```text
输入：image path + question text
输出：decode 后的自然语言 answer + runtime / device / memory 记录
```

本次模型：

- model id：`HuggingFaceTB/SmolVLM-256M-Instruct`
- processor：`Idefics3Processor`
- model：`Idefics3ForConditionalGeneration`
- device / dtype：`cuda` / `bfloat16`
- CUDA 显存峰值：约 `855 MB`

结果文件：`experiments/vlm-qa/results/smolvlm_256m_sample_results.jsonl`

三类测试观察：

| 类型 | 人工期望 | 模型回答 | 边界 |
| --- | --- | --- | --- |
| 场景描述 | laptop、notebook、pen、coffee cup 及相对位置 | There is a pen and a coffee cup on the desk. | 漏掉 laptop、notebook 和相对位置 |
| OCR / 截图 | RUN DEMO；READY | RUN DEMO；FOOTER: Day12 VLM QA | 按钮读对，status 读错 |
| 图表理解 | Q4 最高，Q3 最低 | Q4. | 最高值答对，最低值漏答 |

Day12 收口结论：最小 VLM 闭环已经跑通，但轻量模型在细节定位、OCR 文本区分和图表完整比较上仍有明显边界。
