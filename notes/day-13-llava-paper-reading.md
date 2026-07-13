# Day 13 - LLaVA 论文阅读

> 当前阶段：Day13 已收口。Step 01-06 已完成，下一步进入 Day14 Week2 复盘与 VLM 小助手设计。

## 今日目标

- 从 Day12 的 VLM 推理链路过渡到 LLaVA 的训练和论文主线。
- 理解 LLaVA 为什么不是 CLIP，也不是单纯的 LLM，而是视觉编码器 + 投影层 + LLM 的视觉助手训练路线。
- 掌握视觉指令数据和两阶段训练的关系。

## 主学习入口

- HTML 教学文档：`docs/day-13-llava-paper-reading/index.html`
- 论文笔记：`papers/llava.md`

## 当前最短主线

```text
image
-> CLIP ViT-L/14 vision encoder
-> visual features
-> projection matrix
-> language embedding space
-> Vicuna / LLM
-> answer tokens
```

## Step 01：论文定位

LLaVA 的核心任务不是图文相似度匹配，而是视觉条件下的指令跟随和多轮对话。它的输出是自然语言回答，不是 CLIP 那样的 matching score。

## Step 02：三个组件

| 组件 | 作用 | Day12 对应概念 |
| --- | --- | --- |
| CLIP ViT-L/14 | 把图片编码成视觉特征 | vision encoder |
| Projection matrix | 把视觉特征映射到语言模型词向量空间 | projector |
| Vicuna / LLM | 根据图片上下文和文本指令生成回答 | LLM decoder |

## Step 03：视觉指令数据

普通 caption 数据通常是 `image -> caption`。它能教模型描述图片，但不足以教模型像助手一样按用户问题回答。

原始 LLaVA 的数据生成阶段使用 language-only GPT-4。GPT-4 没有直接看图片像素，而是看到了 caption、物体类别、bounding box 等文本化场景信息，然后生成更像人类对话的数据。真正训练 LLaVA 时，样本仍然绑定真实图片，模型学习 `image + instruction -> answer`。

## Step 04：两阶段训练

### 什么叫冻结

冻结一个模块，不是不用这个模块，而是不更新它的参数。冻结模块仍然参与前向计算。

```text
训练 = forward -> loss -> backward -> optimizer 更新参数
冻结 = forward 仍然执行，但 optimizer 不更新该模块参数
```

### 什么叫只训练 projection matrix

projector 是 CLIP 和 LLM 之间的桥。CLIP 输出视觉特征，projector 把它变成 LLM hidden size 的 image tokens。

```text
visual_features = frozen_clip(image)
image_tokens    = visual_features @ W_proj + b
```

只训练 projection matrix，意思是只更新 `W_proj` 和 `b`，CLIP 和 LLM 的参数不变。

### CC3M 是什么

CC3M 是 Conceptual Captions 3M，是从网页图片和 alt-text 中抽取、清洗出来的大规模图片-caption 数据集。LLaVA Stage 1 使用过滤后的 595K image-caption pairs。

CC3M 样本通常是：

```text
image: 一张真实网页图片
caption: 一句图片描述
```

它适合做初步对齐，但不足以训练复杂视觉问答。

### Stage 1: Feature Alignment Pretraining

目标：让视觉特征能进入语言模型。

```text
1. 输入真实图片
2. frozen CLIP 输出 visual features
3. trainable projector 输出 image tokens
4. frozen LLM 根据 image tokens 生成 caption
5. loss 算在 caption / assistant answer tokens 上
6. optimizer 只更新 projector
```

为什么冻结 CLIP：CLIP 已经有稳定视觉特征，不希望第一阶段把视觉塔改乱。

为什么冻结 LLM：LLM 已经有语言生成能力，第一阶段只需要学习视觉特征如何接入 LLM。

为什么只训练 projector：把问题缩小成一个映射学习问题，先把接口接通。

### Stage 2: Visual Instruction Tuning

目标：让模型学会根据图片回答用户指令。

```text
1. 输入真实图片 + 用户问题
2. frozen CLIP 输出 visual features
3. trainable projector 输出 image tokens
4. trainable LLM 读取 image tokens + question tokens
5. LLM 预测 assistant answer tokens
6. optimizer 更新 projector + LLM，仍然不更新 CLIP
```

为什么要训练 LLM：只训练 projector 只能调整视觉特征如何送进来，但 LLM 还需要学会在注意力计算中使用 image tokens、按用户问题筛选视觉信息、生成符合助手格式的回答。

### loss 算在哪里

图片 token 和用户问题 token 是条件上下文，loss 主要算在 assistant answer tokens 上。否则模型会被训练去预测用户问题，而不是学习回答问题。

## Step 05：CLIP 与 LLaVA 对比

| 对比项 | CLIP | LLaVA |
| --- | --- | --- |
| 任务目标 | 图文表征对齐与匹配 | 视觉条件下的指令跟随与文本生成 |
| 典型输出 | embedding / similarity score | 自回归生成的 answer tokens |
| 训练信号 | 对比学习 loss | assistant answer token prediction loss |
| 典型任务 | 检索、零样本分类、匹配 | 图片描述、视觉问答、视觉对话 |

LLaVA 使用 CLIP vision encoder，但并不等于 CLIP。CLIP 只负责视觉特征提取；projector 负责维度与表示空间对齐；LLM 负责理解用户指令并生成自然语言回答。

## Step 06：5 分钟复述主线

1. LLaVA 的目标是把 CLIP 的视觉能力与 Vicuna 的语言生成能力连接成视觉助手。
2. 架构是 `image -> CLIP -> projector -> LLM -> answer tokens`。
3. GPT-4 在原论文中主要负责生成视觉指令训练数据，不参与 LLaVA 推理。
4. Stage 1 冻结 CLIP 和 LLM，只训练 projector，完成 feature alignment。
5. Stage 2 冻结 CLIP，训练 projector + LLM，完成 visual instruction tuning。
6. 图片和问题作为上下文，loss 主要计算在 assistant answer tokens 上。

## Day13 收口结论

- 已完成论文定位、架构、数据、训练与 CLIP/LLaVA 对比。
- 已产出 HTML 教学文档、Day13 状态笔记和 LLaVA 论文笔记。
- 当前不复现完整 LLaVA 训练，不下载大模型权重。
- 下一步：Day14 Week2 复盘与 VLM 小助手方案设计。
