# LLaVA - Visual Instruction Tuning

## 基本信息

- 论文：Visual Instruction Tuning
- 模型简称：LLaVA，Large Language-and-Vision Assistant
- 作者：Haotian Liu、Chunyuan Li、Qingyang Wu、Yong Jae Lee
- 会议：NeurIPS 2023 Oral
- arXiv：https://arxiv.org/abs/2304.08485
- 项目页：https://llava-vl.github.io/
- GitHub：https://github.com/haotian-liu/LLaVA
- 数据集：https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K
- CC3M 论文：https://aclanthology.org/P18-1238/

## 一句话总结

LLaVA 把 CLIP 视觉编码器和 Vicuna 语言模型连接起来，通过投影层做视觉-语言特征对齐，再用视觉指令数据训练模型，让它能围绕图片进行问答和对话。

## 架构主线

```text
image
-> CLIP ViT-L/14 vision encoder
-> visual features
-> projection matrix
-> language embedding space
-> Vicuna / LLM
-> generated answer
```

## Step 03：视觉指令数据

LLaVA 论文指出，多模态 instruction-following 数据相对缺乏。直接把 `image-caption` 扩展成“请描述图片 -> caption”虽然便宜，但问题和回答都不够多样，也缺少深入推理。

论文采用 GPT-assisted visual instruction data generation：先用 caption、物体类别、bounding box 等文本化信息表示图像内容，再让 language-only GPT-4 / ChatGPT 基于这些文本信息生成视觉指令数据。生成的数据仍然和真实图像绑定，用于训练 LLaVA 学习 `image + instruction -> answer`。

三类数据：conversation、detailed description、complex reasoning。论文和项目页给出的规模是 158K unique language-image instruction-following samples，包括 58K conversations、23K detailed description、77K complex reasoning。

## Step 04：两阶段训练

### 冻结与训练

冻结模块仍然参与前向计算，只是不被优化器更新参数。训练模块会根据 loss 的梯度更新参数。

| 模块 | Stage 1 | Stage 2 |
| --- | --- | --- |
| CLIP vision encoder | 冻结 | 冻结 |
| Projection matrix / layer | 训练 | 训练 |
| Vicuna / LLM | 冻结 | 训练 |

### Stage 1: Pre-training for Feature Alignment

- 数据：过滤后的 CC3M image-caption pairs，论文中为 595K。
- CC3M：Conceptual Captions 3M，从网页图片和 alt-text 中清洗得到的大规模图片-caption 数据集。
- 格式：单轮图片描述任务，答案是原始 caption。
- 冻结：CLIP vision encoder 和 LLM。
- 训练：projection matrix。
- 目标：让图像特征对齐到预训练 LLM 的词向量空间。
- 直觉：先训练一个兼容 frozen LLM 的 visual tokenizer。

### Stage 2: Fine-tuning End-to-End / Visual Instruction Tuning

- 数据：LLaVA-Instruct-158K；ScienceQA 是另一个使用场景。
- 冻结：CLIP vision encoder。
- 训练：projection layer 和 LLM。
- 目标：让模型从“能接收图片”变成“能围绕图片遵循用户指令回答”。
- 注意：论文称 end-to-end，但不代表所有模块都训练；CLIP vision encoder 仍然冻结。

### loss

训练时使用 LLM 原本的自回归目标，但 loss 主要作用在 assistant answer tokens 上。图片 tokens 和用户问题 tokens 作为条件上下文。

```text
context = image_tokens + user_question_tokens
label   = assistant_answer_tokens
loss    = answer prediction loss
```

## 暂不展开

- 暂不复现 LLaVA 训练。
- 暂不下载 LLaVA 大模型权重。
- 暂不比较 LLaVA-1.5、LLaVA-NeXT 等后续版本。
- 暂不深挖具体 GPT-4 prompt 模板和 COCO 数据筛选细节。
- 暂不背诵训练超参数，只需知道它们属于工程实现细节。