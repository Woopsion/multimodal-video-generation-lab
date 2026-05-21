# Week 2 - CLIP、ViT 与多模态理解

## 当前阶段

- 日期：2026-05-21
- 学习状态：Week2 已启动，Day8 准备进入 ViT patch embedding。
- 主笔记：`docs/week-02-multimodal-understanding/index.html`

## Week2 主线

```text
图片像素
-> patch
-> patch embedding
-> ViT image encoder
-> image embedding

文本
-> tokenizer
-> text encoder
-> text embedding

image embedding 与 text embedding
-> 相似度
-> CLIP 图文对齐 / 检索 / 零样本分类
```

## Day8 起点

先不训练 ViT，也不急着追求分类效果。第一步只做 shape demo：

```text
image.shape = [B, C, H, W]
patches.shape = [B, num_patches, C * patch_h * patch_w]
patch_embeddings.shape = [B, num_patches, embed_dim]
```

目标：

- 理解 patch 为什么可以类比 token。
- 理解 patch embedding 和 token embedding 的相似点。
- 打印并解释图片、patch、embedding、position embedding 的 shape。

## 暂不推进

- 暂不训练大模型。
- 暂不做完整 VLM 问答。
- 暂不进入 Diffusion。
