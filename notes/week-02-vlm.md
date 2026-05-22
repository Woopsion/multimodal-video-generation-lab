# Week 2 - CLIP、ViT 与多模态理解

## 当前阶段

- 日期：2026-05-22
- 学习状态：Day8 ViT patch embedding 已收口，下一步进入 Day9 CLIP 图文对齐。
- 主笔记：`docs/week-02-multimodal-understanding/index.html`
- Day8 笔记：`docs/day-08-vit-patch-embedding/index.html`

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

## Day8 收口

Day8 采用“概念 HTML + 最小 shape demo + MiniViT 封装”的方式完成，不训练模型，不加载真实预训练 ViT。

最终掌握链路：

```text
image.shape = [B, C, H, W]
patches.shape = [B, num_patches, C * patch_h * patch_w]
patch_embeddings.shape = [B, num_patches, embed_dim]
tokens.shape = [B, 1 + num_patches, embed_dim]
vit_input.shape = [B, 1 + num_patches, embed_dim]
encoder_output.shape = [B, 1 + num_patches, embed_dim]
cls_output.shape = [B, embed_dim]
```

产出：

- `docs/day-08-vit-patch-embedding/index.html`
- `experiments/day-08-vit-patch-embedding/vit_patch_shape_demo.py`
- `experiments/day-08-vit-patch-embedding/mini_vit_shape_demo.py`
- `experiments/day-08-vit-patch-embedding/README.md`

已理解：

- patch 可以类比文本 token。
- patch embedding 用 Linear 或 Conv2d 把 patch 转成 token vector。
- position embedding 让 patch token 带上位置信息。
- class token 是可学习汇总向量，也可以用 GAP 替代或融合。
- MiniViT 的 `forward()` 只是把 patchify、projection、CLS、position、encoder 按顺序组织起来。

## 暂不推进

- 暂不训练大模型。
- 暂不做完整 VLM 问答。
- 暂不进入 Diffusion。
- 真实预训练 ViT 分类推理后续补充。
