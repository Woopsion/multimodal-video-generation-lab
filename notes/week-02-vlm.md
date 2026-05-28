# Week 2 - CLIP、ViT 与多模态理解

## 当前阶段

- 日期：2026-05-28
- 学习状态：Day8 ViT patch embedding 已收口，Day9 CLIP 图文对齐已收口，下一步进入 Day10 CLIP 检索 demo。
- 主笔记：`docs/week-02-multimodal-understanding/index.html`
- Day8 笔记：`docs/day-08-vit-patch-embedding/index.html`
- Day9 笔记：`docs/day-09-clip-alignment/index.html`

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

## Day9 收口

Day9 采用和 Day8 一样的节奏：先用 HTML 解释概念和数据流，再补最小 shape demo 和真实 CLIP 推理脚本。

已完成到 Step 08：

- CLIP 全称是 Contrastive Language-Image Pre-training。
- CLIP 的核心目标是把图片和文本映射到同一个向量空间。
- Image Encoder 输出 `image_embedding`。
- Text Encoder 输出 `text_embedding`。
- 图文是否匹配，先通过两个向量的相似度判断。
- 图片塔和文本塔内部 shape 可以不同，但最终都要输出 `[B, D]`。
- `D` 是共享向量空间的维度，后续相似度矩阵会依赖这个维度。
- `image_embeddings[b]` 是第 `b` 张图片的整图向量，不是某个 patch 的向量；patch 级向量在 `patch_embeddings` 或 `image_hidden_states` 里。
- ViT 可以用 class token 汇总整图：`encoder_output[:, 0, :] -> [B, D]`。
- ViT 也可以用 mean pooling 汇总整图：`encoder_output[:, 1:, :].mean(dim=1) -> [B, D]`，这里是沿 patch/token 维度求平均，不是把所有数字平均成一个标量。
- `image_embeddings @ text_embeddings.T` 会得到 `[B, B]` 的图文相似度矩阵。
- 矩阵第 `i` 行第 `j` 列表示第 `i` 张图和第 `j` 条文本的匹配分数。
- 如果 batch 按图文对排列，那么对角线就是正确匹配，非对角线就是错误匹配。
- 图片找文本：`CrossEntropyLoss(S, labels)`。
- 文本找图片：`CrossEntropyLoss(S.T, labels)`。
- CLIP loss 通常是两个方向 loss 的平均。
- 梯度会让对角线分数上升，让非对角线分数下降，并继续影响两个 encoder 的参数。
- 最小 demo 已经跑通 `[B, D] -> [B, B] -> labels -> loss -> backward()`。
- Demo 路径：`experiments/day-09-clip-alignment/clip_similarity_demo.py`
- `F.normalize(x, dim=1)` 是对 `[B, D]` 的每一行做 L2 归一化，即除以这一行向量长度，不是减均值除标准差。
- 真实 CLIP 推理 demo 已添加：`experiments/day-09-clip-alignment/clip_real_inference_demo.py`
- 真实 demo 默认本地离线加载 `openai/clip-vit-base-patch32`，本机当前未缓存该模型；需要显式加 `--allow-download` 才会下载。
- 真实推理中，`outputs.image_embeds` 和 `outputs.text_embeds` 是真实模型算出来的共享空间向量，`outputs.logits_per_image` 是图片对文本候选的匹配分数。
- `AutoProcessor` 只负责把原始图片和文本整理成 `pixel_values`、`input_ids`、`attention_mask`。
- `pixel_values` 和 `input_ids` 不是 embedding；真正的 `image_embeds` 和 `text_embeds` 来自 `CLIPModel`。
- `return_tensors="pt"` 表示返回 PyTorch Tensor；`padding=True` 表示把不同长度文本补齐成同一个 batch 矩阵。
- `CLIPModel.forward()` 会分别调用图片路径和文本路径，得到 `image_embeds`、`text_embeds` 后再做 L2 normalize 和相似度矩阵。
- `vision hidden_size = 768` 是视觉塔内部 token 宽度；`projection_dim = 512` 是最终共享图文空间宽度。
- `visual_projection` 把整图向量从 `[num_images, 768]` 映射到 `[num_images, 512]`。
- `text_projection` 把整句向量整理到 `[num_texts, 512]`，即使输入输出维度都为 512，它仍然是可学习线性变换。

Day9 任务已收口。真实 CLIP demo 默认离线，不会自动下载模型；后续如果需要实际概率，可以显式运行带 `--allow-download` 的命令。下一步进入 Day10 CLIP 图文检索 demo。
