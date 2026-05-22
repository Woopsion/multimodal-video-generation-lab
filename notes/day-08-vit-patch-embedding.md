# Day 8 - ViT Patch Embedding

## 状态

- 日期：2026-05-22
- 状态：已收口
- HTML 主笔记：`docs/day-08-vit-patch-embedding/index.html`

## 完成内容

- 理解图片张量 `image.shape = [B, C, H, W]`。
- 理解 patch、patch_rows / patch_cols、patch_h / patch_w 的区别。
- 理解 `unfold` 如何从高度和宽度取 patch。
- 理解 `permute + reshape` 如何把 patch 网格整理成序列。
- 理解 Patch Embedding 如何用 Linear 把 `patch_dim` 转成 `embed_dim`。
- 理解 Position Embedding、CLS token、GAP 与 class token 的差异。
- 跑通最小 shape demo：`vit_patch_shape_demo.py`。
- 封装最小 `MiniViT` 类：`mini_vit_shape_demo.py`。

## 最终主线

```text
image [B, C, H, W]
-> patches [B, num_patches, patch_dim]
-> patch_embeddings [B, num_patches, embed_dim]
-> tokens [B, 1 + num_patches, embed_dim]
-> vit_input [B, 1 + num_patches, embed_dim]
-> encoder_output [B, 1 + num_patches, embed_dim]
-> cls_output [B, embed_dim]
```

## 当前 demo

```bash
.venv/bin/python experiments/day-08-vit-patch-embedding/vit_patch_shape_demo.py
.venv/bin/python experiments/day-08-vit-patch-embedding/mini_vit_shape_demo.py
```

核心输出：

```text
[1, 1, 4, 4] -> [1, 4, 4] -> [1, 4, 6] -> [1, 5, 6] -> [1, 5, 6]
```

## 后续

- Day9 进入 CLIP 图文对齐。
- 真实预训练 ViT 分类推理后续补充。

