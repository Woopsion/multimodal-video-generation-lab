# Week 2 - CLIP、ViT 与多模态理解

## 当前阶段

- 日期：2026-06-29
- 学习状态：Day8 ViT patch embedding 已收口，Day9 CLIP 图文对齐已收口，Day10 CLIP 图文检索已收口，Day11 CLIP 论文阅读已收口，Day12 VLM 图像问答入门已收口；当前进入 Day13 LLaVA 论文阅读。
- 主笔记：`docs/week-02-multimodal-understanding/index.html`
- Day8 笔记：`docs/day-08-vit-patch-embedding/index.html`
- Day9 笔记：`docs/day-09-clip-alignment/index.html`
- Day10 笔记：`docs/day-10-clip-retrieval/index.html`
- Day11 笔记：`docs/day-11-clip-paper-reading/index.html`
- Day12 笔记：`docs/day-12-vlm-image-qa/index.html`
- Day12 状态：`notes/day-12-vlm-image-qa.md`

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

## Day10 收口

Day10 的目标是把 Day9 的 CLIP 相似度矩阵变成可用的检索流程。

当前完成到 Step 06：

- Day10 不重新训练模型，先学习 retrieval 任务的工程视角。
- `query` 是发起搜索的对象，`gallery` 是被搜索的候选库。
- image-to-text retrieval 是给图片找文本。
- text-to-image retrieval 是给文本找图片。
- 核心分数仍然来自 `scores = image_embeddings @ text_embeddings.T`。
- `scores.shape = [num_images, num_texts]` 时，按行排序就是图搜文，按列排序或对 `scores.T` 按行排序就是文搜图。
- `top-k` 只是按相似度从高到低取前 K 个结果，不是新模型也不是新 loss。
- Toy retrieval demo 已添加：`experiments/day-10-clip-retrieval/clip_retrieval_toy_demo.py`
- Toy demo 使用手写图文向量，跑通 `scores -> torch.topk -> ranking -> Recall@K`。
- `topk_values` 是 top-k 分数，`topk_indices` 是 gallery 下标。
- `Recall@K` 检查正确答案是否出现在前 K 个结果里。
- 真实 CLIP retrieval demo 已添加：`experiments/day-10-clip-retrieval/clip_real_retrieval_demo.py`
- 本机可以默认离线加载 `openai/clip-vit-base-patch32`，真实 demo 已跑通 `[4, 512] x [7, 512] -> logits_per_image [4, 7]`。
- 真实 demo 的图搜文前 4 个明确配对样例均能 top-1 命中。
- 当 query 文本没有对应图片时，检索仍会返回 gallery 里相对最像的候选，因此需要阈值或“无答案”判断。
- 默认真实 demo 现在包含 4 个有答案文本和 3 个无答案文本。
- `accept_threshold = 28.0` 在当前教学样例上能接受前 4 个正确匹配，并拒绝 3 个无答案 query。
- `Recall@K` 只统计有标准答案的 query；无答案 query 要单独看 correct reject / false accept。
- Day10 HTML 已记录成功案例与失败边界：`docs/day-10-clip-retrieval/index.html#evaluation`
- 最小图片搜索 Web demo 已添加：`demos/clip-image-search/`
- Web demo 的核心调用链是 `browser query -> /api/search -> CLIPModel.forward() -> logits_per_text [1, 4] -> top-k -> threshold`。
- API 验证：`red square` 返回 `accept`，top-1 是红色方块；`dog on grass` 返回 `reject`，因为候选库里没有狗。
- 生产环境没有 oracle 直接知道候选库里是否有正确答案，只能用分数阈值、top1/top2 margin、reranker、metadata filter 和线上反馈来判断结果是否足够可信。

Day10 的 checklist 已完成。下一步进入 Day11：阅读 CLIP 论文，把代码中的 encoder、projection、similarity matrix、contrastive loss 和论文方法对齐。

## Day11 收口

Day11 的目标是阅读 CLIP 论文 `Learning Transferable Visual Models From Natural Language Supervision`，并把论文方法和 Day9/Day10 已经跑过的代码主线对齐。

完成内容：

- Day11 HTML 主笔记已添加：`docs/day-11-clip-paper-reading/index.html`
- CLIP 原论文 PDF 已下载：`papers/clip/learning-transferable-visual-models-from-natural-language-supervision.pdf`
- 正式论文笔记已添加：`papers/clip.md`
- 已理解 Figure 1：从固定类别预测转向自然语言描述匹配，再迁移到 zero-shot。
- 已理解 Figure 3：`N x N` 相似度矩阵、对角线正样本、in-batch negatives、双向 contrastive loss。
- 已理解 zero-shot classifier：类别 prompt 经过 text encoder 动态构造分类器。
- 已理解实验链路：zero-shot transfer、linear probe、robustness、limitations 分别证明什么。
- 已理解工程边界：CLIP 是开放词表视觉语义匹配器，不是万能视觉逻辑推理器。
- 5 分钟复述检查已完成。

Day11 已收口。当前进入 Day12：VLM 图像问答入门，重点学习视觉特征如何进入语言模型。

## Day12 收口

Day12 的目标是从 CLIP 图文匹配过渡到 VLM 图像问答，重点理解视觉特征如何进入语言模型，并用轻量 VLM 跑通最小本地图像问答。

已完成：

- Day12 HTML 教学入口：`docs/day-12-vlm-image-qa/index.html`
- Day12 状态笔记：`notes/day-12-vlm-image-qa.md`
- 最小 VLM QA 脚本：`experiments/vlm-qa/image_qa.py`
- 三类测试样例：`experiments/vlm-qa/sample_tasks.jsonl`
- SmolVLM-256M 结果：`experiments/vlm-qa/results/smolvlm_256m_sample_results.jsonl`

最终主线：

```text
image path + question
-> processor(image, text)
-> pixel_values / input_ids / attention_mask
-> Idefics3ForConditionalGeneration.generate()
-> decode new tokens
-> answer + runtime boundary
```

Day12 已收口。下一步进入 Day13：阅读 LLaVA 论文，理解 VLM 的两阶段训练和视觉指令数据构造。
## Day13 启动

Day13 的目标是阅读 LLaVA 原论文 `Visual Instruction Tuning`，理解一个 VLM 如何从 CLIP vision encoder、projection matrix 和 Vicuna / LLM 组合成视觉助手。

已创建：

- Day13 HTML 教学入口：`docs/day-13-llava-paper-reading/index.html`
- Day13 状态笔记：`notes/day-13-llava-paper-reading.md`
- LLaVA 论文笔记：`papers/llava.md`

当前只推进 Step 01-02：

```text
image -> CLIP ViT-L/14 -> visual features -> projection matrix -> Vicuna -> answer tokens
```

暂不复现训练，先把论文定位、架构组件和两阶段训练预告讲清楚。
