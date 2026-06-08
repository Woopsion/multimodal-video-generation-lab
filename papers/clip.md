# CLIP - Learning Transferable Visual Models From Natural Language Supervision

## 基本信息

- Paper: Learning Transferable Visual Models From Natural Language Supervision
- Model: CLIP, Contrastive Language-Image Pre-training
- Authors: Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever
- Local PDF: `papers/clip/learning-transferable-visual-models-from-natural-language-supervision.pdf`
- Main HTML note: `docs/day-11-clip-paper-reading/index.html`

## 一句话总结

CLIP 用大规模图文对进行对比预训练，让图片和自然语言描述进入同一个向量空间。训练完成后，它不需要为每个下游分类任务重新训练固定分类头，而是可以把类别写成文本 prompt，用图文相似度完成 zero-shot 分类、图文检索和开放词表识别。

## 背景问题

传统视觉分类通常是：

```text
image -> image_encoder -> hidden state -> fixed classification head -> class logits
```

这种方式的核心限制是类别空间固定。模型最后输出的是训练阶段定义好的 `num_classes` 个类别分数。如果要新增类别，通常至少需要新的分类头参数；如果新类别和原任务差异大，还需要补充数据并微调 encoder。

固定类别监督还有几个问题：

- 闭集分类：类别表之外的概念没有输出位置。
- 标签信息稀疏：单个类别标签通常丢掉颜色、动作、关系、场景等信息。
- 类别 id 没有语义：CrossEntropyLoss 看到的是类别下标，不理解类别名称本身。
- 人工标注成本高：新增任务需要重新定义标签、收集数据、人工标注。
- 多标签和长尾表达弱：一张图往往包含多个对象和关系，单标签表达不完整。

CLIP 要解决的问题是：视觉模型能不能不只学习固定类别，而是直接从自然语言描述中学习开放的视觉概念。

## 核心思想

CLIP 把监督信号从：

```text
image -> class_id
```

改成：

```text
image <-> text
```

自然语言监督的关键不在于“文字更漂亮”，而是监督空间变了。类别 id 是固定枚举的，文本描述是开放、可组合、带语义关系的。比如：

```text
"dog"
"a black dog"
"a black dog standing on grass"
"a black dog catching a frisbee on grass"
```

这些描述可以表达对象、属性、动作、场景和关系。

## Figure 1 主线

Figure 1 可以理解为三段式路线：

1. 传统监督分类：图片进入模型，输出固定类别分数。
2. CLIP 预训练：图片和文本分别编码，训练图文是否匹配。
3. CLIP zero-shot：把下游类别写成 prompt，用文本编码器动态构造分类器。

关键转变是：

```text
固定类别预测 -> 自然语言描述匹配 -> zero-shot transfer
```

CLIP 不是训练时就知道所有下游类别。它训练的是图文匹配能力；下游任务是在测试时通过 prompt 临时组织出来的。

## Figure 3 方法

一个 batch 中有 `N` 个图文对：

```text
image_0 <-> text_0
image_1 <-> text_1
...
image_N-1 <-> text_N-1
```

训练过程：

```text
images -> image_encoder -> image_features
texts  -> text_encoder  -> text_features

image_embeddings = image_features @ W_image
text_embeddings  = text_features  @ W_text

image_embeddings = normalize(image_embeddings)
text_embeddings  = normalize(text_embeddings)

logits = exp(logit_scale) * image_embeddings @ text_embeddings.T
labels = [0, 1, 2, ..., N - 1]

loss_i2t = cross_entropy(logits, labels)
loss_t2i = cross_entropy(logits.T, labels)
loss = (loss_i2t + loss_t2i) / 2
```

其中：

- 对角线是正样本。
- 非对角线是 in-batch negatives。
- image-to-text loss 训练图片找正确文本。
- text-to-image loss 训练文本找正确图片。
- `logit_scale` 或 temperature 控制 softmax 分布的尖锐程度。

## 训练数据

论文使用大规模网络图文对进行训练，核心思路是利用自然语言文本作为弱监督。论文构建了大规模 WebImageText 数据集，规模约为 4 亿图文对。

这个选择有两面：

- 优势：覆盖概念广，天然包含对象、场景、动作、属性和关系，比固定类别标签更开放。
- 风险：网络数据有噪声、偏见和覆盖不均问题，模型会继承一部分数据分布问题。

## Zero-shot 分类

CLIP 做 zero-shot 分类时，不训练新的固定分类头，而是把类别名写成 prompt。

```text
classes = ["cat", "dog", "car"]

prompts = [
  "a photo of a cat",
  "a photo of a dog",
  "a photo of a car",
]

text_embeddings = text_encoder(prompts)
image_embedding = image_encoder(image)
logits = image_embedding @ text_embeddings.T
prediction = argmax(logits)
```

这里的文本向量临时充当分类器权重。类别集合换了，只需要换 prompt。

Prompt engineering 很重要。`"cat"` 通常不如 `"a photo of a cat"`，因为后者更接近训练时的自然语言描述分布。论文还使用 prompt ensemble：给同一个类别写多个模板，编码后平均，降低单一模板带来的偏差。

## 实验证据链

论文实验主要证明：CLIP 训练出来的图文共享空间确实有迁移能力。

### Zero-shot transfer

CLIP 不使用下游训练集标注，也能做分类。论文里一个标志性结果是：最佳 CLIP zero-shot 在 ImageNet 上达到 76.2% top-1，匹配原始监督 ResNet-50 的表现，而且没有用 ImageNet 的 128 万人工标注训练样本训练分类头。

这说明自然语言监督和图文对齐可以在经典视觉分类任务上达到强监督模型级别的迁移效果。

### Linear probe

Linear probe 冻结 CLIP 的 image encoder，只训练一个线性分类器。它回答的问题是：CLIP 的视觉表示本身是否有质量。

论文在 27 个数据集上评估，CLIP 特征在 21 个数据集上超过 Noisy Student EfficientNet-L2。它在 OCR、场景识别、地理定位、动作识别和部分细粒度任务上表现突出。

这说明 CLIP 的 image encoder 不只是会配文本，它学到的视觉表示本身也很强。

### Robustness

论文还看分布迁移，例如 ImageNetV2、ImageNet-A、ImageNet-R、ObjectNet 和 ImageNet Sketch。

核心发现是 zero-shot CLIP 能明显缩小 ImageNet 准确率和分布迁移准确率之间的 gap，最多缩小 75%。这说明 CLIP 不只是为某个固定 benchmark 训练分类头，它的 zero-shot 表现可能更接近真实泛化能力。

## 局限性

CLIP 很强，但不是万能视觉推理模型。

主要限制：

- 计数不稳定：图文整体匹配不要求精确数出数量。
- 空间关系弱：整图 embedding 会压缩局部位置关系。
- 细粒度分类弱：网络 caption 不一定稳定标注具体型号、品种、专业类别。
- 否定和逻辑弱：对比学习不是逻辑证明训练。
- Prompt 敏感：不同模板会改变文本向量和类别边界。
- 偏见与安全风险：网络图文数据可能带入社会偏见和有害关联。

根本原因是：CLIP 被优化成匹配图文整体语义，而不是解析图像中的结构化事实。

## 和 Day9 / Day10 代码的对应关系

| 论文概念 | Day9 / Day10 代码概念 | 解释 |
| --- | --- | --- |
| image encoder | `image_encoder` / CLIP vision tower | 把图片变成整图视觉表示。 |
| text encoder | `text_encoder` / CLIP text tower | 把文本 prompt 或 caption 变成整句文本表示。 |
| projection | `image_projection`, `text_projection` | 把两边特征映射到共享维度。 |
| normalize | `F.normalize(..., dim=1)` | 对每一行向量做 L2 归一化。 |
| similarity matrix | `image_embeddings @ text_embeddings.T` | 得到 `[N, N]` 或 `[num_images, num_texts]` 匹配分数。 |
| contrastive loss | `loss_i2t + loss_t2i` | 同时训练图片找文本和文本找图片。 |
| retrieval | `topk(scores)` | 训练后的共享空间可以直接用于检索。 |
| zero-shot classifier | prompt text embeddings | 类别文本向量临时充当分类器权重。 |

## 5 分钟讲稿

1. 传统视觉模型通常依赖固定类别标签，这限制了开放词表和跨任务迁移。
2. CLIP 改用大规模图文对作为自然语言监督，学习图片和文本是否匹配。
3. 方法上，CLIP 用 image encoder 和 text encoder 分别编码图片和文本，再 projection 到同一个向量空间。
4. 一个 batch 中计算 `N x N` 相似度矩阵，对角线是正样本，非对角线是 in-batch negatives。
5. 训练目标是双向 contrastive loss，同时训练 image-to-text 和 text-to-image。
6. 测试时，CLIP 可以把类别名写成 prompt，用 text encoder 动态生成 zero-shot 分类器。
7. 实验证明 CLIP 的共享空间能做 zero-shot transfer、linear probe 和更稳的分布迁移。
8. 但 CLIP 不擅长精确计数、空间推理、细粒度专业判断、否定逻辑和安全敏感判断。
9. 所以 CLIP 是多模态理解的重要地基，但不是终点；后续 VLM 需要在它之上补充语言生成、指令跟随和推理能力。

## 学习收口判断

能够解释下面两个问题，说明 CLIP 论文主线已经掌握：

1. Figure 1：为什么 CLIP 从固定类别预测转向自然语言描述匹配？
2. Figure 3：CLIP 如何用 `N x N` 相似度矩阵和双向 contrastive loss 训练图文共享空间？
