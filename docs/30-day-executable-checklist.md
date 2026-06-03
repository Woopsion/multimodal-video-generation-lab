# 多模态与视频生成 30 天可执行打卡清单

> 来源：根据 `multimodal-video-generation-learning-plan.md` 拆解。  
> 用法：每天完成「学习」「实践」「产出」三类任务，并在当天末尾填写复盘。  
> 建议节奏：每天 1.5-3 小时；周末可集中补实验与整理笔记。

---

## 使用规则

- [ ] 每天至少留下一个可检查产出：代码、笔记、截图、实验记录或 demo。
- [ ] 所有笔记统一放到 `notes/`，论文记录放到 `papers/`，实验代码放到 `experiments/`，demo 放到 `demos/`。
- [ ] 每个实验记录显存占用、耗时、模型名称、参数设置和失败原因。
- [ ] 每周至少完成一篇总结：本周学到什么、遇到什么问题、下周解决什么。
- [ ] 不追求一天学完所有材料，优先保证能复现、能解释、能记录。

---

## 推荐目录结构

```text
notes/
  week-01-foundation.md
  week-02-vlm.md
  week-03-diffusion.md
  week-04-video-generation.md
papers/
  paper-reading-template.md
  clip.md
  llava.md
  latent-diffusion.md
experiments/
  pytorch-mnist/
  clip-retrieval/
  vlm-qa/
  diffusion-inference/
  video-understanding/
demos/
  vlm-visual-assistant/
  video-understanding-assistant/
  local-video-generation-lab/
logs/
  daily-checkin.md
  gpu-benchmark.md
```

---

## 每日复盘模板

```markdown
## Day X - 标题

- 日期：
- 今日目标：
- 完成内容：
- 代码/笔记路径：
- 遇到的问题：
- 解决办法：
- 显存/耗时记录：
- 明天计划：
```

---

## Week 1：深度学习、PyTorch 与 Transformer 基础

### Day 1：环境与学习仓库初始化

- [x] 学习：阅读学习计划第 1-3 章，明确 30 天目标。
- [x] 实践：创建 `notes/`、`papers/`、`experiments/`、`demos/`、`logs/` 目录。
- [x] 实践：确认 Python、CUDA、PyTorch、Transformers、Diffusers 是否可用。
- [x] 产出：写 `logs/daily-checkin.md` 的 Day 1 记录。
- [x] 验收：能说清楚 VLM、Diffusion、Video Generation 分别解决什么问题。

### Day 2：PyTorch 张量与自动微分

- [x] 学习：张量、矩阵乘法、广播、计算图、自动微分。
- [x] 实践：写一个张量运算 notebook 或脚本，包含 `backward()` 示例。
- [x] 实践：手动实现一个线性回归训练循环。
- [x] 产出：`experiments/day-02-pytorch-autograd-linear-regression/linear_regression.py`。
- [x] 验收：能解释 `requires_grad`、`loss.backward()`、`optimizer.step()` 的作用。

### Day 3：MNIST 完整训练脚本

- [x] 学习：Dataset、DataLoader、训练/验证循环、checkpoint。
- [x] 实践：用 PyTorch 训练 MNIST 分类器。
- [x] 实践：保存最佳模型，并写一个单张图片推理函数。
- [x] 产出：`experiments/day-03-mnist-classifier/train.py` 和训练曲线截图。
- [x] 验收：脚本包含训练、验证、保存、加载和推理。

### Day 4：CNN 与 CIFAR-10

- [x] 学习：卷积、池化、BatchNorm、Dropout、数据增强。
- [x] 实践：训练一个 CIFAR-10 CNN 分类器。
- [x] 实践：记录 batch size 对显存和速度的影响。
- [x] 产出：`experiments/day-04-cifar10-cnn/train.py` 和 `logs/gpu-benchmark.md`。
- [x] 验收：能解释 CNN 为什么适合图像任务。

### Day 5：Transformer Encoder 手写版

- [x] 学习：Embedding、位置编码、Self-Attention、Multi-Head Attention、FFN、Residual、LayerNorm。
- [x] 实践：手写一个最小 Transformer Encoder block 的 forward shape demo。
- [x] 实践：用固定 token id 和随机初始化参数检查输入输出 shape。
- [x] 产出：`experiments/day-05-transformer-encoder/encoder_shape_demo.py`。
- [x] 产出：`docs/day-05-transformer-encoder/index.html` 作为最终学习笔记。
- [x] 验收：能画出 Transformer Encoder block 的数据流。

### Day 6：LLM 基础与推理流程

- [x] 学习：Tokenization、Causal LM、KV Cache。
- [x] 实践：使用 Transformers 加载一个小型文本模型做生成。
- [x] 实践：比较不同 `temperature`、`top_p`、`max_new_tokens` 的输出差异。
- [x] 产出：`experiments/day-06-llm-inference/transformers_real_generation_demo.py`。
- [x] 产出：`experiments/day-06-llm-inference/generation_controls_demo.py`。
- [x] 产出：`docs/day-06-llm-inference/index.html` 作为最终学习笔记。
- [x] 验收：能解释 tokenizer、prompt、decode、sampling 的关系。
- [ ] 后续专题：Instruction Tuning、LoRA/QLoRA。

### Day 7：第一周复盘

- [x] 整理：合并本周笔记到 `notes/week-01-foundation.md`。
- [x] 整理：列出后续观察概念。
- [x] 实践：复查 Day 3 checkpoint / 训练曲线与 Day 6 推理结果作为复盘证据。
- [x] 产出：`docs/week-01-foundation-review/index.html` 作为第一周复盘页。
- [x] 验收：能解释基础 PyTorch 训练脚本的 forward、loss、backward、step、checkpoint、inference 主线。

---

## Week 2：CLIP、ViT 与多模态理解模型

### Day 8：ViT 与图像 Patch 表示

- [x] 学习：ViT 的 patch embedding、class token、position embedding。
- [x] 实践：打印并记录图像输入、patch、embedding、CLS token、position embedding、encoder output 的 shape。
- [x] 实践：封装最小 `MiniViT` 类，解释 `forward()` 数据流与可学习参数。
- [x] 产出：`docs/day-08-vit-patch-embedding/index.html`。
- [x] 产出：`experiments/day-08-vit-patch-embedding/vit_patch_shape_demo.py`。
- [x] 产出：`experiments/day-08-vit-patch-embedding/mini_vit_shape_demo.py`。
- [x] 验收：能解释 CNN 特征与 ViT patch token 的区别，以及图片如何变成 Transformer token 序列。
- [ ] 后续补充：用现成 ViT 模型做图像分类推理。

### Day 9：CLIP 图文对齐

- [x] 学习：CLIP 的 image encoder、text encoder、contrastive loss。
- [x] 实践：跑通 CLIP 图文相似度计算，并解释 `[B, D] -> [B, B] -> loss` 主线。
- [x] 实践：用 toy demo 输出图文相似度矩阵，并补充真实 CLIP 推理脚本。
- [x] 产出：`docs/day-09-clip-alignment/index.html`。
- [x] 产出：`experiments/day-09-clip-alignment/clip_similarity_demo.py`。
- [x] 产出：`experiments/day-09-clip-alignment/clip_real_inference_demo.py`。
- [x] 验收：能解释为什么 CLIP 可以做零样本分类和图文检索。

### Day 10：CLIP 图文检索 Demo

- [x] 学习：向量归一化、余弦相似度、top-k 检索。
- [x] 实践：用 toy demo 跑通 `scores -> topk -> ranking -> Recall@K`。
- [x] 产出：`docs/day-10-clip-retrieval/index.html`。
- [x] 产出：`experiments/day-10-clip-retrieval/clip_retrieval_toy_demo.py`。
- [x] 实践：构建一个小型图片库，输入文本返回 top-3 图片。
- [x] 产出：`experiments/day-10-clip-retrieval/clip_real_retrieval_demo.py`。
- [x] 实践：记录成功案例和失败案例各 3 个。
- [x] 产出：`demos/clip-image-search/`。
- [x] 验收：能运行一个最小图文检索 demo。

### Day 11：阅读 CLIP 论文

- [ ] 学习：阅读 CLIP 论文摘要、方法图、训练数据、实验结果。
- [ ] 记录：用论文模板整理背景、方法、损失函数、优缺点。
- [ ] 实践：把 Day 9 的代码和论文方法对应起来。
- [ ] 产出：`papers/clip.md`。
- [ ] 验收：能用 5 分钟讲清楚 CLIP 的核心思想。

### Day 12：VLM 图像问答入门

- [ ] 学习：VLM 的视觉编码器、投影层、LLM、视觉指令微调。
- [ ] 实践：选择 Qwen-VL、MiniCPM-V 或其他轻量 VLM 做图片问答。
- [ ] 实践：测试 OCR、图表理解、场景描述三类图片。
- [ ] 产出：`experiments/vlm-qa/image_qa.py`。
- [ ] 验收：能解释 VLM 如何把图片信息交给 LLM。

### Day 13：阅读 LLaVA 论文

- [ ] 学习：阅读 LLaVA 的模型结构、训练流程和视觉指令数据构造。
- [ ] 记录：总结 pretrain、instruction tuning 两阶段的区别。
- [ ] 实践：对比 CLIP 与 LLaVA 的任务目标差异。
- [ ] 产出：`papers/llava.md`。
- [ ] 验收：能解释“视觉编码器 + 投影层 + LLM”的基本范式。

### Day 14：第二周复盘与 VLM 小助手设计

- [ ] 整理：合并本周笔记到 `notes/week-02-vlm.md`。
- [ ] 设计：规划 `vlm-visual-assistant` 的输入、输出和模型选择。
- [ ] 实践：整理 10 张测试图片，包含截图、图表、普通照片。
- [ ] 产出：`demos/vlm-visual-assistant/design.md`。
- [ ] 验收：能完成一个图像问答 demo 的技术方案。

---

## Week 3：Diffusion、Stable Diffusion 与图像生成

### Day 15：扩散模型基本原理

- [ ] 学习：前向加噪、反向去噪、噪声预测、采样步数。
- [ ] 学习：理解 DDPM、DDIM 的基本区别。
- [ ] 记录：画出扩散模型训练和推理流程图。
- [ ] 产出：`notes/week-03-diffusion.md` 中的 Diffusion 原理笔记。
- [ ] 验收：能解释“训练时预测噪声，推理时逐步去噪”。

### Day 16：Latent Diffusion 与 Stable Diffusion

- [ ] 学习：VAE、latent space、UNet、text encoder、cross attention。
- [ ] 实践：用 Diffusers 跑通一次文生图推理。
- [ ] 实践：记录 prompt、seed、steps、guidance scale、分辨率。
- [ ] 产出：`experiments/diffusion-inference/txt2img.py`。
- [ ] 验收：能解释 Stable Diffusion 的核心组件分工。

### Day 17：阅读 Latent Diffusion 论文

- [ ] 学习：阅读 LDM 论文摘要、方法图、VAE 压缩、cross-attention 条件控制。
- [ ] 记录：总结为什么在 latent 空间扩散更高效。
- [ ] 实践：把 Diffusers pipeline 的组件和论文结构对应起来。
- [ ] 产出：`papers/latent-diffusion.md`。
- [ ] 验收：能解释 latent diffusion 与 pixel diffusion 的区别。

### Day 18：SDXL 与参数实验

- [ ] 学习：SDXL 相比 SD 1.5 的基本变化。
- [ ] 实践：固定 seed，比较不同 steps、guidance scale、negative prompt 的效果。
- [ ] 实践：生成至少 12 张对比图。
- [ ] 产出：`experiments/sdxl-parameter-study/README.md`。
- [ ] 验收：能根据效果调节基础生成参数。

### Day 19：ComfyUI 工作流入门

- [ ] 学习：ComfyUI 的节点、模型、采样器、VAE、工作流 JSON。
- [ ] 实践：搭建一个最小文生图 workflow。
- [ ] 实践：保存 workflow 和生成结果。
- [ ] 产出：`demos/comfyui-txt2img/workflow.json`。
- [ ] 验收：能说明每个核心节点的输入输出。

### Day 20：LoRA 与轻量微调概念

- [ ] 学习：LoRA、QLoRA、DreamBooth、PEFT 的适用场景。
- [ ] 记录：整理图像 LoRA 训练所需数据、caption、显存、超参数。
- [ ] 实践：不要求训练，先写一份训练准备清单。
- [ ] 产出：`notes/lora-training-preparation.md`。
- [ ] 验收：能判断什么时候该用 LoRA，而不是全量微调。

### Day 21：第三周复盘

- [ ] 整理：完善 `notes/week-03-diffusion.md`。
- [ ] 实践：选 3 个最佳 prompt 重新生成并记录参数。
- [ ] 总结：列出 Diffusion 仍不理解的 5 个问题。
- [ ] 产出：第三周复盘和图像生成结果集。
- [ ] 验收：能跑通并解释一次 Stable Diffusion 推理。

---

## Week 4：视频理解、视频生成与综合 Demo

### Day 22：视频抽帧与视频理解

- [ ] 学习：视频帧、fps、clip、关键帧、时间维度信息。
- [ ] 实践：写脚本对短视频抽帧，统一尺寸和 fps。
- [ ] 实践：把抽帧图片送入 VLM，总结视频内容。
- [ ] 产出：`experiments/video-understanding/extract_frames.py`。
- [ ] 验收：能完成“视频 -> 帧 -> VLM 总结”的流程。

### Day 23：视频生成模型概览

- [ ] 学习：Video Diffusion、3D VAE、时空注意力、DiT、Flow Matching。
- [ ] 学习：了解 Stable Video Diffusion、CogVideoX、Wan、HunyuanVideo、Open-Sora 的定位。
- [ ] 记录：整理各模型任务类型、硬件需求、许可证、仓库地址。
- [ ] 产出：`notes/video-generation-model-map.md`。
- [ ] 验收：能区分文生视频、图生视频、视频编辑三类任务。

### Day 24：本地视频生成推理

- [ ] 学习：选择一个本地可跑的视频生成方案，如 ComfyUI + AnimateDiff、CogVideoX 或 Wan 轻量版本。
- [ ] 实践：跑通官方或社区最小推理样例。
- [ ] 实践：记录显存、耗时、分辨率、帧数、采样步数。
- [ ] 产出：`experiments/local-video-generation/README.md` 和首个生成视频。
- [ ] 验收：能生成一个本地视频样例。

### Day 25：视频数据处理小数据集

- [ ] 学习：视频切片、抽帧、caption、数据清洗。
- [ ] 实践：收集 10-20 个短视频或使用公开视频样例。
- [ ] 实践：统一 fps、分辨率、时长，并为每个 clip 写 caption。
- [ ] 产出：`experiments/video-dataset-mini/metadata.csv`。
- [ ] 验收：得到一个可用于后续训练或评估的小型视频数据集。

### Day 26：视频生成评估

- [ ] 学习：了解 VBench、T2V-CompBench 的评估思路。
- [ ] 实践：人工评估 10 个生成视频。
- [ ] 记录：从文本对齐、运动质量、主体一致性、画面质量四个维度打分。
- [ ] 产出：`experiments/video-evaluation/evaluation_table.md`。
- [ ] 验收：能形成一张清晰的视频生成评估表。

### Day 27：综合 Demo 设计

- [ ] 设计：选择一个综合 demo 主题，如“上传图片/视频 -> 理解内容 -> 生成描述或视频”。
- [ ] 设计：确定 Gradio 或 FastAPI 作为界面方案。
- [ ] 设计：画出系统流程图，明确 VLM、Diffusion/Video Generation 的调用边界。
- [ ] 产出：`demos/multimodal-video-agent-demo/design.md`。
- [ ] 验收：得到一个可实现的 demo 设计文档。

### Day 28：实现最小综合 Demo

- [ ] 实践：接入 VLM 做图片或视频理解。
- [ ] 实践：接入图像生成或视频生成推理脚本。
- [ ] 实践：提供一个最小可用 UI 或命令行入口。
- [ ] 产出：`demos/multimodal-video-agent-demo/`。
- [ ] 验收：能完成一次端到端流程演示。

### Day 29：论文与开源项目追踪表

- [ ] 整理：列出至少 10 篇论文，标注任务、方法、是否已读、核心贡献。
- [ ] 整理：列出至少 10 个开源项目，标注用途、难度、硬件需求、许可证。
- [ ] 产出：`papers/paper-tracker.md`。
- [ ] 产出：`notes/open-source-project-tracker.md`。
- [ ] 验收：能明确下一阶段最值得深入的 3 篇论文和 3 个项目。

### Day 30：最终复盘与下月计划

- [ ] 总结：整理 30 天所有代码、笔记、截图和 demo。
- [ ] 总结：写出 3 个最强收获、3 个薄弱点、3 个下月目标。
- [ ] 规划：选择下月主线：VLM 微调、图像 LoRA、视频生成部署或视频 LoRA。
- [ ] 产出：`notes/30-day-review.md`。
- [ ] 产出：`docs/next-4-week-plan.md`。
- [ ] 验收：形成一份可展示的 30 天学习报告。

---

## 最低完成标准

如果时间有限，至少完成以下 10 项：

- [ ] 一个 PyTorch 训练脚本。
- [x] 一个 CLIP 图文检索 demo。
- [ ] 一个 VLM 图片问答 demo。
- [ ] 一个视频抽帧 + VLM 总结 demo。
- [ ] 一个 Stable Diffusion 或 SDXL 推理脚本。
- [ ] 一个 ComfyUI 图像生成 workflow。
- [ ] 一个本地视频生成样例。
- [ ] 三篇论文笔记：CLIP、LLaVA、Latent Diffusion。
- [ ] 一份开源项目追踪表。
- [ ] 一份 30 天复盘报告。

---

## 每周验收口径

- [ ] Week 1：能独立写基础训练脚本，并解释 Transformer block。
- [ ] Week 2：能跑通 CLIP 检索和 VLM 图片问答，并讲清楚多模态对齐。
- [ ] Week 3：能跑通图像生成推理，并解释 Stable Diffusion 组件。
- [ ] Week 4：能完成视频理解、视频生成样例和一个综合 demo 设计或原型。

---

## 建议优先级

1. 先做 VLM：部署成本低、反馈快、应用广。
2. 再做 Diffusion：它是理解视频生成的基础。
3. 最后做视频生成：硬件、数据和调参成本更高。

推荐入门组合：

- 多模态理解：Qwen-VL、MiniCPM-V 或同类轻量 VLM。
- 图像生成：Diffusers + SDXL。
- 视频生成：ComfyUI + AnimateDiff、CogVideoX 或 Wan 轻量版本。
- 微调：PEFT + LoRA。
- 部署：Gradio 或 FastAPI。
