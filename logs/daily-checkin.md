# 每日学习打卡记录

## Day 1 - 环境与学习仓库初始化

- 日期：2026-05-11
- 今日目标：阅读学习计划第 1-3 章，完成学习目录初始化，确认本地环境，理解 VLM、Diffusion、Video Generation 的基本任务边界。
- 完成内容：
  - 已创建 `notes/`、`papers/`、`experiments/`、`demos/`、`logs/` 目录。
  - 已确认本机 Python 版本、GPU 信息和关键 Python 包状态。
  - 已阅读学习计划的学习目标、整体知识地图和 PyTorch 基础阶段要求。
- 环境检查：
  - Python：3.14.0
  - GPU：NVIDIA GeForce RTX 5060 Laptop GPU，显存约 8GB
  - Driver：577.00
  - PyTorch：未安装
  - Transformers：未安装
  - Diffusers：未安装
  - Accelerate：未安装
  - PEFT：未安装
- 代码/笔记路径：
  - 学习清单：`docs/30-day-executable-checklist.md`
  - 今日记录：`logs/daily-checkin.md`
- 遇到的问题：
  - 当前 Python 是 3.14.0，深度学习生态对 Python 3.14 支持可能不稳定。
  - PyTorch、Transformers、Diffusers 等关键库尚未安装。
- 解决办法：
  - 建议后续创建 Python 3.11 或 3.12 的虚拟环境。
  - 再安装 CUDA 版 PyTorch、Transformers、Diffusers、Accelerate、PEFT。
- 显存/耗时记录：
  - 当前 GPU 总显存约 8GB，适合优先学习 VLM 小模型、CLIP、Diffusers 小规模推理和轻量视频生成方案。
- Day 1 核心理解：
  - VLM：让模型理解图像/视频等视觉输入，并用语言回答、描述、推理。
  - Diffusion：从噪声逐步去噪生成图像或视频，是现代图像/视频生成的重要基础。
  - Video Generation：根据文本、图片或已有视频生成新视频，重点处理时间一致性、运动质量和主体一致性。
- 明天计划：
  - 进入 Day 2：学习 PyTorch 张量、矩阵乘法、广播机制、自动微分。
  - 准备一个适合深度学习的 Python 3.11/3.12 环境。
### 追加整理

- 已整理 Day 1 阶段学习记录：`notes/day-01-learning-record.md`
- 记录内容包括：Day 1 目标、学习路线理解、VLM/Diffusion/Video Generation 区分、PyTorch 角色、目录结构设计、Latent Diffusion、latent、张量、VAE 等概念。

### 环境补充与 Day 2 准备

- 已使用 `uv` 初始化项目环境：
  - `uv init --bare --python 3.12 --name multimodal-video-generation-lab`
  - `uv python pin 3.12`
- 已生成并保留：
  - `.python-version`
  - `pyproject.toml`
  - `uv.lock`
- 已安装 PyTorch 基础阶段依赖：
  - `torch`
  - `torchvision`
  - `torchaudio`
  - `numpy`
  - `matplotlib`
  - `pillow`
  - `tqdm`
  - `scikit-learn`
  - `jupyter`
  - `ipykernel`
- 当前 Mac 环境验证：
  - Python：3.12.13
  - PyTorch：2.11.0
  - CUDA：不可用
  - MPS：可用
- MPS 理解：
  - MPS 是 Apple Mac 上的 GPU 计算后端，类似 NVIDIA 平台上的 CUDA。
  - 当前机器可以用 `torch.device("mps")` 跑部分 PyTorch 计算。
- 已补齐仓库目录：
  - `papers/`
  - `experiments/`
  - `demos/`
- 已创建 Day 2 实验：
  - `experiments/day-02-pytorch-autograd-linear-regression/linear_regression.py`
  - `experiments/day-02-pytorch-autograd-linear-regression/README.md`
  - `experiments/day-02-pytorch-autograd-linear-regression/outputs/linear_regression_result.png`
- 已新增环境说明：
  - `docs/uv-environment-setup.md`
- 关于 uv 缓存的结论：
  - 正常终端中直接使用默认 uv 缓存即可。
  - 只有遇到权限问题时，才临时使用 `UV_CACHE_DIR=.uv-cache`。
## Day 2 复现记录 - Windows 本机

- 日期：2026-05-12
- 目标：在当前 Windows 电脑上复现另一台电脑完成的 Day 2 PyTorch 自动微分与线性回归实验。
- 环境检查：
  - `.venv\Scripts\python.exe` 可用。
  - Python：3.12.11
  - PyTorch：2.11.0+cpu
  - CUDA：不可用
  - MPS：不可用
  - 当前复现实验使用 CPU。
- 运行命令：
  - `.\.venv\Scripts\python.exe experiments/day-02-pytorch-autograd-linear-regression/linear_regression.py`
  - `.\.venv\Scripts\python.exe experiments/day-02-pytorch-autograd-linear-regression/gradient_debug.py`
- 主实验结果：
  - 设备：CPU
  - 自动微分示例：`y = x^2 + 3x + 1`，当 `x = 2` 时，`y = 11`，`dy/dx = 7`。
  - 线性回归最终 loss：约 `0.146679`
  - 学到的函数：`y = 2.007x + 1.032`
  - 输出图片：`experiments/day-02-pytorch-autograd-linear-regression/outputs/linear_regression_result.png`
- 梯度调试结果：
  - 单样本 `x = 2, y = 5`，初始 `weight = 1, bias = 0`。
  - `pred = 2`，`error = -3`，`loss = 9`。
  - `loss.backward()` 后：`weight.grad = -12`，`bias.grad = -6`。
  - `optimizer.step()` 后：`weight = 1.6`，`bias = 0.3`。
  - 连续两次 `backward()` 且不执行 `zero_grad()` 时，梯度会累积：`weight.grad = -24`，`bias.grad = -12`。
- 需要注意：
  - `uv run` 在当前机器上仍遇到权限问题，直接使用 `.venv\Scripts\python.exe` 可以运行。
  - 主实验保存图像时出现 Matplotlib cache 权限警告，但不影响实验结果生成。
- 今日理解：
  - `loss.backward()` 负责计算梯度。
  - `optimizer.step()` 负责根据梯度更新参数。
  - `optimizer.zero_grad()` 必须在每次反向传播前清空旧梯度，否则梯度会累积。
## Day 3 - MNIST 完整训练脚本

- 日期：2026-05-12
- 今日目标：完成一个包含训练、验证、checkpoint、加载和单张图片推理的 MNIST 分类脚本。
- 完成内容：
  - 新增实验目录：`experiments/day-03-mnist-classifier/`
  - 实现训练脚本：`experiments/day-03-mnist-classifier/train.py`
  - 补充实验说明：`experiments/day-03-mnist-classifier/README.md`
  - 补充学习笔记：`notes/day-03-mnist-classifier.md`
  - 更新 `.gitignore`，忽略 MNIST 原始数据目录 `experiments/*/data/`
- 运行命令：
  - `.\.venv\Scripts\python.exe experiments/day-03-mnist-classifier/train.py --epochs 3 --batch-size 256`
  - `.\.venv\Scripts\python.exe experiments/day-03-mnist-classifier/train.py --infer-only --sample-index 0`
- 训练结果：
  - epoch 01：`train_loss 0.3686`，`val_loss 0.2010`，`val_acc 0.9397`
  - epoch 02：`train_loss 0.1668`，`val_loss 0.1356`，`val_acc 0.9603`
  - epoch 03：`train_loss 0.1167`，`val_loss 0.1121`，`val_acc 0.9651`
- 推理结果：
  - sample index：`0`
  - true label：`7`
  - prediction：`7`
  - confidence：`0.9979`
- 输出文件：
  - `experiments/day-03-mnist-classifier/outputs/best_mnist_mlp.pt`
  - `experiments/day-03-mnist-classifier/outputs/training_curves.png`
  - `experiments/day-03-mnist-classifier/outputs/sample_prediction.png`
- 今日理解：
  - `Dataset` 提供单条样本，`DataLoader` 组织 batch。
  - MNIST 图片进入模型时形状是 `[batch_size, 1, 28, 28]`。
  - `CrossEntropyLoss` 适合多分类任务。
  - 完整训练脚本需要包含训练、验证、保存、加载和推理。

## Day 4 - CNN 与 CIFAR-10 概念学习阶段总结

- 日期：2026-05-13
- 今日目标：从 Day 3 的 MNIST MLP 过渡到 Day 4 的 CIFAR-10 CNN，优先理解 CNN 基础概念，而不是急着跑训练脚本。
- 完成内容：
  - 整理 Day 4 学习笔记：`notes/day-04-cnn-cifar10.md`
  - 初步建立 CNN 在 AI / 机器学习 / 深度学习 / 神经网络体系中的位置。
  - 对比 MNIST 灰度图 `[batch, 1, 28, 28]` 与 CIFAR-10 RGB 图 `[batch, 3, 32, 32]`。
  - 理解 RGB 图片如果 Flatten 会得到 `3 * 32 * 32 = 3072` 个数，但会破坏空间结构。
  - 理解卷积核是可学习的小型图案探测器，一个卷积核产生一张特征图。
  - 理解 `out_channels=32` 表示 32 个卷积核，因此输出 32 张特征图。
  - 推导卷积输出尺寸公式：`floor((N + 2P - K) / S) + 1`。
  - 理解 `padding=1` 是在四周补一圈 0，让 `3x3` 卷积后宽高保持不变。
  - 理解 `stride` 是卷积核每次滑动的步幅。
  - 理解 `BatchNorm2d(C)` 会对每个通道分别统计 `batch * height * width` 个值。
  - 理解 `MaxPool2d(2)` 是对卷积输出后的每张特征图分别做 `2x2` 最大池化。
- 当前理解：
  - CNN 的核心不是一开始把图片压平，而是先保留空间结构，在二维图像上提取局部特征。
  - `Conv2d` 主要改变通道数/特征图数量，`MaxPool2d` 主要缩小每张特征图的空间尺寸。
  - `BatchNorm2d` 负责稳定每个通道的特征分布，使训练更平稳。
  - CNN 是后续理解 ViT、Diffusion、UNet 和视频生成模型的重要基础。
- 明天/下一步计划：
  - 手算一个最小卷积例子：`1` 张 `1` 通道 `4x4` 图片，`1` 个 `3x3` 卷积核，`padding=1`，`stride=1`，再接 `MaxPool2d(2)`。
  - 再逐行理解 `experiments/day-04-cifar10-cnn/train.py` 的 CNN 模型结构与训练流程。

## Day 5 - Transformer Encoder Shape Demo 与最终笔记收口

- 日期：2026-05-19
- 今日目标：完成 Transformer Encoder Block 核心骨架理解，并将 Day5 HTML 整理为最终学习笔记。
- 完成内容：
  - 完成最终学习笔记：`docs/day-05-transformer-encoder/index.html`
  - 完成 shape demo：`experiments/day-05-transformer-encoder/encoder_shape_demo.py`
  - 补充实验说明：`experiments/day-05-transformer-encoder/README.md`
  - 更新学习状态索引：`notes/day-05-transformer-encoder.md`
  - 更新 30 天执行清单中的 Day5 完成状态。
- 当前脚本覆盖：
  - token embedding + position embedding
  - Q/K/V projection
  - scaled dot-product attention
  - Multi-Head split / merge
  - output projection
  - residual connection
  - LayerNorm
  - Feed Forward Network
  - EncoderBlock 最终输出 shape
- 验证命令：
  - `.venv/bin/python experiments/day-05-transformer-encoder/encoder_shape_demo.py`
- 最终 shape：
  - `encoder_input.shape = [2, 5, 4]`
  - `ffn_hidden.shape = [2, 5, 16]`
  - `encoder_output.shape = [2, 5, 4]`
- 今日理解：
  - Self-Attention 负责 token 之间交换信息。
  - Multi-Head 让多个 head 各自计算关系，再合并回完整 token 向量。
  - output projection 在 shape 不变的情况下混合多个 head 的输出。
  - residual connection 保留原路径，LayerNorm 稳定每个 token 的 D 维数值。
  - FFN 逐 token 做 `D -> 4D -> D` 的非线性加工。
- 下一步计划：
  - 不再扩写 Day5 概念笔记。
  - 后续进入工程封装：把当前散开的 forward 操作整理成可复用的 `EncoderBlock` 类。

## Day 6 - LLM 基础与推理流程启动

- 日期：2026-05-19
- 今日目标：延续 Day5 的 HTML 学习方式，先建立 LLM 推理流程的直观主线。
- 完成内容：
  - 新增 Day6 主学习笔记：`docs/day-06-llm-inference/index.html`
  - 新增 Day6 状态笔记：`notes/day-06-llm-inference.md`
  - 明确 Day6 第一阶段只学习推理流程，不提前完成真实模型加载和训练相关内容。
- 当前理解重点：
  - `prompt` 是原始文本。
  - tokenizer 把文本切成 token，再映射成 `input_ids`。
  - `input_ids` 是整数 id 序列，不是 token 向量。
  - embedding 会把 token id 查表变成向量。
  - Causal LM 按从左到右的方式预测下一个 token。
  - logits 是词表所有 token 的原始分数。
- 下一步计划：
  - 围绕 Day6 HTML 的 Part 1，继续写一个不加载真实模型的 shape demo。

### Day 6 Part 2 - LLM 推理 shape demo

- 新增实验目录：`experiments/day-06-llm-inference/`
- 新增 demo：`experiments/day-06-llm-inference/llm_inference_shape_demo.py`
- 新增说明：`experiments/day-06-llm-inference/README.md`
- 已更新 HTML 主笔记：`docs/day-06-llm-inference/index.html`
- 当前 demo 覆盖：
  - prompt 到 tokens
  - tokens 到 `input_ids`
  - `input_ids` 到 token vectors
  - position vectors
  - hidden states
  - `lm_head`
  - logits
  - `next_token_logits`
  - softmax 概率
  - greedy 选择 `next_token_id`
  - append 后 decode
- 验证命令：
  - `.venv/bin/python experiments/day-06-llm-inference/llm_inference_shape_demo.py`

### Day 6 Part 3 - Sampling 采样策略

- 新增 demo：`experiments/day-06-llm-inference/sampling_strategy_demo.py`
- 已更新 HTML 主笔记：`docs/day-06-llm-inference/index.html`
- 已更新实验说明：`experiments/day-06-llm-inference/README.md`
- 当前学习重点：
  - greedy decoding：永远选择 logit 最大的 token。
  - temperature：调整概率分布的尖锐程度。
  - top_p：保留累计概率达到阈值的候选集合，再重新归一化采样。
  - max_new_tokens：限制最多追加多少个新 token，不直接控制质量。
  - `<eos>`：生成结束符，可以让生成提前停止。
- 验证命令：
  - `.venv/bin/python experiments/day-06-llm-inference/sampling_strategy_demo.py`

### Day 6 Part 4 - KV Cache

- 新增 demo：`experiments/day-06-llm-inference/kv_cache_shape_demo.py`
- 已更新 HTML 主笔记：`docs/day-06-llm-inference/index.html`
- 已更新 Day6 状态笔记：`notes/day-06-llm-inference.md`
- 已更新实验说明：`experiments/day-06-llm-inference/README.md`
- 当前学习重点：
  - KV Cache 保存历史 token 的 Key / Value。
  - Prefill 阶段一次性处理 prompt，并保存每层 K/V。
  - Decode 阶段每次只计算最新 token 的 Q/K/V。
  - 不用 cache 时 attention score shape 是 `[B, H, S, S]`。
  - 使用 cache 时 decode 阶段 score shape 是 `[B, H, 1, T_total]`。
  - KV Cache 用空间换时间，会增加显存占用。
- 验证命令：
  - `.venv/bin/python experiments/day-06-llm-inference/kv_cache_shape_demo.py`

### Day 6 Part 5 - Transformers 真实模型推理

- 新增 demo：`experiments/day-06-llm-inference/transformers_real_generation_demo.py`
- 本机可选使用本地 pretrained 模型目录：`experiments/day-06-llm-inference/pretrained/sshleifer-tiny-gpt2/`，该目录不提交到 git。
- 新增模型文件检查脚本：`experiments/day-06-llm-inference/inspect_pretrained_model_files.py`
- 已更新 HTML 主笔记：`docs/day-06-llm-inference/index.html`
- 当前学习重点：
  - `AutoTokenizer.from_pretrained(...)` 加载 tokenizer 文件。
  - `AutoModelForCausalLM.from_pretrained(...)` 加载 causal language model 权重。
  - `encoded = tokenizer(prompt, return_tensors="pt")` 得到 `input_ids` 和 `attention_mask`。
  - `model(**encoded, use_cache=True)` 返回 `logits` 和 `past_key_values`。
  - `model.generate(...)` 内部仍然是 forward、选 token、append、cache 的循环。
- 验证命令：
  - `.venv/bin/python experiments/day-06-llm-inference/transformers_real_generation_demo.py`

### Day 6 Part 6 - 真实生成参数与 Day6 收口

- 新增 demo：`experiments/day-06-llm-inference/generation_controls_demo.py`
- 已将参数控制运行报告追加到 HTML 主笔记。
- 已更新 Day6 状态笔记：`notes/day-06-llm-inference.md`
- 已更新 30 天执行清单中的 Day6 完成状态。
- 运行命令：
  - `.venv/bin/python experiments/day-06-llm-inference/generation_controls_demo.py --device cpu --prompt "Deep learning is"`
- 本次运行输入：
  - `input_ids = [29744, 4673, 318]`
  - `tokens = ['Deep', 'Ġlearning', 'Ġis']`
  - `eos_token_id = 50256`
  - `pad_token_id = 50256`
- 结果观察：
  - greedy baseline 会重复生成 `factors`。
  - `max_new_tokens=4` 会更早停止，但选择规则没有改变。
  - 低温采样更保守，高温采样更发散。
  - `top_k` 固定候选数量，`top_p` 按累计概率动态决定候选集合。
  - `repetition_penalty` 能缓解重复，但不能提升 tiny-gpt2 的语言能力。
  - 本次所有 case 都因达到 `max_new_tokens` 停止，没有由 `eos_token_id` 提前停止。
- Day6 收口判断：
  - LLM 推理主线已完成，可以进入 Week 1 复盘。
  - Instruction Tuning、LoRA、QLoRA 属于后续微调专题，暂不塞进 Day6。

## Day 7 - 第一周复盘收口

- 日期：2026-05-21
- 今日目标：收口 Week1 基础阶段，确认训练 / 推理 / logits / loss / 梯度 / checkpoint / LLM 推理主线已经连起来。
- 完成内容：
  - 新增 Week1 HTML 复盘页：`docs/week-01-foundation-review/index.html`
  - 更新 Week1 复盘笔记：`notes/week-01-foundation.md`
  - 更新 30 天清单中的 Day7 完成状态。
  - 新增 Week2 学习入口：`docs/week-02-multimodal-understanding/index.html`
- 本周收口主线：
  - 数据或文本先变成张量 / token id。
  - 模型 forward 得到 logits。
  - 训练阶段用 loss 和 backward 计算梯度。
  - `optimizer.step()` 用梯度更新参数。
  - checkpoint 保存训练后的权重。
  - 推理阶段固定参数，只做 forward 和输出选择。
- 后续观察项：
  - CNN 卷积核梯度。
  - Transformer Q/K/V 参数梯度。
  - attention weight 与 token probability 的区别。
  - KV Cache 工程实现。
  - LoRA / QLoRA 微调。
- 下一步计划：
  - 进入 Week2：CLIP、ViT 与多模态理解。
  - Day8 从 ViT patch embedding shape demo 开始。

## Day 8 - ViT Patch Embedding 收口

- 日期：2026-05-22
- 今日目标：理解图片如何变成 Transformer 可处理的 patch token 序列。
- 完成内容：
  - 新增 Day8 HTML 主笔记：`docs/day-08-vit-patch-embedding/index.html`
  - 新增 shape demo：`experiments/day-08-vit-patch-embedding/vit_patch_shape_demo.py`
  - 新增 MiniViT demo：`experiments/day-08-vit-patch-embedding/mini_vit_shape_demo.py`
  - 新增 Day8 状态笔记：`notes/day-08-vit-patch-embedding.md`
- 当前学习重点：
  - `unfold` 从高度和宽度切 patch。
  - `permute + reshape` 把 patch 网格整理成 `[B, num_patches, patch_dim]`。
  - Patch Embedding 用 Linear 把 `patch_dim` 转成 `embed_dim`。
  - Position Embedding 让 patch token 带上位置信息。
  - CLS token 是可学习汇总向量，也可以用 GAP 作为替代或补充。
  - MiniViT 的 `forward()` 只是把 patchify、projection、CLS、position、encoder 串起来。
- 验证命令：
  - `.venv/bin/python experiments/day-08-vit-patch-embedding/vit_patch_shape_demo.py`
  - `.venv/bin/python experiments/day-08-vit-patch-embedding/mini_vit_shape_demo.py`
- 下一步计划：
  - Day9 进入 CLIP 图文对齐。

## Day 9 - CLIP 图文对齐收口

- 日期：2026-05-24
- 今日目标：理解 CLIP 如何把图片和文本映射到同一个向量空间，并用相似度矩阵完成图文对齐。
- 完成内容：
  - 新增 Day9 HTML 主笔记：`docs/day-09-clip-alignment/index.html`
  - 新增 toy 相似度 demo：`experiments/day-09-clip-alignment/clip_similarity_demo.py`
  - 新增真实 CLIP forward demo：`experiments/day-09-clip-alignment/clip_real_inference_demo.py`
  - 更新 Week2 状态笔记：`notes/week-02-vlm.md`
  - 更新 30 天执行清单中的 Day9 完成状态。
- 当前学习重点：
  - Image Encoder 输出 `image_embedding`，Text Encoder 输出 `text_embedding`。
  - 两边内部 hidden size 可以不同，但最终 projection 后必须进入同一个 `projection_dim`。
  - `image_embeddings @ text_embeddings.T` 得到 `[B, B]` 相似度矩阵。
  - 对角线是 batch 内正确图文对，非对角线是负样本。
  - CLIP contrastive loss 是 image-to-text 和 text-to-image 两个方向的 CrossEntropyLoss 平均。
  - `dLoss / dS = p - y` 不是直接更新分数表，而是通过链式法则回传到 encoder 和 projection 参数。
- 验证命令：
  - `.venv/bin/python experiments/day-09-clip-alignment/clip_similarity_demo.py`
  - `.venv/bin/python experiments/day-09-clip-alignment/clip_real_inference_demo.py`
- 下一步计划：
  - Day10 进入 CLIP 图文检索，把相似度矩阵变成 top-k ranking。

## Day 10 - CLIP 图文检索收口

- 日期：2026-05-24
- 今日目标：把 Day9 的 CLIP 相似度矩阵变成可运行的 retrieval workflow。
- 完成内容：
  - 新增 Day10 HTML 主笔记：`docs/day-10-clip-retrieval/index.html`
  - 新增 toy retrieval demo：`experiments/day-10-clip-retrieval/clip_retrieval_toy_demo.py`
  - 新增真实 CLIP retrieval demo：`experiments/day-10-clip-retrieval/clip_real_retrieval_demo.py`
  - 新增本地图片搜索 Web demo：`demos/clip-image-search/`
  - 更新 Week2 总览页：`docs/week-02-multimodal-understanding/index.html`
  - 更新 Week2 状态笔记：`notes/week-02-vlm.md`
  - 更新 30 天执行清单中的 Day10 完成状态和最低完成标准。
- 当前学习重点：
  - Retrieval 的核心对象是 query、gallery、score、top-k、ranking。
  - 文本搜图时，`logits_per_text.shape = [num_texts, num_images]`。
  - 图片搜文本时，`logits_per_image.shape = [num_images, num_texts]`。
  - Recall@K 衡量正确答案是否进入 top-k。
  - 生产环境不会天然知道候选库是否有正确答案，只能用阈值、top1/top2 margin、reranker、metadata filter、用户反馈和人工复核降低误检。
  - 阈值拒答不是证明“没有答案”，而是表达“当前候选库里没有足够可信的答案”。
- 真实 demo 观察：
  - `outputs.image_embeds.shape = (4, 512)`
  - `outputs.text_embeds.shape = (7, 512)`
  - `outputs.logits_per_image.shape = (4, 7)`
  - `outputs.logits_per_text.shape = (7, 4)`
  - 阈值示例：`threshold = 28.0`
  - 匹配案例：red square、blue circle、green triangle、noodles。
  - 拒答案例：dog on grass、city street at night、person riding a horse。
- 验证命令：
  - `.venv/bin/python experiments/day-10-clip-retrieval/clip_retrieval_toy_demo.py`
  - `.venv/bin/python experiments/day-10-clip-retrieval/clip_real_retrieval_demo.py`
  - `.venv/bin/python demos/clip-image-search/app.py`
- 下一步计划：
  - Day11 阅读 CLIP 论文，把 encoder、projection、similarity matrix、contrastive loss 和论文方法逐项对齐。

## Day 11 - CLIP 论文阅读收口

- 日期：2026-06-08
- 今日目标：阅读 CLIP 论文主线，理解自然语言监督、Figure 1、Figure 3、zero-shot、实验和局限性。
- 完成内容：
  - 新增 Day11 HTML 主笔记：`docs/day-11-clip-paper-reading/index.html`
  - 下载 CLIP 原论文 PDF：`papers/clip/learning-transferable-visual-models-from-natural-language-supervision.pdf`
  - 新增正式论文笔记：`papers/clip.md`
  - 更新 Week2 总览页：`docs/week-02-multimodal-understanding/index.html`
  - 更新 Week2 状态笔记：`notes/week-02-vlm.md`
  - 更新 30 天执行清单中的 Day11 完成状态。
- 当前学习重点：
  - Figure 1 展示训练范式迁移：固定类别预测 -> 自然语言描述匹配 -> zero-shot transfer。
  - Figure 3 展示预训练方法：双 encoder、projection、normalize、`N x N` 相似度矩阵、双向 contrastive loss。
  - Natural language supervision 的监督信号来自 image-text pair，不是固定 class id。
  - Zero-shot classifier 由 prompt 文本向量动态构造，不是固定分类头。
  - 实验证据链包括 zero-shot transfer、linear probe、robustness 和 limitations。
  - CLIP 是开放词表视觉语义匹配器，不是万能视觉逻辑推理器。
- 验收：
  - 5 分钟复述检查已完成。
  - 能解释 CLIP 论文背景、方法、zero-shot、实验和局限性。
- 下一步计划：
  - Day12 进入 VLM 图像问答入门，学习视觉特征如何进入语言模型。

## Day 12 - VLM 图像问答入门启动

- 日期：2026-06-29
- 今日目标：从 CLIP 图文匹配过渡到 VLM 图像问答，循序渐进理解视觉特征如何进入语言模型。
- 文档更新：
  - 新增 Day12 HTML 教学入口：`docs/day-12-vlm-image-qa/index.html`
  - 新增 Day12 状态笔记：`notes/day-12-vlm-image-qa.md`
  - 更新 README、Week2 总览页、Week2 状态笔记和 30 天 checklist。
- 当前学习重点：
  - 区分 CLIP 的相似度匹配和 VLM 的自然语言生成。
  - 理解 VLM 的基本结构：vision encoder、projector、LLM。
  - 理解图片不是直接进入 LLM，而是先变成 visual tokens，再对齐到 LLM hidden size。
- 暂不推进：
  - 暂不下载远程模型权重。
  - 暂不做 OCR、图表、场景三类完整评测。
- 下一步计划：
  - 完成 Day12 Step 01：能解释 `image -> vision encoder -> visual tokens -> projector -> LLM -> answer` 这条主线。
## Day 12 - VLM 图像问答收口

- 日期：2026-06-29
- 已完成：
  - 理解 CLIP 与 VLM 的任务差异：匹配 vs 生成。
  - 拆解 VLM 主线：vision encoder -> projector -> LLM context -> answer tokens。
  - 补充理解 Q/K/V 的并行计算、causal mask 和答案 token 自回归生成顺序。
  - 设计场景描述、OCR / 截图、图表理解三类测试任务。
  - 创建真实轻量 VLM 推理脚本：`experiments/vlm-qa/image_qa.py`。
  - 使用 `HuggingFaceTB/SmolVLM-256M-Instruct` 跑通本地 CUDA 图像问答实验。
- 实验记录：
  - processor：`Idefics3Processor`
  - model：`Idefics3ForConditionalGeneration`
  - device / dtype：`cuda` / `bfloat16`
  - 显存峰值：约 `855 MB`
  - 结果文件：`experiments/vlm-qa/results/smolvlm_256m_sample_results.jsonl`
- 边界观察：
  - 场景描述漏掉 laptop、notebook 和相对位置。
  - OCR 读对按钮 `RUN DEMO`，但把 status 误读成 footer。
  - 图表任务答对最高值 `Q4`，但没有回答最低值 `Q3`。
- Day12 已收口。下一步进入 Day13：LLaVA 论文阅读。
## Day 13 - LLaVA 论文阅读启动

- 日期：2026-06-30
- 今日目标：从 Day12 的 VLM 推理过渡到 LLaVA 的论文和训练主线。
- 文档更新：
  - 新增 Day13 HTML 教学入口：`docs/day-13-llava-paper-reading/index.html`
  - 新增 Day13 状态笔记：`notes/day-13-llava-paper-reading.md`
  - 新增 LLaVA 论文笔记入口：`papers/llava.md`
- 当前学习重点：
  - 区分 CLIP 图文匹配与 LLaVA 视觉指令跟随。
  - 理解 LLaVA 三个组件：CLIP ViT-L/14、projection matrix、Vicuna。
  - 先知道两阶段训练：feature alignment 与 visual instruction tuning。
- 暂不推进：
  - 暂不复现 LLaVA 训练。
  - 暂不下载 LLaVA 大模型权重。

## Day 13 - LLaVA 论文阅读收口

- 日期：2026-07-13
- 完成内容：
  - 完成 LLaVA 论文定位、三组件架构和视觉指令数据学习。
  - 详细理解 Stage 1 feature alignment 与 Stage 2 visual instruction tuning。
  - 明确冻结模块仍参与前向计算，只是不更新参数。
  - 对比 CLIP 的 embedding / similarity 目标与 LLaVA 的 answer token generation 目标。
  - 完成 5 分钟论文复述主线与 Day13 验收总结。
- 文档产出：
  - `docs/day-13-llava-paper-reading/index.html`
  - `notes/day-13-llava-paper-reading.md`
  - `papers/llava.md`
- Day13 已收口。下一步进入 Day14：Week2 复盘与 VLM 小助手方案设计。
