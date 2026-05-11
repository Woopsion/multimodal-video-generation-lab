# Week 1：基础阶段笔记

## Day 1：整体目标与知识地图

### 1. 30 天学习目标

这 30 天的重点不是只收藏资料，而是形成可运行、可解释、可复盘的学习闭环：

1. 能写基础 PyTorch 训练脚本。
2. 能跑通 CLIP 图文检索。
3. 能跑通 VLM 图片问答。
4. 能理解 Diffusion 和 Stable Diffusion 的基本组件。
5. 能完成一个视频抽帧 + VLM 总结流程。
6. 能尝试本地视频生成，并记录显存、耗时和参数。

### 2. 三个核心方向

#### VLM / MLLM

VLM 是 Vision-Language Model，核心目标是让模型同时处理视觉和语言信息。

常见任务：

- 图片描述
- 图片问答
- OCR 理解
- 图表理解
- 截图理解
- 视频问答

典型结构：

```text
图片/视频 -> 视觉编码器 -> 投影层 -> LLM -> 文本回答
```

你需要重点理解：视觉特征如何变成 LLM 能读懂的 token 或 embedding。

#### Diffusion

Diffusion 是生成模型的一类核心方法，常用于图像生成，也扩展到视频生成。

核心思想：

```text
训练：干净图像 -> 加噪 -> 模型学习预测噪声
推理：随机噪声 -> 逐步去噪 -> 生成图像
```

Stable Diffusion 的常见组件：

- Text Encoder：理解 prompt
- VAE：把图像压缩到 latent 空间，再从 latent 解码回图像
- UNet / DiT：负责去噪
- Scheduler：控制采样步骤

#### Video Generation

Video Generation 是图像生成的进一步扩展，难点不只是单帧质量，还包括时间维度。

核心挑战：

- 主体一致性：同一个人物/物体不能每帧都变样
- 运动合理性：动作要自然
- 文本对齐：视频内容要符合 prompt
- 时序稳定：不能闪烁、漂移、抖动严重
- 显存压力：视频比图像多了时间维度，计算成本更高

常见任务：

- 文生视频：text-to-video
- 图生视频：image-to-video
- 视频编辑：video-to-video
- 视频补全/延长

### 3. 推荐学习顺序

```text
PyTorch 基础
-> Transformer / LLM
-> ViT / CLIP / 多模态对齐
-> Diffusion / Stable Diffusion
-> 视频理解 / 视频生成
-> LoRA / 部署 / 综合 Demo
```

### 4. 本机环境判断

当前机器有约 8GB 显存，适合的学习路线：

- 优先跑 CLIP、ViT、小型 VLM、轻量 Diffusers 推理。
- 图像生成优先使用较低分辨率或轻量模型。
- 视频生成优先使用 ComfyUI 轻量工作流、AnimateDiff、低分辨率/低帧数实验。
- 大模型训练和大型视频生成模型不要一开始硬跑，先理解流程和使用轻量版本。

### 5. Day 1 验收答案

#### VLM 解决什么问题？

VLM 解决“让 AI 看懂视觉内容并用语言交互”的问题。它可以读取图片、截图、图表或视频帧，然后完成描述、问答、OCR、推理、总结等任务。

#### Diffusion 解决什么问题？

Diffusion 解决“从随机噪声生成高质量图像/视频”的问题。它通过学习去噪过程，把噪声一步步变成符合文本或其他条件的视觉内容。

#### Video Generation 解决什么问题？

Video Generation 解决“根据文本、图片或视频条件生成连续动态画面”的问题。相比图像生成，它还要保证帧与帧之间的时间一致性、运动合理性和主体稳定性。

## Day 1 补充：uv、PyTorch 与 MPS

### 1. 为什么使用 uv？

`uv` 用来管理本仓库的 Python 版本、虚拟环境和依赖锁定。相比直接用系统 Python，`uv` 更适合学习仓库：

- 可以把 Python 固定到 3.12。
- 可以把依赖写入 `pyproject.toml` 和 `uv.lock`。
- 可以用 `uv run` 直接运行脚本，不必手动激活 `.venv`。

本仓库采用：

```bash
uv init --bare --python 3.12 --name multimodal-video-generation-lab
uv python pin 3.12
```

### 2. 为什么不用裸 `uv init`？

裸 `uv init` 会生成默认项目骨架，例如 `main.py`、`.gitignore`、`.python-version` 等。当前仓库已经有学习计划、README 和目录规划，因此更适合使用：

```bash
uv init --bare --python 3.12
```

这样只创建 `pyproject.toml`，避免引入无关文件。

### 3. MPS 是什么？

MPS 是 Metal Performance Shaders，是 Apple Mac 上的 GPU 计算后端。

可以这样类比：

```text
CPU  -> 普通处理器
CUDA -> NVIDIA GPU
MPS  -> Apple GPU
```

当前验证结果：

```text
mps: True
mps built: True
```

说明当前 PyTorch 版本支持 MPS，并且这台 Mac 可以使用 Apple GPU。

### 4. linear_regression.py 的作用

`experiments/pytorch-basics/linear_regression.py` 是 Day 2 的第一个 PyTorch 实验。它用于理解：

- 张量如何参与计算。
- `requires_grad=True` 如何开启梯度追踪。
- `loss.backward()` 如何计算梯度。
- `optimizer.step()` 如何更新参数。
- 一个最小训练循环长什么样。

脚本会自动选择设备：

```python
if torch.cuda.is_available():
    return torch.device("cuda")
if torch.backends.mps.is_available():
    return torch.device("mps")
return torch.device("cpu")
```

也就是：

```text
优先 NVIDIA CUDA，其次 Apple MPS，最后 CPU。
```

### 5. 当前进度判断

Day 1 的环境初始化已经从“发现问题”推进到“可运行 PyTorch 基础实验”：

- Python 已固定到 3.12。
- PyTorch 已安装并验证。
- Mac MPS 可用。
- Day 2 线性回归实验脚本已创建。

下一步应正式进入 Day 2 的概念学习：张量、矩阵乘法、广播机制、计算图和自动微分。
