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
