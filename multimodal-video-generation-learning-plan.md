# 多模态大模型与视频生成模型学习路线

> 版本日期：2026-05-09  
> 适用对象：希望系统学习多模态模型、视频生成模型、论文研究、开源项目、本地部署与轻量微调的初学者到进阶学习者。  
> 目标周期：长期路线 3-6 个月；文末附 30 天入门计划。

---

## 1. 学习目标

本计划的最终目标不是只会“跑 demo”，而是逐步建立以下能力：

1. 理解多模态大模型的核心结构：视觉编码器、投影层、LLM、跨模态对齐、视觉指令微调。
2. 理解视频生成模型的底层原理：扩散模型、Latent Diffusion、DiT、3D VAE、时空注意力、Flow Matching。
3. 能读懂主流论文：看懂模型图、训练流程、损失函数、数据构造、评估指标与消融实验。
4. 能运行主流开源项目：LLaVA、Qwen-VL、InternVL、MiniCPM-V、Diffusers、ComfyUI、CogVideoX、Wan、HunyuanVideo、Open-Sora。
5. 能在本地部署模型：包括多模态理解模型和视频生成模型。
6. 能做轻量训练与调优：LoRA、QLoRA、DreamBooth、PEFT、视频 LoRA、小数据集微调。
7. 能追踪前沿动态：arXiv、Hugging Face、GitHub、技术报告、benchmark、model card。
8. 具备工程与研究素养：实验记录、数据清洗、显存估算、模型许可证、安全伦理和复现能力。

---

## 2. 整体知识地图

可以把这个领域拆成五层：

```text
应用层：VQA / OCR / 视频问答 / 文生视频 / 图生视频 / 视频编辑 / 数字人 / Agent
模型层：VLM / MLLM / Diffusion / DiT / Video Diffusion / Flow Matching
表示层：Token / Embedding / Vision Token / Latent / Patch / Frame / Spatio-temporal Feature
训练层：Pretrain / SFT / RLHF / DPO / LoRA / QLoRA / Caption / Dataset Filtering
工程层：PyTorch / Transformers / Diffusers / PEFT / Accelerate / DeepSpeed / vLLM / ComfyUI
```

学习顺序建议：

1. 先学深度学习和 PyTorch。
2. 再学 Transformer 和 LLM。
3. 再学 CLIP、ViT、多模态对齐。
4. 再学 Diffusion、Stable Diffusion、DiT。
5. 最后进入视频生成、视频理解、训练部署与前沿论文。

---

## 3. 阶段一：深度学习与 PyTorch 基础

### 3.1 必须掌握的概念

- 张量、矩阵乘法、广播机制。
- 前向传播、反向传播、自动微分。
- 损失函数：MSE、Cross Entropy、Contrastive Loss。
- 优化器：SGD、Adam、AdamW。
- 正则化：Dropout、Weight Decay、LayerNorm、BatchNorm。
- 训练循环：Dataset、DataLoader、model.train、model.eval。
- 混合精度：FP32、FP16、BF16。
- GPU 显存：batch size、activation、optimizer state、checkpoint。

### 3.2 推荐资源

- 动手学深度学习：https://zh.d2l.ai/
- PyTorch 官方教程：https://pytorch.org/tutorials/
- Stanford CS231n：https://cs231n.stanford.edu/
- Stanford CS224n：https://web.stanford.edu/class/cs224n/

### 3.3 必做实践

- 用 PyTorch 训练 MNIST 分类器。
- 用 PyTorch 训练 CIFAR-10 分类器。
- 手写一个简单 MLP、CNN、Transformer Encoder。
- 保存和加载 checkpoint。
- 使用 TensorBoard 或 Weights & Biases 记录训练曲线。

### 3.4 阶段验收标准

完成本阶段后，应能独立写出一个完整训练脚本，包括数据加载、模型定义、训练、验证、保存模型和推理。

---

## 4. 阶段二：Transformer 与 LLM 基础

### 4.1 核心概念

- Tokenization。
- Embedding。
- Position Encoding / RoPE。
- Self-Attention。
- Multi-Head Attention。
- Feed Forward Network。
- Residual Connection。
- LayerNorm。
- Decoder-only Transformer。
- Causal Language Modeling。
- KV Cache。
- Instruction Tuning。
- LoRA / QLoRA。
- 量化：INT8、INT4、GPTQ、AWQ、GGUF。

### 4.2 必读论文

1. Attention Is All You Need  
   https://arxiv.org/abs/1706.03762
2. BERT  
   https://arxiv.org/abs/1810.04805
3. GPT-3  
   https://arxiv.org/abs/2005.14165
4. LoRA  
   https://arxiv.org/abs/2106.09685
5. QLoRA  
   https://arxiv.org/abs/2305.14314

### 4.3 推荐开源项目

- Hugging Face Transformers：https://github.com/huggingface/transformers
- Hugging Face PEFT：https://github.com/huggingface/peft
- TRL：https://github.com/huggingface/trl
- vLLM：https://github.com/vllm-project/vllm
- llama.cpp：https://github.com/ggerganov/llama.cpp

### 4.4 必做实践

- 用 Transformers 加载一个 0.5B-7B 的开源语言模型。
- 对一个小型问答数据集做 LoRA 微调。
- 用 bitsandbytes 进行 4-bit 量化加载。
- 用 vLLM 或 llama.cpp 部署本地推理服务。

---

## 5. 阶段三：视觉模型与图文多模态

### 5.1 多模态模型基本结构

典型视觉语言模型可以抽象为：

```text
图片/视频帧
  -> Vision Encoder，例如 ViT、CLIP、SigLIP、EVA-CLIP
  -> Projector / Q-Former / Adapter
  -> 视觉 Token
  -> Large Language Model
  -> 文本回答
```

关键问题：

- 图像如何切成 patch？
- patch 如何变成 embedding？
- 视觉 token 如何进入 LLM？
- 图文如何做对齐？
- 为什么需要视觉指令微调？
- 视频理解为什么通常先抽帧？

### 5.2 核心概念

- CNN 与 ViT。
- CLIP 对比学习。
- 图文检索。
- Image Captioning。
- Visual Question Answering。
- OCR / Document Understanding。
- Visual Instruction Tuning。
- Projector / MLP Connector。
- Q-Former。
- 高分辨率切图与动态分辨率。
- 视频抽帧、时序建模、帧级上下文压缩。

### 5.3 必读论文

1. ViT: An Image is Worth 16x16 Words  
   https://arxiv.org/abs/2010.11929
2. CLIP  
   https://arxiv.org/abs/2103.00020
3. BLIP-2  
   https://arxiv.org/abs/2301.12597
4. LLaVA  
   https://arxiv.org/abs/2304.08485
5. LLaVA-1.5  
   https://arxiv.org/abs/2310.03744
6. Qwen-VL  
   https://arxiv.org/abs/2308.12966
7. Qwen2-VL  
   https://arxiv.org/abs/2409.12191
8. Qwen2.5-VL Technical Report  
   https://arxiv.org/abs/2502.13923

### 5.4 推荐开源项目

- LLaVA：https://github.com/haotian-liu/LLaVA
- LLaVA-NeXT：https://github.com/LLaVA-VL/LLaVA-NeXT
- Qwen2.5-VL：https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- InternVL：https://github.com/OpenGVLab/InternVL
- MiniCPM-V：https://github.com/OpenBMB/MiniCPM-V

### 5.5 必做实践

- 用 CLIP 做图文检索。
- 用 Qwen2.5-VL 或 LLaVA 做图片问答。
- 用 VLM 做 OCR、图表理解、截图理解。
- 把视频抽成多帧后送给 VLM，总结视频内容。
- 准备一个小型图片问答数据集并做 LoRA 微调。

---

## 6. 阶段四：Diffusion 与图像生成基础

### 6.1 为什么要先学图像生成

视频生成模型的大量思想来自图像生成。Stable Video Diffusion、AnimateDiff、CogVideoX、Wan、HunyuanVideo、Open-Sora 等都与 Diffusion、Latent Diffusion、DiT 或 Flow Matching 强相关。

### 6.2 核心概念

- 正向扩散：逐步给数据加噪。
- 反向去噪：模型学习从噪声恢复数据。
- Noise Prediction / Score Prediction / Velocity Prediction。
- DDPM / DDIM。
- Scheduler。
- Classifier-Free Guidance。
- VAE 与 Latent Space。
- U-Net。
- DiT：Diffusion Transformer。
- 文本条件：CLIP Text Encoder、T5 Text Encoder。
- ControlNet。
- IP-Adapter。
- LoRA。

### 6.3 必读论文

1. DDPM  
   https://arxiv.org/abs/2006.11239
2. DDIM  
   https://arxiv.org/abs/2010.02502
3. Latent Diffusion Models  
   https://arxiv.org/abs/2112.10752
4. Classifier-Free Diffusion Guidance  
   https://arxiv.org/abs/2207.12598
5. DiT  
   https://arxiv.org/abs/2212.09748
6. ControlNet  
   https://arxiv.org/abs/2302.05543

### 6.4 推荐开源项目

- Diffusers：https://github.com/huggingface/diffusers
- ComfyUI：https://github.com/comfyanonymous/ComfyUI
- Stable Diffusion WebUI：https://github.com/AUTOMATIC1111/stable-diffusion-webui
- kohya_ss：https://github.com/bmaltais/kohya_ss

### 6.5 必做实践

- 用 Diffusers 跑 Stable Diffusion 或 SDXL。
- 调整 prompt、negative prompt、CFG、steps、scheduler、seed。
- 用 ComfyUI 搭建一个文生图工作流。
- 用 20-100 张图训练一个 LoRA。
- 用 ControlNet 做姿态、深度、边缘控制。

---

## 7. 阶段五：视频生成模型

### 7.1 视频生成的核心难点

相对图像生成，视频生成多了时间维度，因此难点明显增加：

- 时间一致性：主体不能闪烁、变形或身份漂移。
- 运动建模：动作要自然，不能只做图像平移。
- 长视频生成：上下文长度和显存压力很大。
- 物理一致性：碰撞、光照、遮挡、镜头运动要合理。
- 可控性：首尾帧、相机运动、动作轨迹、风格、角色一致性。
- 数据成本：高质量视频 caption 数据非常昂贵。
- 评估困难：主观质量、运动质量、文本对齐、安全性都需要评估。

### 7.2 核心概念

- Text-to-Video。
- Image-to-Video。
- Video-to-Video。
- First-Last Frame to Video。
- Video Inpainting / Editing。
- Latent Video Diffusion。
- 3D VAE / Causal 3D VAE。
- Temporal Attention。
- Spatial-Temporal Attention。
- Video DiT。
- Flow Matching / Rectified Flow。
- Frame Interpolation。
- Video Super Resolution。
- Motion LoRA。
- Camera Control。

### 7.3 必读论文与技术报告

1. Video Diffusion Models  
   https://arxiv.org/abs/2204.03458
2. Make-A-Video  
   https://arxiv.org/abs/2209.14792
3. Imagen Video  
   https://arxiv.org/abs/2210.02303
4. Tune-A-Video  
   https://arxiv.org/abs/2212.11565
5. AnimateDiff  
   https://arxiv.org/abs/2307.04725
6. Stable Video Diffusion  
   https://arxiv.org/abs/2311.15127
7. VideoPoet  
   https://arxiv.org/abs/2312.14125
8. Sora Technical Report  
   https://openai.com/research/video-generation-models-as-world-simulators
9. CogVideoX 项目  
   https://github.com/THUDM/CogVideo
10. HunyuanVideo 项目  
    https://github.com/Tencent-Hunyuan/HunyuanVideo
11. Open-Sora 项目  
    https://github.com/hpcaitech/Open-Sora
12. Wan-Video 项目  
    https://github.com/Wan-Video

### 7.4 重点开源模型

| 模型/项目 | 方向 | 适合学习内容 |
| --- | --- | --- |
| Stable Video Diffusion | 图生视频 | Latent video diffusion 入门 |
| AnimateDiff | 动作模块 / 视频扩展 | Motion module、ComfyUI 工作流 |
| CogVideoX | 文生视频 / 图生视频 | 3D VAE、Video DiT、Diffusers 部署 |
| HunyuanVideo | 大规模视频基础模型 | 大模型视频生成架构与高质量推理 |
| Wan | 文生视频 / 图生视频 / 编辑 | 开源视频生成与多任务视频生成能力 |
| Open-Sora | 开源训练框架 | 数据、训练、checkpoint、复现实验 |

### 7.5 必做实践

- 用 ComfyUI 跑 AnimateDiff 或 Stable Video Diffusion。
- 用 Diffusers 跑 CogVideoX 的推理脚本。
- 尝试 Wan 的轻量版本或量化版本。
- 对同一 prompt 比较不同模型的运动质量、主体一致性和文本对齐。
- 准备 20-100 个短视频 clip，完成 caption、切片、抽帧和质量过滤。
- 尝试一个视频 LoRA 或 Motion LoRA 微调实验。

---

## 8. 本地部署路线

### 8.1 多模态理解模型部署

推荐顺序：

1. Qwen2.5-VL-3B / 7B。
2. MiniCPM-V。
3. LLaVA-1.5 / LLaVA-NeXT。
4. InternVL。

推荐工具：

- Transformers：最适合研究、调试、看源码。
- vLLM：适合服务化和高吞吐推理。
- llama.cpp：适合 GGUF 量化模型。
- Ollama：适合快速体验本地模型。
- Gradio：适合快速做交互 demo。
- FastAPI：适合封装 API 服务。

### 8.2 视频生成模型部署

推荐顺序：

1. ComfyUI + AnimateDiff。
2. ComfyUI + Stable Video Diffusion。
3. Diffusers + CogVideoX。
4. Wan 轻量版本。
5. HunyuanVideo 或 Open-Sora 大模型版本。

### 8.3 显存建议

| 显存 | 建议任务 |
| --- | --- |
| 8GB | 小型 VLM 量化推理、ComfyUI 轻量工作流、图像 LoRA 很吃力 |
| 12GB | 7B VLM 4-bit 推理、部分轻量视频生成 |
| 16GB | Qwen2.5-VL-7B 量化、CogVideoX 部分量化推理 |
| 24GB | 比较适合 LoRA、图像生成、轻量视频模型实验 |
| 48GB+ | 更适合视频 LoRA、较大分辨率和更长帧数 |
| 多卡 A100/H100/H200 | 适合大规模视频模型训练与复现 |

### 8.4 现实建议

个人学习阶段不要尝试从零训练大型视频生成模型。更合理的路线是：

1. 先跑推理。
2. 再调工作流。
3. 再做图像 LoRA。
4. 再做 VLM LoRA。
5. 再做视频 LoRA。
6. 最后研究大规模训练框架。

---

## 9. 训练与调优路线

### 9.1 多模态模型训练

推荐顺序：

1. CLIP 图文对比学习小实验。
2. LLaVA/Qwen-VL 图片问答 LoRA。
3. OCR/文档理解小数据微调。
4. 视频抽帧后的多帧问答微调。
5. 多模态 Agent 应用微调。

常用数据：

- COCO Captions：https://cocodataset.org/
- Visual Genome：https://visualgenome.org/
- VQAv2：https://visualqa.org/
- TextVQA：https://textvqa.org/
- ChartQA：https://github.com/vis-nlp/ChartQA
- LLaVA-Instruct-150K：https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K

### 9.2 视频生成训练

推荐顺序：

1. 图像 LoRA。
2. DreamBooth。
3. AnimateDiff Motion LoRA。
4. CogVideoX / Wan LoRA。
5. Open-Sora 小规模训练。

视频数据处理要点：

- 统一 fps。
- 统一分辨率。
- 控制 clip 时长。
- 去除水印、字幕、低清、强压缩视频。
- 标注主体、动作、场景、镜头、风格、光照。
- 做训练集、验证集、测试集拆分。
- 保存数据版本和处理脚本。

### 9.3 必须掌握的训练工程能力

- LoRA / QLoRA。
- Gradient Checkpointing。
- Mixed Precision。
- FlashAttention / xFormers。
- Accelerate。
- DeepSpeed ZeRO。
- FSDP。
- 多 GPU 启动。
- Checkpoint resume。
- TensorBoard / Weights & Biases。
- 实验配置管理。

---

## 10. 论文阅读方法

读论文不要从公式硬啃，建议按以下顺序：

1. 先读 Abstract，明确论文解决什么问题。
2. 看 Introduction，理解动机和贡献。
3. 看模型结构图，画出自己的流程图。
4. 看 Method，理解输入、输出、模块、训练目标。
5. 看 Experiments，关注数据集、指标、baseline。
6. 看 Ablation，判断哪些设计真正有效。
7. 看 Limitations，理解模型边界。
8. 最后再看公式和实现细节。

每篇论文建议输出一页笔记：

```text
论文标题：
核心问题：
关键贡献：
模型结构：
训练数据：
损失函数：
实验指标：
优点：
局限：
可复现项目：
我能借鉴什么：
```

---

## 11. 前沿追踪清单

### 11.1 常看平台

- arXiv：https://arxiv.org/
- Papers with Code：https://paperswithcode.com/
- Hugging Face Papers：https://huggingface.co/papers
- Hugging Face Models：https://huggingface.co/models
- GitHub Trending：https://github.com/trending
- OpenReview：https://openreview.net/

### 11.2 关键词

多模态方向：

- multimodal large language model
- vision-language model
- visual instruction tuning
- video understanding
- document understanding
- OCR VLM
- embodied multimodal agent

视频生成方向：

- text-to-video generation
- image-to-video generation
- video diffusion model
- diffusion transformer video
- flow matching video generation
- world model video generation
- controllable video generation
- video editing diffusion

### 11.3 重要 Benchmark

- MMBench。
- MMMU。
- MathVista。
- SEED-Bench。
- Video-MME。
- MVBench。
- VBench。
- T2V-CompBench。
- GenEval。

---

## 12. 专业素养清单

### 12.1 工程素养

- 所有实验都保存配置文件。
- 保存随机种子。
- 记录模型版本、数据版本、代码 commit。
- 记录显存、耗时、batch size、学习率。
- 区分训练集、验证集、测试集。
- 不只看 demo，要做系统评估。

### 12.2 研究素养

- 不迷信 benchmark，要看数据污染和评估方式。
- 不只看生成效果，要看失败案例。
- 关注 ablation，而不是只看主表格。
- 关注模型规模、数据规模、训练成本。
- 关注许可证和商用限制。
- 关注安全、隐私、版权、深伪风险。

### 12.3 论文判断标准

判断一个项目是否值得深入，可以问：

1. 是否开源权重？
2. 是否开源训练代码？
3. 是否开源数据处理代码？
4. 是否有清晰 model card？
5. 是否有可复现实验？
6. 是否有明确 license？
7. 是否被主流社区广泛使用？
8. 是否有足够活跃的 issue 和维护？

---

## 13. 3-6 个月长期路线

### 第 1 个月：基础能力建立

- PyTorch 基础。
- Transformer 基础。
- CLIP 入门。
- Stable Diffusion 推理。
- 完成图文检索 demo。

产出：

- 一个 PyTorch 分类训练脚本。
- 一个 CLIP 图文检索项目。
- 一篇 Transformer 学习笔记。

### 第 2 个月：多模态理解入门

- 阅读 LLaVA、BLIP-2、Qwen-VL。
- 跑通 Qwen2.5-VL 或 LLaVA。
- 做图片问答、OCR、图表理解。
- 尝试一个 VLM LoRA 微调。

产出：

- 一个图片问答 Gradio demo。
- 一份多模态模型结构图。
- 一个小型 VQA 微调实验。

### 第 3 个月：图像生成与调优

- 学 DDPM、LDM、DiT、ControlNet。
- 熟悉 Diffusers 和 ComfyUI。
- 完成图像 LoRA 训练。
- 比较不同 scheduler、CFG、steps。

产出：

- 一个 ComfyUI 图像生成工作流。
- 一个自训练 LoRA。
- 一篇 Diffusion 原理笔记。

### 第 4 个月：视频生成入门

- 学 Stable Video Diffusion、AnimateDiff、CogVideoX。
- 跑通本地视频生成。
- 做 prompt 对比和模型对比。
- 学视频数据预处理。

产出：

- 一个文生视频或图生视频 demo。
- 一个视频数据处理脚本。
- 一份视频生成模型对比报告。

### 第 5 个月：视频模型微调与复现

- 学 Wan、HunyuanVideo、Open-Sora。
- 做视频 LoRA 或 Motion LoRA。
- 学多卡训练与显存优化。
- 研究视频生成评估指标。

产出：

- 一个小型视频 LoRA 实验。
- 一份训练成本估算表。
- 一篇视频生成评估笔记。

### 第 6 个月：综合项目与前沿跟踪

- 做完整应用：上传图片/视频，理解内容，再生成视频或摘要。
- 系统阅读 10-20 篇前沿论文。
- 写一份技术综述。
- 建立自己的论文和项目追踪表。

产出：

- 一个端到端多模态 + 生成式应用。
- 一份 20 篇论文阅读笔记。
- 一份个人技术路线图。

---

## 14. 30 天入门计划

### 第 1 周：打基础，能跑模型

#### Day 1：建立环境与学习地图

- 安装 Python、CUDA、PyTorch、Git、VS Code。
- 注册 Hugging Face、GitHub、arXiv 账号。
- 阅读本计划第 1-2 节。
- 建一个学习仓库，包含 `notes/`、`papers/`、`experiments/`、`datasets/`。

产出：

- 环境检查截图或日志。
- 一份个人学习目标清单。

#### Day 2：PyTorch 张量与训练循环

- 学 Tensor、Autograd、Module、Optimizer。
- 跑通 MNIST 或 Fashion-MNIST。
- 保存 checkpoint 并加载推理。

产出：

- `mnist_train.py`。
- 训练 loss 曲线。

#### Day 3：Transformer 基础

- 阅读 Attention Is All You Need 的 Abstract、Introduction、模型图。
- 学 Self-Attention 和 Multi-Head Attention。
- 手写一个简化 attention 函数。

产出：

- 一页 Transformer 笔记。
- 一个 attention 最小实现。

#### Day 4：Hugging Face Transformers 入门

- 安装 `transformers`、`accelerate`。
- 加载一个小型语言模型。
- 完成文本生成、流式输出、参数调整。

产出：

- 一个本地文本生成脚本。

#### Day 5：CLIP 与图文检索

- 阅读 CLIP 论文摘要和方法部分。
- 用 CLIP 编码图片和文本。
- 做一个本地图文检索 demo。

产出：

- 一个输入文字检索图片的小脚本。

#### Day 6：ViT 与视觉编码器

- 阅读 ViT 论文模型结构。
- 理解 patch embedding。
- 用预训练 ViT 做图片分类或特征提取。

产出：

- 一份 ViT 结构图。
- 一个图片特征提取脚本。

#### Day 7：第一周复盘

- 整理 PyTorch、Transformer、CLIP、ViT 笔记。
- 把代码统一放进一个 repo。
- 写一份问题清单：哪些概念还不懂。

产出：

- 第一周学习总结。

---

### 第 2 周：进入多模态理解

#### Day 8：理解 VLM 架构

- 学 Vision Encoder + Projector + LLM 架构。
- 画出 LLaVA / BLIP-2 / Qwen-VL 的结构对比。

产出：

- 一张多模态模型结构图。

#### Day 9：阅读 LLaVA

- 阅读 LLaVA 论文。
- 重点理解 Visual Instruction Tuning。
- 浏览 LLaVA GitHub 项目结构。

产出：

- 一页 LLaVA 论文笔记。

#### Day 10：运行一个 VLM

- 选择 Qwen2.5-VL、MiniCPM-V 或 LLaVA。
- 本地或云端跑通图片问答。
- 测试普通图片、截图、图表、带文字图片。

产出：

- 一个图片问答推理脚本。
- 5 个测试样例和结果。

#### Day 11：OCR 与文档理解

- 用 VLM 分析截图、发票、表格、网页。
- 对比 VLM 和传统 OCR 的差异。
- 记录失败案例。

产出：

- 一份 OCR/VLM 对比笔记。

#### Day 12：视频理解入门

- 用 OpenCV 或 moviepy 抽帧。
- 每隔 N 秒抽一帧，送入 VLM。
- 让模型总结视频内容。

产出：

- 一个视频抽帧 + 总结脚本。

#### Day 13：VLM 小型数据集

- 构造 20-50 条图片问答数据。
- 格式包括图片路径、问题、答案。
- 学习 LLaVA/Qwen-VL 的微调数据格式。

产出：

- 一个小型 VQA JSON 数据集。

#### Day 14：第二周复盘

- 总结 VLM 架构。
- 总结不同图片任务的效果。
- 明确后续是否要做 VLM LoRA。

产出：

- 第二周学习总结。

---

### 第 3 周：Diffusion 与图像生成

#### Day 15：扩散模型直觉

- 阅读 DDPM 论文摘要和核心图。
- 理解加噪和去噪。
- 看一个 DDPM 可视化教程。

产出：

- 一页 DDPM 直觉笔记。

#### Day 16：Latent Diffusion 与 Stable Diffusion

- 阅读 Latent Diffusion 论文结构。
- 理解 VAE、U-Net、Text Encoder。
- 用 Diffusers 跑一张图。

产出：

- 一个 Stable Diffusion 推理脚本。

#### Day 17：Diffusers 深入

- 学 pipeline、scheduler、VAE、UNet、text encoder。
- 调整 steps、CFG、seed、scheduler。
- 记录结果差异。

产出：

- 一份参数对生成效果影响的表格。

#### Day 18：ComfyUI 入门

- 安装并运行 ComfyUI。
- 搭建基础文生图工作流。
- 理解 checkpoint、sampler、VAE、latent、decode 节点。

产出：

- 一个 ComfyUI workflow 文件。

#### Day 19：ControlNet / IP-Adapter

- 了解 ControlNet 原理。
- 尝试姿态、深度或边缘控制。
- 对比普通文生图和条件控制效果。

产出：

- 一组 ControlNet 对比样例。

#### Day 20：LoRA 入门

- 阅读 LoRA 论文核心思想。
- 理解 rank、alpha、target modules。
- 尝试图像 LoRA 训练工具或看完整教程。

产出：

- 一份 LoRA 参数说明笔记。

#### Day 21：第三周复盘

- 总结 Diffusion、LDM、ControlNet、LoRA。
- 整理生成样例。
- 明确下一周视频生成实验目标。

产出：

- 第三周学习总结。

---

### 第 4 周：视频生成与综合项目

#### Day 22：视频生成模型概览

- 阅读 Stable Video Diffusion 或 AnimateDiff 论文。
- 理解视频生成为什么比图像生成难。
- 梳理 T2V、I2V、V2V 区别。

产出：

- 一页视频生成概览笔记。

#### Day 23：运行 AnimateDiff 或 SVD

- 用 ComfyUI 跑 AnimateDiff 或 Stable Video Diffusion。
- 尝试生成 2-4 秒短视频。
- 记录显存占用和耗时。

产出：

- 第一个本地生成视频。

#### Day 24：运行 CogVideoX 或 Wan 轻量模型

- 阅读项目 README。
- 跑通官方推理样例。
- 比较 prompt、分辨率、帧数对显存的影响。

产出：

- 一个视频生成推理脚本或 workflow。

#### Day 25：视频数据处理

- 收集 10-20 个短视频。
- 用脚本切片、抽帧、统一 fps。
- 为每个 clip 写 caption。

产出：

- 一个小型视频数据集。
- 一个视频预处理脚本。

#### Day 26：视频生成评估

- 学 VBench、T2V-CompBench 的基本思想。
- 人工评估 10 个生成视频。
- 从文本对齐、运动质量、主体一致性、画面质量打分。

产出：

- 一个简单视频评估表格。

#### Day 27：综合 Demo 设计

- 设计一个小应用：上传图片或视频，先理解内容，再生成描述或视频。
- 选 Gradio 或 FastAPI。
- 画出系统流程图。

产出：

- 综合 demo 设计文档。

#### Day 28：实现综合 Demo

- 接入 VLM 做图片/视频理解。
- 接入视频生成工作流或命令行推理。
- 做一个最小可用界面。

产出：

- 一个可运行 demo。

#### Day 29：整理论文与项目清单

- 整理 10 篇已读或待读论文。
- 整理 10 个开源项目。
- 标注每个项目的用途、难度、硬件需求、license。

产出：

- 一份论文追踪表。
- 一份开源项目追踪表。

#### Day 30：最终复盘与下一阶段计划

- 总结 30 天学习成果。
- 明确 3 个薄弱点。
- 选择下一个月重点：VLM 微调、图像 LoRA、视频生成部署或视频 LoRA。

产出：

- 30 天学习报告。
- 下一阶段 4 周计划。

---

## 15. 30 天计划的最低完成标准

如果时间有限，至少完成以下内容：

1. 一个 PyTorch 训练脚本。
2. 一个 CLIP 图文检索 demo。
3. 一个 VLM 图片问答 demo。
4. 一个视频抽帧 + VLM 总结 demo。
5. 一个 Stable Diffusion 或 SDXL 推理脚本。
6. 一个 ComfyUI 图像生成 workflow。
7. 一个本地视频生成样例。
8. 三篇论文笔记：CLIP、LLaVA、Latent Diffusion。
9. 一个开源项目追踪表。
10. 一份 30 天复盘报告。

---

## 16. 推荐最终作品集

完成入门后，建议沉淀以下作品：

1. `clip-image-search`：图文检索项目。
2. `vlm-visual-assistant`：图片/截图/图表问答助手。
3. `video-understanding-assistant`：视频抽帧总结助手。
4. `diffusion-lora-lab`：图像 LoRA 训练实验。
5. `local-video-generation-lab`：ComfyUI / CogVideoX / Wan 视频生成实验。
6. `multimodal-video-agent-demo`：多模态理解 + 视频生成的端到端 demo。

这些项目可以组成一个非常有说服力的学习作品集。

---

## 17. 建议优先级

如果只能选一条主线：

1. 先学 VLM，因为部署成本低、反馈快、应用广。
2. 再学 Diffusion，因为是视频生成的基础。
3. 最后学视频生成，因为硬件和数据成本更高。

最推荐的入门组合：

- 多模态理解：Qwen2.5-VL 或 MiniCPM-V。
- 图像生成：Diffusers + SDXL。
- 视频生成：ComfyUI + AnimateDiff / CogVideoX / Wan 轻量版本。
- 微调：PEFT + LoRA。
- 部署：Gradio + FastAPI。
- 论文起点：CLIP -> LLaVA -> Latent Diffusion -> DiT -> Stable Video Diffusion -> CogVideoX/Wan/HunyuanVideo。

---

## 18. 下一步建议

完成本文档后，建议马上做三件事：

1. 建一个独立学习仓库，按 `notes/`、`papers/`、`experiments/`、`demos/` 管理。
2. 第一天不要急着训练，先跑通 PyTorch、Transformers、Diffusers、ComfyUI。
3. 每周至少输出一篇笔记或一个 demo，避免只收藏资料不实践。

