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
  - `experiments/pytorch-basics/linear_regression.py`
  - `experiments/pytorch-basics/README.md`
  - `experiments/pytorch-basics/outputs/linear_regression_result.png`
- 已新增环境说明：
  - `docs/uv-environment-setup.md`
- 关于 uv 缓存的结论：
  - 正常终端中直接使用默认 uv 缓存即可。
  - 只有遇到权限问题时，才临时使用 `UV_CACHE_DIR=.uv-cache`。
