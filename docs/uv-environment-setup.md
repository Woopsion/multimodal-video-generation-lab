# uv 环境初始化与 PyTorch 安装记录

> 日期：2026-05-11  
> 目的：把本仓库从普通学习资料仓库，补成可复现实验环境。

## 1. 推荐初始化命令

在仓库根目录执行：

```bash
uv init --bare --python 3.12 --name multimodal-video-generation-lab
uv python pin 3.12
```

含义：

- `uv init --bare`：只创建 `pyproject.toml`，不生成默认 `main.py`，适合已有仓库。
- `--python 3.12`：把项目 Python 要求写成 `>=3.12`。
- `uv python pin 3.12`：生成 `.python-version`，让本项目默认使用 Python 3.12。

本仓库当前结果：

```text
.python-version -> 3.12
pyproject.toml  -> requires-python = ">=3.12"
```

## 2. 基础依赖安装命令

Day 2-7 的 PyTorch 基础阶段先安装：

```bash
uv add torch torchvision torchaudio numpy matplotlib pillow tqdm scikit-learn jupyter ipykernel
```

这些依赖的用途：

- `torch`：PyTorch 主库，负责张量、自动微分、神经网络训练。
- `torchvision`：图像数据集、图像 transform、MNIST/CIFAR-10 等。
- `torchaudio`：与 PyTorch 版本对齐的音频扩展，先保留。
- `numpy`：基础数值计算。
- `matplotlib`：绘制训练曲线和结果图。
- `pillow`：图片读取与处理。
- `tqdm`：训练进度条。
- `scikit-learn`：基础指标和传统机器学习工具。
- `jupyter`、`ipykernel`：后续 notebook 实验环境。

## 3. 环境验证命令

确认 Python 版本：

```bash
uv run python -c "import sys; print(sys.executable); print(sys.version)"
```

确认 PyTorch 与设备：

```bash
uv run python -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available()); print('mps:', torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False)"
```

当前 Mac 环境验证结果：

```text
Python: 3.12.13
PyTorch: 2.11.0
CUDA: False
MPS: True
MPS built: True
```

结论：

- 当前 Mac 没有 NVIDIA CUDA，`cuda: False` 正常。
- 当前 PyTorch 支持 Apple GPU 后端，`mps: True` 表示可以用 Apple GPU 运行部分 PyTorch 计算。

## 4. 关于 uv 缓存目录

正常终端里直接使用 `uv` 即可，不需要指定缓存目录。

如果遇到默认缓存目录权限问题，例如无法写入 `~/.cache/uv`，可以临时使用项目内缓存：

```bash
export UV_CACHE_DIR=.uv-cache
```

然后继续执行：

```bash
uv add torch
uv run python your_script.py
```

`.uv-cache/` 不应提交到 Git，已加入 `.gitignore`。

## 5. 后续多模态依赖

Week 2/3 再安装多模态和生成相关依赖：

```bash
uv add transformers datasets accelerate peft diffusers safetensors sentencepiece protobuf opencv-python
```

不建议第一天全部安装，先保证 PyTorch 基础训练闭环可运行。

