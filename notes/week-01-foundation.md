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

`experiments/day-02-pytorch-autograd-linear-regression/linear_regression.py` 是 Day 2 的第一个 PyTorch 实验。它用于理解：

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

## Week 1 复盘启动：Day 1 到 Day 6

HTML 复盘入口：

```text
docs/week-01-foundation-review/index.html
```

### 1. 复盘目标

本周复盘不是重新抄一遍笔记，而是检查三件事：

1. 能不能把 Day1 到 Day6 的知识串成一条主线。
2. 能不能指出每段代码到底在验证哪个概念。
3. 能不能暴露还不稳定的概念，后面继续追问和补强。

### 2. 当前已经完成的主线

| 天数 | 主题 | 当前产出 | 复盘时要能说清楚 |
| --- | --- | --- | --- |
| Day 1 | 学习目标、环境、VLM / Diffusion / Video Generation 总览 | `docs/30-day-executable-checklist.md`、`logs/daily-checkin.md` | 为什么先学 PyTorch，再学 Transformer / LLM，再进入多模态和生成模型。 |
| Day 2 | Tensor、矩阵乘法、自动微分、线性回归 | `experiments/day-02-pytorch-autograd-linear-regression/` | `loss.backward()` 如何得到梯度，`optimizer.step()` 如何更新参数。 |
| Day 3 | MNIST 训练、验证、checkpoint、推理、训练曲线 | `experiments/day-03-mnist-classifier/` | CrossEntropyLoss 如何从 logits 产生梯度，以及 checkpoint 为什么能恢复模型。 |
| Day 4 | CNN、卷积核、通道、padding、stride、pooling | `notes/day-04-cnn-cifar10.md` | CNN 为什么保留空间结构，卷积输出 shape 如何计算。 |
| Day 5 | Transformer Encoder、Self-Attention、Multi-Head、FFN、Residual、LayerNorm | `docs/day-05-transformer-encoder/index.html` | 一个 token 如何通过 attention 聚合其他 token 的信息。 |
| Day 6 | LLM 推理流程、Causal LM、KV Cache、真实模型生成参数 | `docs/day-06-llm-inference/index.html` | prompt 如何变成 token id，模型如何得到 logits，生成参数如何影响 next token。 |

### 3. Week 1 复盘的第一轮问题

本组问题用于复盘自测：

1. 训练阶段和推理阶段，模型参数分别会不会更新？
2. logits 为什么还不是概率？softmax 在这里解决什么问题？
3. `loss.backward()` 算出来的梯度最终影响的是哪些参数？
4. CNN 的卷积核和 Transformer 的 Q/K/V 参数矩阵都叫“可学习参数”，它们相同点和不同点是什么？
5. Self-Attention 里的 attention weight 和 LLM 生成时的 token probability 是不是同一种东西？
6. KV Cache 缓存的是最终 hidden state，还是每层 attention 里的 K/V？

### 4. 下一步执行方式

复盘时优先按“问题驱动”推进：每次只拿一个问题深挖，必要时回到对应代码和 HTML 页面验证 shape 与数据流。

### 5. 第一关讲评记录：训练 vs 推理

学习者当前理解：

- 训练、验证、推理都会执行 `model(x)`。
- 训练阶段会更新参数；验证和推理阶段不更新参数。
- logits 是 Linear 映射后的原始分数，不是概率；softmax 将其转成概率分布。
- KV Cache 缓存每层 attention 的 K/V。

需要纠正：

- `loss.backward()` 只负责计算梯度，并把梯度写到参数的 `.grad` 中。
- `optimizer.step()` 才真正根据当前 `.grad` 修改参数。
- `optimizer.step()` 不会选择历史最低 loss 对应的权重；保存 best model 是 checkpoint 逻辑。
- attention weight 不是 hidden state，它是 query 对 key 做 softmax 后得到的位置权重；`attention_weight @ V` 得到的 context 才是新的向量表示。

当前结论：

- 第一关可以通过。
- 下一关进入 logits、softmax、CrossEntropyLoss、梯度的统一复盘。

### 6. 第二关讲评记录：logits、softmax、CrossEntropyLoss

学习者当前理解：

- MNIST 的 10 个 logits 和 LLM 的 50257 个 logits，本质上都是最后 Linear 映射出的候选项原始分数。
- 索引含义不是模型天然知道的，而是由数据集 label 或 tokenizer vocab 固定；训练阶段通过 loss 让模型对齐这些索引。
- `p_i` 可以理解为预测值，`y_i` 可以理解为真实值。

需要补强：

- `CrossEntropyLoss` 的作用是惩罚“正确类别概率低”的情况，奖励“正确类别概率高”的情况。
- PyTorch 的 `nn.CrossEntropyLoss` 直接接收 logits，内部等价于稳定版 `log_softmax` + 负对数似然，不需要手动 softmax。
- 更精确地说，`p_i` 是 softmax 后第 `i` 个类别的预测概率；`y_i` 是 one-hot 真实标签第 `i` 位的 0 或 1。
- 如果真实类别是 7，`y_7 = 1`，其他 `y_i = 0`。
- 梯度公式 `dLoss/dz_i = p_i - y_i` 的含义是：正确类别的 logit 通常被推高，错误类别的 logit 通常被压低。

当前结论：

- logits 的大方向已经掌握。
- CrossEntropyLoss 需要通过第二轮追问继续钉牢。

### 7. 第二关追问讲评：one-hot 与错误类别梯度

学习者当前理解：

- 如果正确类别概率低，`Loss = -log(p_7)` 会变大。
- 错误类别应该被压低，权重会朝着让错误类别分数下降的趋势调整。

需要纠正和补强：

- one-hot 是真实标签的编码方式，不是模型输出。真实类别是 7 时，`y = [0,0,0,0,0,0,0,1,0,0]`。
- `y_7 = 1` 表示目标概率全部放在类别 7 上；其他 `y_i = 0` 表示它们不是真实答案。
- PyTorch 的 `nn.CrossEntropyLoss` 通常直接接收类别编号 `label = 7`，不需要手动传 one-hot。
- `z_3` 是错误类别 3 的 logit，是 forward 输出值，不是参数。
- `p_3` 是错误类别 3 的预测概率，也不是参数。
- `dLoss/dz_3 = p_3` 是 loss 对错误类别 logit 的梯度，表示应该把 `z_3` 往下压。

关键链路：

```text
z_3 = w_3 · h + b_3
dLoss/dz_3 = p_3
dLoss/dw_3 = p_3 * h
dLoss/db_3 = p_3
```

所以 optimizer 更新的是 `w_3` 和 `b_3`，不是直接更新 `z_3`。对类似样本来说，更新后的 `w_3 · h + b_3` 会变小，从而压低错误类别 3 的分数。

当前结论：

- 第二关可以通过。
- 之后进入“参数如何被 logits 梯度影响”的链式法则复盘。

### 8. 第三关启动：logits 梯度如何影响参数

核心链路：

```text
h -> Linear -> logits -> CrossEntropyLoss
```

对类别 `i`：

```text
z_i = w_i · h + b_i
g_i = dLoss/dz_i = p_i - y_i
dLoss/dw_i,k = g_i * h_k
dLoss/db_i = g_i
```

符号含义：

- `h` 是进入最后 Linear 层之前的隐藏向量。
- `z_i` 是第 `i` 个类别的 logit，是 forward 输出值，不是参数。
- `w_i` 是最后 Linear 层第 `i` 个类别对应的权重向量。
- `w_i,k` 是 `w_i` 的第 `k` 个参数。
- `b_i` 是第 `i` 个类别对应的 bias。
- `g_i` 是 loss 对第 `i` 个 logit 的梯度。

batch 规则：

- 单个样本会贡献一份梯度。
- batch size 为 256 时，默认 `reduction="mean"` 下可以理解为 256 份梯度先平均。
- 一次 `optimizer.step()` 使用聚合后的 `parameter.grad` 更新一次参数。

重要边界：

- `h` 通常不是长期保存的参数，而是当前 forward 的中间结果。
- 反向传播会通过 `h` 继续找到产生它的前面层参数，例如更早的 Linear、CNN 卷积核、Attention 参数、Embedding 参数。

保留自测题：

1. 为什么 `dLoss/dw_7,k = dLoss/dz_7 * h_k`？
2. batch size 为 256 时，最终是每个样本单独更新一次参数，还是梯度聚合后更新一次？
3. 为什么 `h` 本身通常不是参数，但梯度还能通过 `h` 继续影响更前面的层？

### 9. 第三关讲评：参数梯度链路

学习者当前理解：

- `dLoss/dw = dLoss/dz * dz/dw` 是链式求导。
- batch size 为 256 时，梯度聚合后更新一次参数。
- `h` 可以由上一层计算公式表达，所以梯度能继续往前传。

需要精确化：

- `w_7,k` 是最后 Linear 层中类别 7 对应权重向量的第 `k` 个参数。
- `z_7` 是类别 7 的 logit，是 forward 输出值，不是参数。
- `h_k` 是进入最后 Linear 层的隐藏向量第 `k` 个数。
- 默认 `nn.CrossEntropyLoss(reduction="mean")` 是先对 batch 内样本 loss 求平均，再对平均 loss 做 `backward()`；因此梯度等价于样本梯度的平均。
- 一次 batch 对应一次聚合后的 `parameter.grad`，一次 `optimizer.step()` 更新一次参数。
- `h` 不是上一层的权重矩阵，而是上一层使用参数和输入算出来的中间输出。

关键公式：

```text
h = f(x; theta_previous)
dLoss/dtheta_previous = dLoss/dh * dh/dtheta_previous
```

当前结论：

- 第三关可以通过。
- CNN 卷积核梯度、Q/K/V 参数梯度留作后续观察项。

## Week 1 最终收口

Week1 到此收口，不再继续追加密集关卡式追问。当前基础主线已经建立：

```text
数据 / 文本
-> 张量 / token id
-> 模型 forward
-> logits
-> loss / sampling
-> backward 计算梯度
-> optimizer.step 更新参数
-> checkpoint 保存权重
-> 推理阶段固定参数使用
```

当前已经掌握的核心边界：

- 训练、验证、推理都会 forward；只有训练执行 `backward()` 和 `optimizer.step()` 更新参数。
- `CrossEntropyLoss` 直接接收 logits，内部等价于稳定版 `log_softmax` + 负对数似然。
- `dLoss/dz_i = p_i - y_i` 是 logits 层面的梯度，后续通过链式法则影响产生 logits 的参数。
- `h` 是前面层根据参数计算出来的中间输出，不是参数本身，但梯度可以通过它继续传回前面的参数。
- LLM 推理阶段固定参数，通过 tokenizer、embedding、causal Transformer、LM Head、sampling 逐 token 生成。

后续观察项：

- CNN 卷积核梯度。
- Transformer Q/K/V 参数梯度。
- attention weight 与 token probability 的区别。
- KV Cache 的工程实现细节。
- LoRA / QLoRA 微调。

Week2 入口：

```text
docs/week-02-multimodal-understanding/index.html
```
