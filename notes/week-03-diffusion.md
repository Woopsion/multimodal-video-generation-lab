# Week 3 - Diffusion、Stable Diffusion 与图像生成

> 当前阶段：Day15 Step 03。Step 01-02 已完成，现在学习 DDPM 的反向概率分布、scheduler 与逐步采样。

## Week3 主线

```text
clean image x_0
-> known forward noising process
-> noisy sample x_t
-> neural network predicts noise
-> scheduler converts prediction into reverse update
-> iterative sampling
-> generated image
```

## Day15 学习步骤

1. Step 01：前向加噪、正态分布与高斯噪声。
2. Step 02：训练时为什么让模型预测噪声。
3. Step 03：反向去噪如何计算 `x_{t-1}`。
4. Step 04：采样步数、随机性与 DDPM。
5. Step 05：DDIM 与更少采样步数。
6. Step 06：完整复述训练流和推理流。

## Step 01：前向过程

前向加噪不是为了增强图片，而是把困难的一步生成拆成许多小的反向去噪任务，并自动构造不同噪声难度的训练样本。

现代扩散概率模型的重要来源：

- 2015：Sohl-Dickstein et al., *Deep Unsupervised Learning using Nonequilibrium Thermodynamics*。
- 2020：Ho et al., *Denoising Diffusion Probabilistic Models* / DDPM。

### 正态分布

```text
f(x) = 1 / (sigma * sqrt(2*pi))
     * exp(-(x-mu)^2 / (2*sigma^2))
```

- `mu`：均值，决定分布中心。
- `sigma^2`：方差，表示离散程度。
- `sigma`：标准差。
- `N(0,1)`：均值 0、方差 1 的标准正态分布。
- 约 68%、95%、99.7% 的值分别位于均值左右 1、2、3 个标准差范围内。

经典连续图像扩散选择高斯噪声，是因为高斯变量的线性组合仍是高斯分布、任意时间步可以直接采样、最终先验 `N(0,I)` 容易生成，并且连续扰动适合图像张量。

### epsilon

`epsilon` 是与图片形状相同的随机噪声张量，不是一个标量，也不是 loss。

```text
x_0.shape     = [C, H, W]
epsilon.shape = [C, H, W]
x_t.shape     = [C, H, W]
```

前向公式：

```text
x_t = sqrt(alpha_bar_t) * x_0
    + sqrt(1 - alpha_bar_t) * epsilon
```

## Step 02：噪声预测训练

一次训练迭代：

```text
sample x_0
-> sample random t
-> sample epsilon ~ N(0,I)
-> construct x_t
-> epsilon_pred = model(x_t, t)
-> loss = MSE(epsilon_pred, epsilon)
-> backward + optimizer.step
```

为什么预测噪声：

- `epsilon` 是程序自己采样的，监督答案天然已知。
- 所有噪声目标来自统一的标准正态分布，训练尺度稳定。
- 预测出噪声意味着获得了当前状态的去噪方向。
- 预测噪声不是唯一参数化，其他模型也可能预测 `x_0`、score 或 `v`。

## 当前必须掌握

- 前向过程没有可训练参数，神经网络训练发生在噪声预测阶段。
- 模型输入是 `x_t` 和 `t`，输出是与图片同形状的预测噪声。
- 时间步必须输入模型，因为不同噪声程度需要不同去噪策略。
- 反向一步不是简单执行 `x_t - epsilon_pred`，还需要 scheduler 系数。

## 主学习入口

- `docs/day-15-diffusion-basics/index.html`
## Step 03：DDPM 与反向去噪

DDPM 全称是 Denoising Diffusion Probabilistic Models，即去噪扩散概率模型。它不是某个具体神经网络名称，而是前向扩散、噪声预测训练和反向概率采样的整体框架。

### 前向一步

```text
q(x_t | x_{t-1}) = N(sqrt(alpha_t) * x_{t-1}, beta_t * I)
```

- `beta_t`：第 t 步加入的噪声强度。
- `alpha_t = 1 - beta_t`：该步保留比例。
- `alpha_bar_t = product(alpha_1 ... alpha_t)`：累计信号保留量。
- `I`：单位矩阵，表示各维度独立、同尺度的高斯噪声。

### 反向一步

```text
p_theta(x_{t-1} | x_t)
= N(mu_theta(x_t,t), sigma_t^2 * I)
```

经典 epsilon 参数化：

```text
mu_theta = 1 / sqrt(alpha_t)
         * (x_t - beta_t / sqrt(1-alpha_bar_t) * epsilon_pred)

x_{t-1} = mu_theta + sigma_t * z
z ~ N(0,I)
```

神经网络负责根据 `x_t` 和 `t` 预测 `epsilon_pred`；scheduler 保存时间步系数，并把噪声预测转换为反向均值、方差和 `x_{t-1}`。

### DDPM 推理

```text
x_T ~ N(0,I)
for t = T ... 1:
    epsilon_pred = model(x_t, t)
    x_{t-1} = scheduler.step(epsilon_pred, t, x_t)
return x_0
```

训练时可以随机抽取一个时间步独立构造样本；推理时必须从高噪声状态开始，按照 scheduler 的时间步序列反复调用模型。
