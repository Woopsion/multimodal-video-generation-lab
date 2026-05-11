# Day 2：PyTorch 张量、自动微分与线性回归

运行：

```bash
uv run python experiments/day-02-pytorch-autograd-linear-regression/linear_regression.py
```

观察梯度：

```bash
uv run python experiments/day-02-pytorch-autograd-linear-regression/gradient_debug.py
```

脚本内容：

1. 自动选择计算设备：`cuda -> mps -> cpu`。
2. 用 `requires_grad=True` 和 `backward()` 演示自动微分。
3. 生成 `y = 2x + 1 + noise` 的模拟数据。
4. 用 `torch.nn.Linear(1, 1)` 训练线性回归模型。
5. 保存拟合结果和 loss 曲线到 `outputs/linear_regression_result.png`。

关键训练循环：

```python
pred = model(x)
loss = loss_fn(pred, y)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

需要理解：

- `requires_grad=True`：要求 PyTorch 追踪这个张量上的计算，用于后续求梯度。
- `loss.backward()`：从 loss 反向传播，计算参数梯度。
- `optimizer.step()`：根据梯度更新模型参数。

## 复盘重点

### `unsqueeze(1)`

`torch.linspace(-3.0, 3.0, steps=128)` 的形状是 `[128]`，表示 128 个数字。

`.unsqueeze(1)` 会把它变成 `[128, 1]`：

```text
128 条样本，每条样本 1 个特征
```

`torch.nn.Linear(1, 1)` 需要这种 `[样本数量, 特征数量]` 的输入。

### `torch.nn.Linear(1, 1)`

`torch.nn` 是 PyTorch 的神经网络模块。`Linear(1, 1)` 表示：

```text
每条样本输入 1 个数字
每条样本输出 1 个数字
```

它内部学习两个参数：

```text
pred = weight * x + bias
```

### MSE loss

```python
loss = loss_fn(pred, y)
```

这里的 `loss_fn` 是 `torch.nn.MSELoss()`：

```text
loss = 平均((pred - y)^2)
```

它衡量当前预测值和真实答案之间的平均差距。

本实验有 128 条样本，因此当前的 `loss` 是：

```text
128 条样本的平方误差取平均
```

也就是：

```text
loss = (error_1^2 + error_2^2 + ... + error_128^2) / 128
```

### epoch 和 full-batch

当前实验没有把数据拆成多个 batch。

每一轮训练都会：

```text
使用全部 128 条样本 -> 算一个平均 MSE loss -> 更新一次 weight 和 bias
```

所以在本实验里：

```text
1 个 epoch = 全部 128 条数据被看过一遍 = 参数更新 1 次
```

这叫 `full-batch gradient descent`。

后续 MNIST、CIFAR-10 会使用 mini-batch，例如每次只取 64 张图片训练；那时 1 个 epoch 里面会发生很多次参数更新。

### 梯度和参数更新

单条样本时：

```text
pred = w * x + b
error = pred - y
loss = error^2
```

链式法则给出：

```text
d loss / d w = 2 * error * x
d loss / d b = 2 * error
```

SGD 更新公式：

```text
新参数 = 旧参数 - 学习率 * 梯度
```

梯度告诉参数往哪个方向改，学习率决定每次改多大一步。

多条样本时，梯度也来自平均 MSE：

```text
d loss / d weight = 平均(2 * error_i * x_i)
d loss / d bias   = 平均(2 * error_i)
```

也就是 128 条样本一起决定这一次 `weight` 和 `bias` 应该怎么调。

### `gradient_debug.py` 是干嘛的？

`linear_regression.py` 是正式实验脚本，目标是训练出接近 `y = 2x + 1` 的模型。

`gradient_debug.py` 是调试脚本，目标不是训练效果，而是把训练过程里看不见的东西打印出来：

```text
loss
weight
bias
weight.grad
bias.grad
```

它包含三个观察点：

1. 单条样本时，验证 `d loss / d weight = 2 * error * x` 和 `d loss / d bias = 2 * error`。
2. 连续调用两次 `backward()` 但不调用 `zero_grad()`，观察梯度会累加。
3. 用全部 128 条样本训练几轮，观察每个 epoch 的 loss、gradient 和参数变化。

### `optimizer.zero_grad()` 为什么需要？

PyTorch 默认会累加梯度。

如果第一轮算出：

```text
weight.grad = -12
```

第二轮不清空梯度，又算出：

```text
本轮新梯度 = -12
```

那么参数里保存的会变成：

```text
weight.grad = -24
```

这通常不是我们想要的。每一轮训练应该只根据“当前这一轮”的 loss 计算梯度，所以训练循环里要先执行：

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

含义是：

```text
清空旧梯度 -> 计算当前梯度 -> 用当前梯度更新参数
```
