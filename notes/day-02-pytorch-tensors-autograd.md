# Day 2：PyTorch 张量、自动微分与线性回归

> 日期：2026-05-11  
> 对应实验：`experiments/day-02-pytorch-autograd-linear-regression/linear_regression.py`

## 1. 今天真正要理解什么？

Day 2 的目标不是记住很多 PyTorch API，而是理解训练模型的最小闭环：

```text
数据 -> 模型预测 -> 计算误差 -> 反向传播 -> 更新参数
```

这个闭环以后会反复出现：

```text
线性回归
MNIST 分类
CNN
Transformer
VLM
Diffusion
LoRA 微调
```

## 2. 张量 Tensor 是什么？

张量可以先理解成“多维数组”。

```text
0 维张量：一个数字，例如 3.14
1 维张量：一列数字，例如 [1, 2, 3]
2 维张量：表格 / 矩阵，例如 3 行 4 列
3 维张量：一张图片，例如 高 x 宽 x 通道
4 维张量：一批图片，例如 batch x 通道 x 高 x 宽
```

PyTorch 用 `torch.Tensor` 表示这些数据。

## 3. shape 是什么？

`shape` 描述张量每个维度的大小。

例如：

```python
x = torch.linspace(-3.0, 3.0, steps=128).unsqueeze(1)
```

这里的 `x.shape` 是：

```text
torch.Size([128, 1])
```

含义是：

```text
128 条样本，每条样本有 1 个特征
```

### 3.1 `unsqueeze(1)` 到底是什么意思？

`torch.linspace(-3.0, 3.0, steps=128)` 会生成 128 个数字。

此时形状是：

```text
[128]
```

可以想象成：

```text
[-3.0, -2.95, -2.90, ..., 3.0]
```

这是“一维的一串数字”。

调用：

```python
.unsqueeze(1)
```

意思是：

```text
在第 1 个维度的位置插入一个新维度。
```

形状从：

```text
[128]
```

变成：

```text
[128, 1]
```

直观上从：

```text
[-3.0, -2.95, -2.90]
```

变成：

```text
[
  [-3.0],
  [-2.95],
  [-2.90]
]
```

也就是：

```text
128 行，每行 1 个数字。
```

### 3.2 样本数量和特征数量是什么？

一条数据叫一个样本。

例如房价预测：

```text
样本 1：面积 80，房间数 2，楼龄 10 -> 房价 300 万
样本 2：面积 120，房间数 3，楼龄 5 -> 房价 500 万
```

这里：

```text
样本数量 = 有几套房子
特征数量 = 每套房子有几个输入信息
```

如果输入信息是：

```text
面积、房间数、楼龄
```

那么特征数量就是 `3`。

在本实验中，每条样本只有一个输入 `x`，所以：

```text
样本数量 = 128
特征数量 = 1
x.shape = [128, 1]
```

## 4. 线性回归在学什么？

实验里我们先制造一批假数据：

```text
y = 2x + 1 + noise
```

其中：

- `x` 是输入。
- `y` 是真实答案。
- `noise` 是随机噪声，让数据更像真实世界。

模型不知道真实公式。模型只能学习：

```text
pred = weight * x + bias
```

训练目标是让：

```text
weight -> 接近 2
bias   -> 接近 1
```

### 4.1 `torch.nn.Linear(1, 1)` 是什么？

`torch.nn` 是 PyTorch 里专门放神经网络组件的模块。`nn` 可以理解为 `neural network` 的缩写。

常见组件包括：

```python
torch.nn.Linear
torch.nn.Conv2d
torch.nn.ReLU
torch.nn.MSELoss
torch.nn.CrossEntropyLoss
```

`torch.nn.Linear(1, 1)` 表示一个线性层：

```text
输出 = 输入 * weight + bias
```

也就是：

```text
pred = x * w + b
```

两个 `1` 的含义是：

```python
torch.nn.Linear(输入特征数量, 输出特征数量)
```

所以：

```python
torch.nn.Linear(1, 1)
```

表示：

```text
每条样本输入 1 个数字
每条样本输出 1 个数字
```

如果是房价预测，输入是面积、房间数、楼龄 3 个特征，输出是 1 个房价，可以写成：

```python
torch.nn.Linear(3, 1)
```

## 5. loss 是什么？

`loss` 是模型“错得有多离谱”的数字。

本实验使用 MSE：

```text
loss = 平均((预测值 - 真实值)^2)
```

loss 越小，说明预测越接近真实答案。

### 5.1 `loss = loss_fn(pred, y)` 到底算什么？

本实验中：

```python
loss_fn = torch.nn.MSELoss()
```

MSE 的含义是均方误差：

```text
loss = 平均((pred - y)^2)
```

其中：

- `pred` 是模型当前用 `weight` 和 `bias` 算出来的预测值。
- `y` 是提前制造好的真实答案。

真实答案来自：

```python
y = 2.0 * x + 1.0 + noise
```

模型预测来自：

```python
pred = model(x)
```

所以 `loss = loss_fn(pred, y)` 的意思就是：

```text
当前模型预测值和真实答案之间平均差多少。
```

例如只有 3 条数据：

```text
真实值 y：     [3, 5, 7]
模型预测 pred：[2, 6, 10]
```

差值：

```text
pred - y = [-1, 1, 3]
```

平方：

```text
[1, 1, 9]
```

平均：

```text
(1 + 1 + 9) / 3 = 3.67
```

此时 loss 就是 `3.67`。

### 5.2 当前实验里的 loss 是 128 条样本的平均 MSE

当前实验里：

```text
x.shape = [128, 1]
y.shape = [128, 1]
```

也就是说，一共有 128 条数据。

模型会一次性得到 128 个预测值：

```text
pred.shape = [128, 1]
```

每条样本都有自己的误差：

```text
error_i = pred_i - y_i
```

每条样本也都有自己的平方误差：

```text
loss_i = error_i^2
```

`torch.nn.MSELoss()` 默认会把 128 个平方误差取平均：

```text
loss = (loss_1 + loss_2 + ... + loss_128) / 128
```

所以当前代码里的 `loss` 是：

```text
128 条样本共同产生的一个平均错误值。
```

## 6. requires_grad=True 是什么？

`requires_grad=True` 的意思是：

```text
请 PyTorch 记录这个张量参与过哪些计算，因为后面我要对它求梯度。
```

例如：

```python
x = torch.tensor([2.0], requires_grad=True)
y = x * x + 3 * x + 1
y.backward()
```

数学上：

```text
y = x^2 + 3x + 1
dy/dx = 2x + 3
x = 2 时，dy/dx = 7
```

所以 `x.grad` 会得到 `7`。

## 7. backward() 在做什么？

`loss.backward()` 会从 loss 开始反向计算梯度。

可以理解成它在问：

```text
如果想让 loss 变小，weight 应该怎么动？
如果想让 loss 变小，bias 应该怎么动？
```

梯度就是这个“应该怎么动”的方向和强度。

## 8. optimizer.step() 在做什么？

`optimizer.step()` 根据梯度更新参数。

例如：

```text
weight 当前太小 -> 梯度提示它应该变大 -> optimizer.step() 把它调大一点
bias 当前太大   -> 梯度提示它应该变小 -> optimizer.step() 把它调小一点
```

训练不是一步到位，而是每轮挪一点。

### 8.0 epoch、batch 和本实验的训练轮次

在当前代码里：

```python
for epoch in range(1, epochs + 1):
    pred = model(x)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

循环执行一次，就是一个 `epoch`。

因为当前实验没有把 128 条数据拆成多个小 batch，所以每一轮都是：

```text
拿全部 128 条样本 -> 算一个平均 MSE loss -> 更新一次 weight 和 bias
```

因此在这个实验里：

```text
1 个 epoch = 看完整个 128 条数据 1 次 = 更新参数 1 次
```

这种训练方式叫：

```text
full-batch gradient descent
```

后面 MNIST、CIFAR-10 这类数据集通常会使用 mini-batch。

例如 60,000 张图片，batch size = 64：

```text
1 个 batch = 64 条数据
1 个 epoch = 60,000 条数据都被模型看过一遍
1 个 epoch 里会有大约 60,000 / 64 = 938 次参数更新
```

所以要区分：

```text
epoch：全体训练数据被看过一遍
batch：每次实际拿来计算 loss 和更新参数的一小批数据
```

当前 Day 2 实验比较小，所以直接用全部 128 条数据作为一个 batch。

## 8.1 梯度公式是怎么来的？

先看单条样本。

模型预测：

```text
pred = weight * x + bias
```

写短一点：

```text
pred = w * x + b
```

误差：

```text
error = pred - y
```

loss：

```text
loss = error^2
```

合起来：

```text
loss = (w * x + b - y)^2
```

现在要问：

```text
如果 w 变大一点，loss 会怎么变？
如果 b 变大一点，loss 会怎么变？
```

这就是求：

```text
d loss / d w
d loss / d b
```

令：

```text
error = w * x + b - y
loss = error^2
```

平方函数的导数是：

```text
d loss / d error = 2 * error
```

对 `w` 来说：

```text
d error / d w = x
```

因为 `error = w * x + b - y`，`w` 变化一点，会按 `x` 的倍数影响 error。

用链式法则：

```text
d loss / d w
= d loss / d error * d error / d w
= 2 * error * x
```

所以：

```text
d loss / d weight = 2 * error * x
```

对 `b` 来说：

```text
d error / d b = 1
```

因为 `b` 增加 1，`error` 也增加 1。

所以：

```text
d loss / d b
= d loss / d error * d error / d b
= 2 * error * 1
= 2 * error
```

也就是：

```text
d loss / d bias = 2 * error
```

### 8.2 用数字走一遍梯度计算

假设：

```text
x = 2
y = 5
w = 1
b = 0
```

预测：

```text
pred = 1 * 2 + 0 = 2
```

误差：

```text
error = pred - y = 2 - 5 = -3
```

loss：

```text
loss = (-3)^2 = 9
```

weight 梯度：

```text
d loss / d w = 2 * error * x
             = 2 * (-3) * 2
             = -12
```

bias 梯度：

```text
d loss / d b = 2 * error
             = 2 * (-3)
             = -6
```

梯度是负数，含义是：

```text
w 变大时，loss 会下降
b 变大时，loss 会下降
```

这符合直觉：当前预测是 `2`，真实答案是 `5`，预测太小，所以应该把 `w` 和 `b` 往大调。

### 8.3 SGD 更新公式和学习率

SGD 的核心更新公式是：

```text
新参数 = 旧参数 - 学习率 * 梯度
```

也就是：

```text
w_new = w_old - lr * d loss / d w
b_new = b_old - lr * d loss / d b
```

为什么是减号？

```text
梯度指向 loss 增长最快的方向。
我们想让 loss 变小，所以要往梯度的反方向走。
```

继续上面的例子：

```text
w = 1
d loss / d w = -12
lr = 0.05
```

更新：

```text
w_new = 1 - 0.05 * (-12)
      = 1 + 0.6
      = 1.6
```

因为梯度是负数，减去负数就会让 `w` 变大。

bias 同理：

```text
b = 0
d loss / d b = -6
lr = 0.05
```

更新：

```text
b_new = 0 - 0.05 * (-6)
      = 0.3
```

学习率 `lr` 控制每次更新走多大一步：

```text
梯度：告诉你往哪个方向走
学习率：告诉你每次走多远
```

如果学习率太小，训练会很慢。  
如果学习率太大，可能一步跨过最优点，甚至让 loss 越来越大。

### 8.4 多条样本时梯度怎么算？

真实实验里不是 1 条样本，而是 128 条样本：

```text
x.shape    = [128, 1]
y.shape    = [128, 1]
pred.shape = [128, 1]
```

每条样本都有自己的误差：

```text
error_i = pred_i - y_i
```

MSELoss 默认取平均，也就是当前代码里的 `loss`：

```text
loss = 平均(error_i^2)
```

所以梯度也是所有样本共同决定的：

```text
d loss / d weight = 平均(2 * error_i * x_i)
d loss / d bias   = 平均(2 * error_i)
```

它不是枚举 `weight` 和 `bias`，而是用公式直接算出当前参数位置的下降方向。

PyTorch 的：

```python
loss.backward()
```

就是自动完成这些梯度计算，并把结果放到：

```python
model.weight.grad
model.bias.grad
```

随后：

```python
optimizer.step()
```

使用这些梯度更新参数。

换句话说，当前实验不是每条样本单独更新一次参数，而是：

```text
128 条样本共同算出一个平均 loss
这个平均 loss 反向传播出 weight 和 bias 的梯度
optimizer.step() 根据这两个梯度整体更新 weight 和 bias
```

## 9. 为什么要 zero_grad()？

PyTorch 默认会累加梯度。

所以每一轮训练前都要执行：

```python
optimizer.zero_grad()
```

否则上一轮的梯度会混进这一轮，导致更新方向不干净。

更具体地说，`loss.backward()` 不是“覆盖”旧梯度，而是把新梯度“加到”旧梯度上。

例如第一轮：

```text
weight.grad = -12
```

第二轮如果没有执行 `optimizer.zero_grad()`，又算出一个新梯度：

```text
new weight.grad = -12
```

PyTorch 会累加成：

```text
weight.grad = -24
```

这会让 `optimizer.step()` 使用混合了旧信息的梯度，参数更新就不再只代表当前这一轮的 loss。

所以标准训练顺序是：

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

含义：

```text
清空旧梯度 -> 计算当前梯度 -> 根据当前梯度更新参数
```

## 9.1 gradient 是什么？

`gradient` 就是梯度。

在当前实验中，它具体指：

```text
loss 对 weight 的梯度
loss 对 bias 的梯度
```

也就是：

```text
d loss / d weight
d loss / d bias
```

它回答的问题是：

```text
如果 weight 增加一点，loss 会怎么变？
如果 bias 增加一点，loss 会怎么变？
```

梯度不是参数本身，也不是 loss 本身，而是：

```text
参数应该往哪个方向调整，才能让 loss 更小。
```

PyTorch 会把梯度保存到：

```python
model.weight.grad
model.bias.grad
```

本仓库的观察脚本：

```bash
uv run python experiments/day-02-pytorch-autograd-linear-regression/gradient_debug.py
```

就是专门用来打印 `weight.grad` 和 `bias.grad`，帮助观察梯度如何产生、如何累加、如何被 `zero_grad()` 清空。

## 10. os.environ.setdefault("MPLCONFIGDIR", ...) 是什么？

这行代码不是训练逻辑：

```python
os.environ.setdefault("MPLCONFIGDIR", str((PROJECT_ROOT / ".matplotlib-cache").resolve()))
```

它只是告诉 Matplotlib：

```text
把画图缓存放到当前项目的 .matplotlib-cache 目录。
```

原因是 Matplotlib 默认可能想写入用户目录，例如：

```text
/Users/woopsion/.matplotlib
```

在某些受限环境下会没有权限。设置 `MPLCONFIGDIR` 可以避免这个问题。

## 11. Day 2 的当前验收问题

学完这个实验后，应该能回答：

1. 张量和普通数字、列表有什么关系？
2. `x.shape` 为什么是 `[128, 1]`？
3. `requires_grad=True` 是让 PyTorch 做什么？
4. `loss.backward()` 算出了什么？
5. `optimizer.step()` 为什么能让模型越来越准？
6. 为什么模型最后能学到接近 `y = 2x + 1`？
7. `d loss / d weight = 2 * error * x` 是怎么用链式法则推出来的？
8. 学习率为什么不能太大也不能太小？
9. 当前实验里 1 个 epoch 和 1 次参数更新是什么关系？
10. 当前实验里的 loss 为什么是 128 条样本的平均 MSE？
