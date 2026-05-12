# Day 3：MNIST 完整训练脚本学习记录

> 日期：2026-05-12  
> 对应实验：`experiments/day-03-mnist-classifier/train.py`

## 1. 今天真正要理解什么？

Day 3 的重点不是 MNIST 本身，而是第一次完成一个完整的监督学习训练闭环：

```text
Dataset -> DataLoader -> Model -> Loss -> Backward -> Optimizer -> Validation -> Checkpoint -> Inference
```

Day 2 的线性回归重点是理解自动微分和参数更新；Day 3 则把这个过程扩展成更真实的分类训练脚本。

## 2. Dataset 和 DataLoader

`Dataset` 负责表示数据集，能够返回单条样本：

```text
image, label = dataset[index]
```

在 MNIST 中：

```text
image：一张 28 x 28 的灰度手写数字图片
label：0 到 9 之间的类别标签
```

`DataLoader` 负责把单条样本组织成 batch：

```text
images.shape = [batch_size, 1, 28, 28]
labels.shape = [batch_size]
```

其中：

```text
batch_size：一次送入模型的样本数量
1：灰度图只有 1 个通道
28 x 28：图片高和宽
```

训练集使用 `shuffle=True`，让每轮训练的数据顺序打乱，减少模型记住固定顺序的风险。

验证集使用 `shuffle=False`，因为验证只关心稳定评估结果。

## 3. 模型结构

本实验使用一个最小 MLP：

```text
28 x 28 图片
-> Flatten
-> Linear(784, 128)
-> ReLU
-> Linear(128, 10)
-> logits
```

其中：

```text
Flatten：把 [1, 28, 28] 展平成 784 个数字
Linear(784, 128)：提取隐藏特征
ReLU：加入非线性
Linear(128, 10)：输出 10 个类别分数
```

输出的 10 个数字叫 `logits`，它们还不是概率，而是模型对 10 个类别的原始打分。

## 4. Loss：CrossEntropyLoss

MNIST 是多分类任务，所以使用：

```python
loss_fn = nn.CrossEntropyLoss()
```

它会比较：

```text
logits：模型输出的 10 类分数
labels：真实类别，例如 7
```

目标是让真实类别对应的分数越来越高，错误类别对应的分数越来越低。

## 5. 训练循环

核心训练循环依然是 Day 2 学过的闭环：

```python
logits = model(images)
loss = loss_fn(logits, labels)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

含义：

```text
model(images)：前向传播，得到预测
loss_fn：计算预测和真实标签的差距
zero_grad：清空旧梯度
backward：计算新梯度
step：根据梯度更新参数
```

## 6. 验证循环

验证时使用：

```python
model.eval()
with torch.no_grad():
    ...
```

原因：

```text
model.eval()：切换到验证/推理模式
torch.no_grad()：不计算梯度，节省内存和计算
```

验证阶段不更新参数，只计算：

```text
val_loss
val_accuracy
```

## 7. Checkpoint

当验证准确率刷新最好结果时，保存 checkpoint：

```text
outputs/best_mnist_mlp.pt
```

checkpoint 中包含：

```text
epoch
accuracy
model_state_dict
```

`model_state_dict` 是模型参数，是后续加载模型做推理的关键。

## 8. 单张图片推理

推理流程：

```text
加载 checkpoint
读取测试集样本
image.unsqueeze(0) 添加 batch 维度
model(image)
softmax 得到概率
argmax 得到预测类别
```

`unsqueeze(0)` 的原因：模型期望输入是一个 batch：

```text
单张图片：[1, 28, 28]
加 batch 后：[1, 1, 28, 28]
```

## 9. 本次复现实验结果

当前 Windows 电脑使用 CPU 完成训练。

训练命令：

```bash
.\.venv\Scripts\python.exe experiments/day-03-mnist-classifier/train.py --epochs 3 --batch-size 256
```

训练结果：

```text
epoch 01 | train_loss 0.3686 | val_loss 0.2010 | val_acc 0.9397
epoch 02 | train_loss 0.1668 | val_loss 0.1356 | val_acc 0.9603
epoch 03 | train_loss 0.1167 | val_loss 0.1121 | val_acc 0.9651
```

最佳验证准确率：

```text
0.9651
```

推理命令：

```bash
.\.venv\Scripts\python.exe experiments/day-03-mnist-classifier/train.py --infer-only --sample-index 0
```

推理结果：

```text
True label: 7
Prediction: 7
Confidence: 0.9979
```

输出文件：

```text
experiments/day-03-mnist-classifier/outputs/best_mnist_mlp.pt
experiments/day-03-mnist-classifier/outputs/training_curves.png
experiments/day-03-mnist-classifier/outputs/sample_prediction.png
```

## 10. 当前阶段要记住的核心句子

1. `Dataset` 提供单条样本，`DataLoader` 组织 batch。
2. 分类模型输出的是 logits，不是直接输出最终概率。
3. `CrossEntropyLoss` 适合多分类任务。
4. 训练时使用 `model.train()`，验证和推理时使用 `model.eval()`。
5. 推理时使用 `torch.no_grad()`，避免构建计算图。
6. checkpoint 保存的是模型参数和训练元信息。
7. 完整训练脚本必须包含训练、验证、保存、加载和推理。

---

## 2026-05-12 追加学习记录：MNIST、Normalize、Logits、Batch 与 Linear 形状理解

> 本节记录今天围绕 Day 3 MNIST 分类实验的逐步学习过程，重点保留提问、纠正后的理解和关键结论。

### 1. MNIST 的全称与含义

MNIST 全称是：

```text
Modified National Institute of Standards and Technology database
```

中文可以理解为：

```text
改进版美国国家标准与技术研究院手写数字数据库
```

它包含：

```text
手写数字图片 + 对应标签
```

经典规模：

```text
训练集：60,000 张图片
测试集：10,000 张图片
图片大小：28 × 28
图片类型：灰度图
类别：0~9 共 10 类
```

正确理解：

```text
10 分类 = 有 10 个类别，不代表数据量小。
数据量 = 样本数量，例如 60,000 张训练图。
```

MNIST 适合入门，不是因为“10 分类”本身，而是因为：

```text
图片小
数据干净
任务直观
训练速度快
容易验证完整训练流程
```

---

### 2. 灰度图、RGB 与像素值范围

MNIST 是灰度图。

单张 MNIST 图片原始形态可以理解为：

```text
28 × 28
```

每个像素只有一个亮度值。

原始 8-bit 灰度图中，像素通常范围是：

```text
0~255
```

含义：

```text
0 = 黑色
255 = 白色
中间值 = 不同程度的灰色
```

RGB 图片每个像素有 3 个通道：

```text
[R, G, B]
```

每个通道通常也是：

```text
0~255
```

例如：

```text
[255, 0, 0] = 红色
[0, 255, 0] = 绿色
[0, 0, 255] = 蓝色
[128, 128, 128] = 灰色
```

PyTorch 中 MNIST 单张图片经过 `ToTensor()` 后形状为：

```text
[1, 28, 28]
```

其中：

```text
1 = 灰度通道数
28 = 高度
28 = 宽度
```

---

### 3. ToTensor 的作用

`transforms.ToTensor()` 对 MNIST 图片主要做两件事：

1. 把图片转换为 PyTorch 可以处理的张量。
2. 把像素值从 `0~255` 缩放到 `0~1`。

缩放公式可以理解为：

```text
value_01 = value_255 / 255
```

例子：

```text
0   -> 0.0
128 -> 约 0.502
255 -> 1.0
```

关键理解：

```text
0~255 -> 0~1 不是丢弃图片信息，而是线性缩放数值尺度。
```

为什么要缩放到 `0~1`？

```text
神经网络训练更喜欢较小、稳定的输入数值。
如果输入范围过大，loss 和梯度可能更大，训练更不稳定，学习率也更难调。
```

类比 Day 2：

```text
x 在 [-3, 3] 比 x 在 [-3000, 3000] 更容易稳定训练。
```

---

### 4. Normalize 的含义

代码：

```python
transforms.Normalize((0.1307,), (0.3081,))
```

含义：

```text
对图片张量做标准化，让输入数据分布更适合神经网络训练。
```

公式：

```text
x_normalized = (x - mean) / std
```

对于 MNIST：

```text
x_normalized = (x - 0.1307) / 0.3081
```

这里：

```text
0.1307 = MNIST 训练集像素均值 mean
0.3081 = MNIST 训练集像素标准差 std
```

注意：第二个值是标准差，不是方差。

---

### 5. 均值、方差、标准差的直观解释

#### 5.1 均值 mean

均值表示数据的平均水平。

例子：

```text
[0, 0, 1, 1, 1]
```

均值：

```text
mean = (0 + 0 + 1 + 1 + 1) / 5 = 0.6
```

对 MNIST 来说：

```text
mean = 0.1307
```

表示：

```text
MNIST 训练集中所有图片的所有像素，经过 ToTensor 到 0~1 后，平均亮度约为 0.1307。
```

因为 MNIST 大部分是黑色背景，所以均值较低。

#### 5.2 方差 variance

方差衡量数据围绕均值的波动程度。

公式：

```text
variance = 平均((每个值 - 均值)^2)
```

例子：

```text
B = [0, 0, 1, 1, 1]
mean = 0.6
```

每个值减均值：

```text
0 - 0.6 = -0.6
0 - 0.6 = -0.6
1 - 0.6 =  0.4
1 - 0.6 =  0.4
1 - 0.6 =  0.4
```

平方：

```text
0.36, 0.36, 0.16, 0.16, 0.16
```

求平均：

```text
variance = (0.36 + 0.36 + 0.16 + 0.16 + 0.16) / 5 = 0.24
```

#### 5.3 标准差 std

标准差是方差开平方：

```text
std = sqrt(variance)
```

如果：

```text
variance = 0.24
```

那么：

```text
std = sqrt(0.24) ≈ 0.49
```

关系：

```text
variance = std^2
std = sqrt(variance)
```

直观理解：

```text
mean：数据中心在哪里。
variance：数据波动有多大，但单位被平方。
std：数据波动有多大，且和原始数据同量级，更常用。
```

---

### 6. Normalize 的实际效果

Normalize 做两件事：

```text
减去 mean：把数据中心移动到 0 附近。
除以 std：按典型波动幅度缩放数据。
```

例子 1：像素正好等于均值

```text
x = 0.1307
x_norm = (0.1307 - 0.1307) / 0.3081 = 0
```

例子 2：白色像素

```text
x = 1.0
x_norm = (1.0 - 0.1307) / 0.3081 ≈ 2.82
```

含义：

```text
这个像素比平均值亮很多，约高出 2.82 个标准差。
```

例子 3：黑色像素

```text
x = 0.0
x_norm = (0.0 - 0.1307) / 0.3081 ≈ -0.42
```

含义：

```text
这个像素比平均值暗，约低 0.42 个标准差。
```

最终理解：

```text
ToTensor：把 0~255 转为 0~1。
Normalize：把 0~1 再调整成围绕 0 分布的标准化浮点数。
```

---

### 7. Dataset 与 DataLoader

`Dataset` 负责描述数据集，并提供单条样本读取方式。

它解决：

```text
数据在哪里
数据有多少条
第 i 条数据怎么取
每条样本返回什么
是否需要 transform
```

MNIST 中一条样本是：

```text
(image, label)
```

其中：

```text
image = 一张 28 × 28 灰度图
label = 0~9 之间的整数类别
```

`DataLoader` 负责把 Dataset 中的单条样本批量组织成 batch。

例如 `batch_size=256`：

```text
images.shape = [256, 1, 28, 28]
labels.shape = [256]
```

注意：

```text
labels.shape 是 [256]，不是 [256, 10]。
```

原因：

```text
PyTorch 的 CrossEntropyLoss 默认使用类别索引作为标签，而不是 one-hot。
```

模型输出的 logits 才是：

```text
logits.shape = [256, 10]
```

---

### 8. 10 分类、logits 与 labels

MNIST 是 10 分类任务，因为它有 10 个类别：

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

模型不会直接输出一个数字，而是输出 10 个类别分数：

```text
logits = [score_0, score_1, ..., score_9]
```

`logits` 是原始分数，不是概率。

原因：

```text
logits 可以是负数，可以大于 1，所有值加起来也不等于 1。
```

预测类别：

```python
prediction = logits.argmax(dim=1)
```

含义：

```text
对每张图片，在 10 个类别分数中找到最大值所在的位置。
```

例子：

```text
logits = [-1.0, 0.2, 0.5, 2.1, -0.3, 0.0, -2.0, 6.8, 1.2, 3.0]
```

最大值是 `6.8`，位置是 `7`，所以预测数字是：

```text
7
```

---

### 9. 回归任务和分类任务的区别

Day 2 线性回归：

```text
输入：一个数 x
输出：一个连续值 y_hat
标签：连续数值 y
loss：MSELoss
```

Day 3 MNIST 分类：

```text
输入：一张图片
输出：10 个类别分数 logits
标签：类别编号 label
loss：CrossEntropyLoss
```

核心区别：

```text
回归任务输出连续数值。
分类任务输出每个类别的分数。
```

不能让 MNIST 直接输出一个数字再四舍五入，因为类别编号不是连续数值意义。

例如：

```text
数字 9 并不代表比数字 0 “大很多”。
数字类别只是类别编号，不是回归目标。
```

---

### 10. Flatten 如何排列 784 个像素

MNIST 单张图片是：

```text
28 × 28
```

可以看作 28 行、28 列的像素表格。

`Flatten` 会把它摊平成一维：

```text
[784]
```

直观理解为：

```text
从上到下，每一行从左到右，把像素一个一个排列起来。
```

也就是：

```text
第 0 行所有像素 -> 第 1 行所有像素 -> ... -> 第 27 行所有像素
```

所以：

```text
28 × 28 = 784
```

---

### 11. `[256, 784]` 的含义

`[256, 784]` 表示：

```text
一个 batch 中有 256 张图片；
每张图片已经被展平成 784 个数字。
```

可以想象成一个二维表格：

```text
第 0 行：第 0 张图片的 784 个像素
第 1 行：第 1 张图片的 784 个像素
...
第 255 行：第 255 张图片的 784 个像素
```

因此：

```text
x[0][0] = 第 0 张图片展平后的第 0 个像素
x[0][1] = 第 0 张图片展平后的第 1 个像素
x[0][28] = 第 0 张图片展平后的第 28 个像素，通常对应第二行第一个像素
x[1][0] = 第 1 张图片展平后的第 0 个像素
```

关键理解：

```text
[256, 784] 是整个 batch 展平后的形状；
x[0][0] 才是具体某张图片的某个像素值。
```

---

### 12. Linear(784, 128) 的含义

代码：

```python
nn.Linear(784, 128)
```

含义：

```text
输入 784 个数字，输出 128 个数字。
```

也就是：

```text
[batch_size, 784] -> [batch_size, 128]
```

对于 batch size = 256：

```text
[256, 784] -> [256, 128]
```

`Linear` 本身是线性变换，不是非线性函数。

它做的是：

```text
输出 = 输入 × 权重 + 偏置
```

数学形式：

```text
y = xW + b
```

非线性来自后面的：

```python
nn.ReLU()
```

当前模型流程：

```text
Flatten
-> Linear(784, 128)
-> ReLU
-> Linear(128, 10)
-> logits
```

---

### 13. 128 是什么？谁规定的？

`128` 是人为设置的超参数，不是 MNIST 数据集规定的，也不是 PyTorch 规定的。

它叫：

```text
hidden size
hidden dimension
隐藏层维度
隐藏单元数量
```

含义：

```text
把 784 个原始像素特征映射成 128 个模型学习到的隐藏特征。
```

正确理解：

```text
784 是图片尺寸决定的。
128 是模型设计者选择的。
10 是 MNIST 类别数量决定的。
```

为什么选 128？

```text
它是一个折中：
太小可能表达能力不足；
太大参数更多、训练更慢，也可能过拟合。
```

常见隐藏维度可以是：

```text
64 / 128 / 256 / 512
```

---

### 14. 矩阵乘法形状理解

以 batch size = 256 为例：

```text
[256, 784] @ [784, 128] -> [256, 128]
```

含义：

```text
256 张图片，每张图片 784 个输入特征；
通过权重矩阵后，每张图片变成 128 个隐藏特征。
```

形状规则：

```text
前一个矩阵的最后一维 784
必须和后一个矩阵的第一维 784 对上。
```

结果保留外侧维度：

```text
[256, 128]
```

PyTorch 中 `nn.Linear(784, 128)` 的实际权重 shape 是：

```text
[128, 784]
```

但计算等价于：

```text
input @ weight.T + bias
```

所以直观上可以理解为：

```text
[batch_size, 784] @ [784, 128] -> [batch_size, 128]
```

---

### 15. 模型完整 shape 流

以 batch size = 256 为例：

```text
输入 images：        [256, 1, 28, 28]
Flatten 后：         [256, 784]
Linear(784,128)：    [256, 128]
ReLU 后：            [256, 128]
Linear(128,10)：     [256, 10]
argmax(dim=1) 后：   [256]
```

对应含义：

```text
[256, 1, 28, 28]：256 张灰度图
[256, 784]：每张图摊平成 784 个像素
[256, 128]：每张图变成 128 个隐藏特征
[256, 10]：每张图得到 10 个类别分数 logits
[256]：每张图得到 1 个预测类别
```

---

### 16. labels 与 predictions 的正确理解

如果：

```text
labels = [7, 2, 1, 0]
predictions = [7, 3, 1, 0]
```

逐个比较：

```text
第 1 张：prediction 7，label 7 -> 对
第 2 张：prediction 3，label 2 -> 错
第 3 张：prediction 1，label 1 -> 对
第 4 张：prediction 0，label 0 -> 对
```

所以：

```text
预测对了 3 张
准确率 = 3 / 4 = 75%
```

注意：

```text
labels[0] = 7 表示第 0 个样本，也就是第一张图片的真实标签是数字 7。
```

不是数字 6。

索引从 0 开始，但标签值本身仍然表示数字类别。

---

### 17. 小练习记录

#### 练习 1：logits 判断

给定：

```text
logits = [-1.0, 0.2, 0.5, 2.1, -0.3, 0.0, -2.0, 6.8, 1.2, 3.0]
```

正确理解：

```text
最大值是 6.8，位于索引 7，所以预测数字是 7。
logits 不是概率，因为可以有负数、大于 1，且总和不等于 1。
```

#### 练习 2：batch 准确率

给定：

```text
batch_size = 3
labels = [2, 0, 9]
predictions = [2, 5, 9]
```

正确答案：

```text
一共有 3 张图片。
第 2 张图片真实标签是 0。
第 2 张图片预测成 5。
预测对了 2 张。
准确率 = 2/3 ≈ 66.7%。
```

#### 练习 3：小模型 shape 流

给定：

```text
图片大小：4 × 4
类别数量：3 类
batch_size：5
隐藏层维度：8
模型：Flatten -> Linear(16, 8) -> ReLU -> Linear(8, 3)
```

正确答案：

```text
输入 batch shape：     [5, 1, 4, 4]
Flatten 后：           [5, 16]
第一层 Linear 后：     [5, 8]
最后 logits shape：    [5, 3]
argmax 后：            [5]
```

---

### 18. 当前阶段必须记住的关键句子

1. MNIST 是手写数字 10 分类任务，10 分类指 10 个类别，不代表数据量小。
2. 灰度图只有 1 个通道，RGB 图有 3 个通道。
3. 原始 8-bit 像素通常是 0~255，`ToTensor()` 会缩放到 0~1。
4. `Normalize(mean, std)` 会执行 `(x - mean) / std`。
5. `0.1307` 是 MNIST 像素均值，`0.3081` 是 MNIST 像素标准差。
6. `Dataset` 提供单条样本，`DataLoader` 组织 batch。
7. `labels.shape` 是 `[batch_size]`，`logits.shape` 是 `[batch_size, num_classes]`。
8. `Flatten` 只是改变形状，不学习参数。
9. `Linear` 学习权重和偏置，执行线性变换。
10. `ReLU` 提供非线性能力。
11. `784` 来自 `28×28`，`128` 是人为选择的隐藏维度，`10` 来自 MNIST 类别数。
12. `argmax(dim=1)` 对每张图片的类别分数取最大值位置，得到预测类别。

---

## 2026-05-12 追加学习记录：Linear 隐藏特征、ReLU、CrossEntropyLoss 与反向传播疑问

> 记录目的：保留截至目前的对话细节，方便后续切换电脑或由其他智能体继续接上学习进度。  
> 当前学习位置：Day 3 MNIST MLP 模型类与分类训练机制。  
> 当前重点：理解 `Linear(784, 128)`、`Linear(128, 10)`、`ReLU`、`CrossEntropyLoss`、分类任务反向传播和 Day 2 回归任务的区别。

### 1. 学习者当前提出的核心理解

学习者对 `Linear(784, 128)` 的理解：

```text
784 来自输入图像的张量数据铺平展开，也就是 28×28。
128 是模型中隐藏层的维度。
一张图的 784 个像素数据，每个像素值可以理解为灰度值。
某一个隐藏维度可以理解为：784 个像素值分别乘以一组权重 w1~w784，再加权求和得到一个新的值。
这个新的值就是 128 维隐藏表示中的某个维度。
```

学习者进一步追问：

```text
这个隐藏维度是否可以理解为描述图片的锐利程度，或者描述图片部分信息的特征？
经过不断训练是否能接近 w1~w784 的真理性参数？
预测时是否对每个输入值都有一个预测值和差值？
然后做梯度，计算每个 w 下一轮的取值方向？
Day 2 和 Day 3 的事实差别到底在哪里？
```

对上述理解的校准：

```text
Linear(784,128) 的确是在把 784 个原始像素值映射成 128 个隐藏特征值。
每个隐藏特征都有自己的一组 784 个权重和 1 个 bias。
128 个隐藏特征不是人工定义好语义的，单个维度不一定能直接解释为“锐利程度”等人类概念。
更准确地说，它们是模型学出来的抽象特征响应，通常是分布式表示。
训练不是寻找唯一的“真理性参数”，而是在寻找一组能让分类 loss 降低、验证效果变好的参数。
```

---

### 2. 学习者对练习题的回答与校准

练习问题：

```text
1. Linear(784, 128) 中的 784、128 分别来自哪里？
2. 128 个隐藏特征是人工定义好含义的吗？
3. 如果没有 ReLU，两个 Linear 层叠起来会有什么问题？
4. Day 3 中 loss 的目标是让输出值接近某个连续数，还是让正确类别的分数相对更高？
5. loss.backward() 会只更新最后一层 Linear 的参数吗？还是也会影响第一层 Linear？
```

学习者回答：

```text
1. 784 来自输入图像的张量数据的铺平展开，也就是 28×28。
2. 128 是模型自己学习得到的，人类很难描述其中隐藏的含义。
3. 没有 ReLU 会使训练出来的模型表达能力较弱，本质上识别图片不是一个线性函数能直接表达的。
4. Day 3 的目标是让正确类别的分数相对更高。
5. loss.backward() 会更新最后一层到第一层中所有可学习参数的数值。
```

校准：

```text
第 1 点中，784 的理解正确。
但 128 不是“内置参数矩阵”，而是隐藏层输出维度 / 隐藏特征数量。
真正的参数矩阵是连接 784 -> 128 的 weight，它的形状和 128 有关。
第 2、3、4、5 点大方向正确。
```

---

### 3. 学习者当前最大的疑问

学习者对以下表述感到抽象：

```text
CrossEntropyLoss 会产生一个较大的 loss。
反向传播会计算：为了让 loss 下降，各个参数应该怎么调整。
整体方向大致是：让类别 7 的分数升高，让类别 3 等错误类别的分数降低。
但它不会只调整最后一层。
梯度会继续往前传：
loss
↓
第二层 Linear(128,10)
↓
ReLU
↓
第一层 Linear(784,128)
```

具体疑问：

```text
CrossEntropyLoss 是如何计算的？
Day 2 的梯度计算比较直观，但 Day 3 分类任务这里比较抽象。
Linear(128,10) 是否也有一个将 128 转为 10 的内置参数矩阵？
这一块很抽象。
```

---

### 4. 最后一段完整讲解留存

以下为针对上述问题的完整讲解，后续学习可以从这里继续。

你这组回答整体非常好。先校准一下：

1. `784`：对，来自 `28×28` 展平后的输入维度。  
2. `128`：这里要修正，`128` 不是“内置参数矩阵”，而是 **隐藏层输出维度 / 隐藏特征数量**。真正的参数矩阵是连接 `784 -> 128` 的 `weight`，它的规模和 `128` 有关。  
3. 没有 ReLU：你答得对，表达能力弱。  
4. Day 3 loss 目标：对，让正确类别分数相对更高。  
5. 反向传播：对，会影响从最后一层往前的所有可学习参数，只要它们参与了 loss 的计算。

现在重点解释你卡住的地方：**CrossEntropyLoss 到底怎么算？第二层 Linear(128,10) 到底是什么？反向传播怎么从 loss 回到前面的层？**

---

# 1. 先看第二层 `Linear(128, 10)`

你问：

> `Linear(128,10)` 其实也是一个将 128 转为 10 的内置参数矩阵吗？

是的。

它和第一层很像。

第一层是：

```text
Linear(784, 128)
```

作用：

```text
784 个像素特征 -> 128 个隐藏特征
```

第二层是：

```text
Linear(128, 10)
```

作用：

```text
128 个隐藏特征 -> 10 个类别分数
```

也就是：

```text
h:      [128]
W2:     [128, 10]
bias2:  [10]
logits: [10]
```

计算：

```text
logits = h @ W2 + b2
```

对于 batch：

```text
H:      [batch, 128]
W2:     [128, 10]
bias2:  [10]
logits: [batch, 10]
```

比如 batch size = 256：

```text
[256, 128] @ [128, 10] -> [256, 10]
```

所以你理解得对：

```text
第二层也有一组可学习权重，用来把 128 维隐藏特征转换成 10 个类别分数。
```

---

# 2. 10 个 logits 是怎么来的？

假设某张图片经过第一层和 ReLU 后，得到 128 个隐藏特征：

```text
h = [h1, h2, h3, ..., h128]
```

第二层要输出 10 个类别分数：

```text
score_0, score_1, ..., score_9
```

其中每个类别分数都是由这 128 个隐藏特征加权求和得到的。

例如：

```text
score_0 = h1*w1,0 + h2*w2,0 + ... + h128*w128,0 + b0
score_1 = h1*w1,1 + h2*w2,1 + ... + h128*w128,1 + b1
...
score_7 = h1*w1,7 + h2*w2,7 + ... + h128*w128,7 + b7
...
score_9 = h1*w1,9 + h2*w2,9 + ... + h128*w128,9 + b9
```

也就是说：

```text
每个类别都有一套自己的 128 个权重。
```

可以直观理解为：

```text
类别 0 有一套判断规则
类别 1 有一套判断规则
...
类别 7 有一套判断规则
...
类别 9 有一套判断规则
```

这些规则不是人工写的，是训练学出来的。

---

# 3. CrossEntropyLoss 为什么会“大”？

我们用一个具体例子。

假设一张图片真实标签是：

```text
label = 7
```

模型输出 logits：

```text
logits = [1.0, 0.5, 0.2, 3.0, -0.5, 0.1, -1.0, 0.7, 0.3, 0.9]
```

这里最高分是：

```text
score_3 = 3.0
```

真实类别 `7` 的分数是：

```text
score_7 = 0.7
```

模型预测错了，因为它更相信是 `3`，不是 `7`。

CrossEntropyLoss 会惩罚这种情况。

---

# 4. CrossEntropyLoss 的两步直觉

对于分类任务，CrossEntropyLoss 可以粗略理解为两步：

```text
第一步：把 logits 转成概率 softmax
第二步：看真实类别对应的概率有多高
```

如果真实类别概率高：

```text
loss 小
```

如果真实类别概率低：

```text
loss 大
```

---

# 5. softmax 做什么？

softmax 把 logits 转成概率。

公式先不用背，理解它做三件事：

```text
1. 每个类别得到一个 0~1 之间的概率
2. 所有类别概率加起来等于 1
3. 原始 logits 越大，对应概率越大
```

例如 logits：

```text
[1.0, 0.5, 0.2, 3.0, -0.5, 0.1, -1.0, 0.7, 0.3, 0.9]
```

softmax 后可能大致类似：

```text
类别 3 概率最高
类别 7 概率不高
```

假设真实类别 7 的概率只有：

```text
p_7 = 0.05
```

也就是模型只给正确答案 5% 的概率。

这时 loss 会比较大。

如果训练后模型给：

```text
p_7 = 0.95
```

那 loss 会很小。

---

# 6. CrossEntropyLoss 实际看的是正确类别概率

对于单个样本，CrossEntropyLoss 可以直观理解为：

```text
loss = -log(正确类别的概率)
```

也就是：

```text
loss = -log(p_true)
```

如果真实标签是 7：

```text
loss = -log(p_7)
```

你不需要现在精通 log，只要知道这个趋势：

```text
p_true 越接近 1，loss 越接近 0
p_true 越接近 0，loss 越大
```

举几个数：

```text
p_true = 0.90 -> loss ≈ 0.105，很小
p_true = 0.50 -> loss ≈ 0.693
p_true = 0.10 -> loss ≈ 2.303
p_true = 0.01 -> loss ≈ 4.605，很大
```

所以：

```text
如果模型不给正确类别足够高的概率，CrossEntropyLoss 就会变大。
```

---

# 7. 为什么它会让类别 7 分数升高、类别 3 分数降低？

这是 softmax + cross entropy 的梯度特性。

对于每个类别的 logit，梯度大致是：

```text
gradient = predicted_probability - true_label_indicator
```

其中：

```text
predicted_probability = softmax 后该类别的概率
true_label_indicator = 如果是真实类别就是 1，否则是 0
```

假设真实类别是 7。

对于类别 7：

```text
gradient_7 = p_7 - 1
```

如果：

```text
p_7 = 0.05
```

那么：

```text
gradient_7 = 0.05 - 1 = -0.95
```

梯度是负的。

参数更新大致是：

```text
参数_new = 参数_old - lr * gradient
```

梯度为负时：

```text
减去负数 = 增加
```

所以会推动：

```text
score_7 变大
```

对于错误类别 3：

```text
gradient_3 = p_3 - 0
```

如果：

```text
p_3 = 0.60
```

那么：

```text
gradient_3 = 0.60
```

梯度是正的。

更新时：

```text
参数_new = 参数_old - lr * 正数
```

所以会推动：

```text
score_3 变小
```

这就是为什么我说：

```text
提高正确类别分数，压低错误类别分数。
```

---

# 8. 它如何影响第二层 Linear？

第二层是：

```text
score_7 = h1*w1,7 + h2*w2,7 + ... + h128*w128,7 + b7
```

如果真实类别是 7，但 `score_7` 太低，loss 会告诉模型：

```text
让 score_7 变高
```

那怎么让 `score_7` 变高？

可以调整：

```text
w1,7, w2,7, ..., w128,7, b7
```

也就是类别 7 对应的那一列权重。

同时，如果错误类别 3 分数太高：

```text
score_3 = h1*w1,3 + h2*w2,3 + ... + h128*w128,3 + b3
```

loss 会告诉模型：

```text
让 score_3 变低
```

于是会调整类别 3 对应的那一列权重。

所以第二层 Linear 的参数会被更新。

---

# 9. 它又如何影响第一层 Linear？

这是你觉得抽象的地方。

第二层的输入是：

```text
h = [h1, h2, ..., h128]
```

如果 `h` 的某些值导致错误类别分数高、正确类别分数低，那么模型不只会调整第二层权重，也会调整产生 `h` 的前面部分。

因为：

```text
h 来自第一层 Linear + ReLU
```

也就是：

```text
h = ReLU(x @ W1 + b1)
```

所以 loss 也会反向告诉第一层：

```text
你产生的这些隐藏特征 h 不够好，需要调整 W1 和 b1，
让下次生成更有利于正确分类的隐藏特征。
```

因此梯度链条是：

```text
loss
↓
logits
↓
第二层 W2, b2
↓
隐藏特征 h
↓
ReLU
↓
第一层 W1, b1
```

这就是“反向传播”。

---

# 10. 用一句话解释反向传播在这里做什么

对于 MNIST：

```text
反向传播会计算：
为了让正确类别的概率更高、错误类别的概率更低，
第一层和第二层的每个权重、每个 bias 应该往哪个方向调整。
```

---

# 11. 和 Day 2 的共同点

Day 2：

```text
loss = (pred - y)^2
```

梯度告诉：

```text
w 和 b 怎么变，才能让 pred 接近 y
```

Day 3：

```text
loss = -log(正确类别概率)
```

梯度告诉：

```text
W1, b1, W2, b2 怎么变，才能让正确类别概率更高
```

共同点：

```text
都是定义一个 loss；
都是用 loss.backward() 计算参数梯度；
都是用 optimizer.step() 更新参数；
都是让 loss 下降。
```

不同点：

```text
Day 2 的 loss 来自数值差距。
Day 3 的 loss 来自分类概率分布和真实类别之间的差距。
```

---

# 12. 当前事实模型

当前 MNIST MLP 的事实过程是：

```text
图片 x
↓
Flatten 成 784 个像素值
↓
第一层 Linear 用 W1,b1 生成 128 个隐藏特征
↓
ReLU 保留正激活，关闭负激活
↓
第二层 Linear 用 W2,b2 生成 10 个 logits
↓
softmax 把 logits 转成 10 个概率
↓
CrossEntropyLoss 只看真实类别对应的概率是否足够高
↓
loss.backward() 计算 W1,b1,W2,b2 的梯度
↓
optimizer.step() 更新 W1,b1,W2,b2
```

注意：PyTorch 的 `CrossEntropyLoss` 内部会高效地结合 `log_softmax + negative log likelihood`，所以代码里不需要你手动先写 softmax。

---

# 13. 为什么代码里没有显式 softmax？

可能的问题：

```text
既然 CrossEntropyLoss 要看概率，为什么代码里没有 softmax？
```

原因：

```python
nn.CrossEntropyLoss()
```

内部已经包含了：

```text
log_softmax + negative log likelihood loss
```

所以它要求输入是：

```text
原始 logits
```

而不是 softmax 后的概率。

训练时：

```python
loss = loss_fn(logits, labels)
```

不要写成：

```python
loss = loss_fn(torch.softmax(logits), labels)
```

预测/展示置信度时才用 softmax：

```python
probabilities = torch.softmax(logits, dim=1)
```

---

# 14. CrossEntropyLoss、logits 和梯度的深入理解

本节记录目前对 Day 3 中最关键问题的理解：

```text
CrossEntropyLoss 到底如何从 logits 计算 loss？
loss 又如何通过梯度影响最后一层 Linear 的 weight 和 bias？
```

---

## 14.1 logits 的位置含义是谁规定的？

MNIST 有 10 个类别：

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

模型最后输出 10 个 logits：

```text
[z_0, z_1, z_2, z_3, z_4, z_5, z_6, z_7, z_8, z_9]
```

这里的含义是我们通过标签约定出来的：

```text
z_0 表示数字 0 的原始分数
z_1 表示数字 1 的原始分数
...
z_9 表示数字 9 的原始分数
```

这个顺序不是神经网络天然知道的，而是训练数据标签和 `CrossEntropyLoss` 的使用方式共同规定的。

在 PyTorch 中：

```python
loss = loss_fn(logits, labels)
```

如果某张图片的真实标签是：

```text
label = 7
```

那么 `CrossEntropyLoss` 会认为：

```text
第 7 个位置对应正确类别
```

也就是希望 `z_7` 经过 softmax 后的概率变高。

理论上，也可以人为规定：

```text
第 9 个位置表示数字 0
```

但前提是训练、验证、推理时都使用同一套标签映射。类别位置本身只是约定，关键是映射必须前后一致。

---

## 14.2 重要符号说明

为了避免公式产生歧义，先定义所有符号。

```text
i
```

表示类别下标。

MNIST 中：

```text
i = 0, 1, 2, ..., 9
```

```text
z_i
```

表示第 `i` 个类别的 logit，也就是模型输出的原始分数。

例如：

```text
z_7 表示数字 7 的原始分数
```

```text
exp(x)
```

表示：

```text
e^x
```

其中：

```text
e ≈ 2.71828
```

所以：

```text
exp(0) = e^0 = 1
exp(1) = e^1 ≈ 2.718
exp(2) = e^2 ≈ 7.389
```

softmax 使用 `exp` 的一个原因是：它可以把任意 logit 转成正数，然后再归一化成概率。

```text
ln(x)
```

表示自然对数，也就是以 `e` 为底的对数。

机器学习公式里常写 `log(x)`，但这里为了避免歧义，统一理解为：

```text
log(x) = ln(x)
```

`exp` 和 `ln` 是反函数：

```text
ln(exp(x)) = x
```

也就是：

```text
ln(e^x) = x
```

所以：

```text
ln(exp(z_0)) = z_0
```

```text
S
```

表示所有类别的 `exp(logit)` 总和。

如果只有 3 类：

```text
S = exp(z_0) + exp(z_1) + exp(z_2)
```

如果是 MNIST 10 类：

```text
S = exp(z_0) + exp(z_1) + ... + exp(z_9)
```

```text
p_i
```

表示 softmax 后第 `i` 个类别的预测概率。

定义是：

```text
p_i = exp(z_i) / S
```

所以：

```text
p_0 = exp(z_0) / S
p_1 = exp(z_1) / S
...
p_9 = exp(z_9) / S
```

所有概率加起来等于 1：

```text
p_0 + p_1 + ... + p_9 = 1
```

```text
y_i
```

表示真实标签的 one-hot 值。

如果真实标签是数字 `7`，那么 one-hot 是：

```text
[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
```

也就是：

```text
y_7 = 1
y_0 = 0
y_1 = 0
...
y_9 = 0
```

含义是：

```text
正确类别希望概率是 1
错误类别希望概率是 0
```

```text
L
```

表示单张图片的 loss。

分类任务中可以理解为：

```text
模型给正确类别的概率越高，L 越小。
模型给正确类别的概率越低，L 越大。
```

```text
dL/dz_i
```

读作：

```text
L 对 z_i 的导数
```

含义是：

```text
如果 z_i 稍微变大一点，L 会如何变化。
```

如果：

```text
dL/dz_i > 0
```

说明增大 `z_i` 会让 loss 变大，训练会倾向于压低它。

如果：

```text
dL/dz_i < 0
```

说明增大 `z_i` 会让 loss 变小，训练会倾向于提高它。

---

## 14.3 softmax 和 CrossEntropyLoss 的关系

假设为了简化，只有 3 个类别：

```text
z_0, z_1, z_2
```

softmax 的总和是：

```text
S = exp(z_0) + exp(z_1) + exp(z_2)
```

每个类别的概率是：

```text
p_0 = exp(z_0) / S
p_1 = exp(z_1) / S
p_2 = exp(z_2) / S
```

假设真实类别是 `0`。

那么 one-hot 是：

```text
y_0 = 1
y_1 = 0
y_2 = 0
```

单张图片的 CrossEntropyLoss 是：

```text
L = -ln(p_0)
```

因为真实类别是 `0`，所以只看正确类别概率 `p_0`。

如果：

```text
p_0 = 0.90
```

那么：

```text
L = -ln(0.90) ≈ 0.105
```

如果：

```text
p_0 = 0.10
```

那么：

```text
L = -ln(0.10) ≈ 2.303
```

所以：

```text
正确类别概率越低，loss 越大。
```

---

## 14.4 为什么 dL/dz_i = p_i - y_i？

继续使用 3 类例子。

真实类别是 `0`，所以：

```text
L = -ln(p_0)
```

而 softmax 定义中：

```text
p_0 = exp(z_0) / S
```

代入：

```text
L = -ln(exp(z_0) / S)
```

根据自然对数规则：

```text
ln(a / b) = ln(a) - ln(b)
```

所以：

```text
L = -[ln(exp(z_0)) - ln(S)]
```

因为：

```text
ln(exp(z_0)) = z_0
```

所以：

```text
L = -z_0 + ln(S)
```

展开 `S`：

```text
L = -z_0 + ln(exp(z_0) + exp(z_1) + exp(z_2))
```

现在分别看 `L` 对不同 logit 的导数。

### 对正确类别 z_0 求导

```text
L = -z_0 + ln(S)
```

第一部分：

```text
d(-z_0)/dz_0 = -1
```

第二部分：

```text
d(ln(S))/dz_0
```

这里要用链式法则：

```text
d(ln(S))/dz_0 = d(ln(S))/dS * dS/dz_0
```

其中：

```text
d(ln(S))/dS = 1 / S
```

而：

```text
S = exp(z_0) + exp(z_1) + exp(z_2)
```

所以：

```text
dS/dz_0 = exp(z_0)
```

因此：

```text
d(ln(S))/dz_0 = (1 / S) * exp(z_0)
```

也就是：

```text
d(ln(S))/dz_0 = exp(z_0) / S
```

根据 softmax 定义：

```text
p_0 = exp(z_0) / S
```

所以：

```text
d(ln(S))/dz_0 = p_0
```

两部分相加：

```text
dL/dz_0 = -1 + p_0
```

也就是：

```text
dL/dz_0 = p_0 - 1
```

因为真实类别是 `0`，所以：

```text
y_0 = 1
```

因此：

```text
dL/dz_0 = p_0 - y_0
```

### 对错误类别 z_1 求导

仍然是：

```text
L = -z_0 + ln(S)
```

对 `z_1` 求导。

第一部分：

```text
d(-z_0)/dz_1 = 0
```

因为 `z_0` 和 `z_1` 是两个不同位置的 logit。

第二部分：

```text
d(ln(S))/dz_1 = d(ln(S))/dS * dS/dz_1
```

其中：

```text
d(ln(S))/dS = 1 / S
```

并且：

```text
dS/dz_1 = exp(z_1)
```

所以：

```text
d(ln(S))/dz_1 = exp(z_1) / S
```

根据 softmax 定义：

```text
p_1 = exp(z_1) / S
```

因此：

```text
dL/dz_1 = 0 + p_1
```

也就是：

```text
dL/dz_1 = p_1
```

因为真实类别不是 `1`，所以：

```text
y_1 = 0
```

因此：

```text
dL/dz_1 = p_1 - y_1
```

同理：

```text
dL/dz_2 = p_2 - y_2
```

最终统一写成：

```text
dL/dz_i = p_i - y_i
```

这句话的准确含义是：

```text
每个 logit 的梯度 = 模型给这个类别的预测概率 - 真实标签希望这个类别拥有的概率
```

正确类别的 `y_i = 1`，错误类别的 `y_i = 0`。

---

## 14.5 这个梯度如何推动正确类别上升？

假设真实类别是 `7`。

如果模型现在给数字 `7` 的概率很低：

```text
p_7 = 0.10
```

因为：

```text
y_7 = 1
```

所以：

```text
dL/dz_7 = p_7 - y_7 = 0.10 - 1 = -0.90
```

这个梯度是负数。

梯度下降的更新方向是：

```text
参数 = 参数 - learning_rate * 梯度
```

所以负梯度会让相关参数朝着“增大 `z_7`”的方向变化。

如果错误类别 `3` 的概率很高：

```text
p_3 = 0.70
```

因为真实类别不是 `3`：

```text
y_3 = 0
```

所以：

```text
dL/dz_3 = p_3 - y_3 = 0.70 - 0 = 0.70
```

这个梯度是正数。

梯度下降会让相关参数朝着“减小 `z_3`”的方向变化。

所以 CrossEntropyLoss 的核心作用是：

```text
提高正确类别的 logit。
压低错误类别的 logit。
```

---

## 14.6 最后一层 Linear 的 weight 和 bias 如何被影响？

MNIST MLP 的最后一层是：

```python
nn.Linear(128, 10)
```

它的含义是：

```text
输入 hidden 有 128 个特征。
输出 logits 有 10 个类别分数。
```

对单张图片：

```text
hidden = [h_1, h_2, ..., h_128]
```

第 `i` 个类别的 logit 是：

```text
z_i = W_i,1 * h_1 + W_i,2 * h_2 + ... + W_i,128 * h_128 + b_i
```

其中：

```text
W_i,k
```

表示最后一层中“第 `i` 个类别、第 `k` 个 hidden 特征”对应的 weight。

```text
b_i
```

表示第 `i` 个类别对应的 bias。

PyTorch 中这层参数形状是：

```text
weight.shape = [10, 128]
bias.shape = [10]
```

现在看某一个具体参数：

```text
W_7,k
```

它只直接参与计算：

```text
z_7
```

因为：

```text
z_7 = W_7,1 * h_1 + ... + W_7,k * h_k + ... + W_7,128 * h_128 + b_7
```

所以根据链式法则：

```text
dL/dW_7,k = dL/dz_7 * dz_7/dW_7,k
```

又因为：

```text
dz_7/dW_7,k = h_k
```

所以：

```text
dL/dW_7,k = (dL/dz_7) * h_k
```

而前面已经得到：

```text
dL/dz_7 = p_7 - y_7
```

因此：

```text
dL/dW_7,k = (p_7 - y_7) * h_k
```

这个值就是参数 `W_7,k` 的梯度。

也就是 PyTorch 里：

```text
weight.grad[7, k]
```

类似地：

```text
dL/db_7 = dL/dz_7 * dz_7/db_7
```

因为：

```text
dz_7/db_7 = 1
```

所以：

```text
dL/db_7 = p_7 - y_7
```

也就是：

```text
bias.grad[7]
```

对任意类别 `i` 和任意 hidden 位置 `k`，统一公式是：

```text
dL/dW_i,k = (p_i - y_i) * h_k
```

```text
dL/db_i = p_i - y_i
```

这说明最后一层的全部参数都会被影响：

```text
10 个类别 * 128 个 hidden 特征 = 1280 个 weight
再加 10 个 bias
```

每个 weight 都有自己的梯度。

---

## 14.7 batch size = 256 时如何计算梯度？

如果 batch size 是 256，那么不是一张图片更新一次参数，而是：

```text
256 张图片一起 forward。
256 张图片各自产生 loss。
PyTorch 默认对 256 个 loss 求平均。
然后 backward 得到平均梯度。
最后 optimizer.step() 更新一次参数。
```

设：

```text
n
```

表示 batch 中第 `n` 张图片。

```text
n = 1, 2, ..., 256
```

第 `n` 张图片的第 `i` 个类别概率是：

```text
p_n,i
```

第 `n` 张图片的 one-hot 标签是：

```text
y_n,i
```

第 `n` 张图片的第 `k` 个 hidden 特征是：

```text
h_n,k
```

那么单张图片对参数 `W_i,k` 的梯度贡献是：

```text
(p_n,i - y_n,i) * h_n,k
```

因为 `CrossEntropyLoss` 默认是：

```python
reduction="mean"
```

所以 batch 的最终梯度是 256 张图片的平均值：

```text
dL_batch/dW_i,k =
[(p_1,i - y_1,i) * h_1,k
+ (p_2,i - y_2,i) * h_2,k
+ ...
+ (p_256,i - y_256,i) * h_256,k] / 256
```

这个平均值会存到：

```text
weight.grad[i, k]
```

然后执行：

```python
optimizer.step()
```

参数更新为：

```text
W_i,k = W_i,k - learning_rate * weight.grad[i, k]
```

所以更准确地说：

```text
一个 batch 中的 256 张图片共同决定一次参数更新方向。
```

---

## 14.8 当前理解总结

目前可以这样理解 Day 3 的分类训练：

```text
1. 最后一层 Linear 把 128 维 hidden 特征转成 10 个 logits。
2. logits 的位置含义由标签约定决定，比如第 7 个位置表示数字 7。
3. CrossEntropyLoss 内部会对 logits 做 log_softmax，本质等价于 softmax + 负对数损失。
4. exp(x) = e^x，ln(x) 是以 e 为底的自然对数，ln(exp(x)) = x。
5. softmax 中 p_i = exp(z_i) / S，S 是所有 exp(logit) 的总和。
6. 对单张图片，CrossEntropyLoss 对每个 logit 的梯度是 dL/dz_i = p_i - y_i。
7. 正确类别的 y_i = 1，错误类别的 y_i = 0。
8. 正确类别概率低时，p_i - 1 是负数，训练会推动正确类别 logit 上升。
9. 错误类别概率高时，p_i - 0 是正数，训练会推动错误类别 logit 下降。
10. 最后一层某个 weight 的梯度是 dL/dW_i,k = (p_i - y_i) * h_k。
11. batch size = 256 时，最终梯度是 256 张图片梯度贡献的平均值。
12. loss.backward() 负责自动计算这些梯度，并写入 weight.grad 和 bias.grad。
13. optimizer.step() 根据这些梯度真正修改 weight 和 bias。
```
