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
