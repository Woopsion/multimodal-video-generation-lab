# Day 4：CIFAR-10 CNN 训练脚本

目标：把 Day 3 的 MNIST MLP 训练闭环迁移到更真实的彩色图像分类任务，并理解 CNN 为什么适合图像。

## 运行训练

```bash
.\.venv\Scripts\python.exe experiments/day-04-cifar10-cnn/train.py --epochs 5 --batch-size 128
```

如果当前机器没有 CIFAR-10 数据，脚本会尝试下载到：

```text
experiments/day-04-cifar10-cnn/data/
```

## 运行推理

训练完成后，可以对测试集中的一张图片做推理：

```bash
.\.venv\Scripts\python.exe experiments/day-04-cifar10-cnn/train.py --infer-only --sample-index 0
```

## 模型结构

```text
3x32x32 image
-> Conv2d(3, 32, 3, padding=1) -> BatchNorm2d -> ReLU -> MaxPool2d
-> Conv2d(32, 64, 3, padding=1) -> BatchNorm2d -> ReLU -> MaxPool2d
-> Conv2d(64, 128, 3, padding=1) -> BatchNorm2d -> ReLU -> MaxPool2d
-> Flatten
-> Linear(128 * 4 * 4, 256) -> ReLU -> Dropout
-> Linear(256, 10)
-> logits
```

## 每一层的直觉

- `Conv2d`：用一组小卷积核扫描图片，提取局部特征，例如边缘、颜色块和纹理。
- `BatchNorm2d`：稳定每层特征分布，让训练更平稳。
- `ReLU`：加入非线性，否则多层线性变换本质上仍接近一个线性模型。
- `MaxPool2d`：降低空间尺寸，保留强响应，让模型更省算力、更关注显著特征。
- `Flatten`：把卷积特征图摊平成向量，交给全连接分类器。
- `Dropout`：训练时随机丢弃部分特征，降低过拟合。
- `Linear(256, 10)`：输出 10 个 logits，对应 CIFAR-10 的 10 个类别。

## 输出文件

- `outputs/best_cifar10_cnn.pt`：验证准确率最高的 checkpoint。
- `outputs/training_curves.png`：训练 loss 和验证 accuracy 曲线。
- `outputs/sample_prediction.png`：单张样本推理结果。
- `logs/gpu-benchmark.md`：batch size、耗时、显存记录。

## 和 Day 3 的关系

Day 3 的 MLP 把 MNIST 图片从 `[batch, 1, 28, 28]` 压平成 `[batch, 784]`，然后分类。

Day 4 的 CNN 保留 CIFAR-10 图片的 `[batch, 3, 32, 32]` 空间结构，先提取局部图像特征，再分类。

```text
Day 3: image -> Flatten -> Linear -> ReLU -> Linear -> logits
Day 4: image -> Conv/Pool feature extractor -> Flatten -> Linear -> logits
```
