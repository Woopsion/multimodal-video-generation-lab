# Day 3：MNIST 完整训练脚本

目标：完成一个最小但完整的 PyTorch 监督学习训练闭环。

## 运行训练

```bash
.\.venv\Scripts\python.exe experiments/day-03-mnist-classifier/train.py --epochs 3
```

如果当前机器没有 MNIST 数据，脚本会尝试下载到：

```text
experiments/day-03-mnist-classifier/data/
```

## 运行推理

训练完成后，可以对测试集中的一张图片做推理：

```bash
.\.venv\Scripts\python.exe experiments/day-03-mnist-classifier/train.py --infer-only --sample-index 0
```

## 本实验要理解的闭环

```text
Dataset -> DataLoader -> Model -> Loss -> Backward -> Optimizer -> Validation -> Checkpoint -> Inference
```

## 输出文件

- `outputs/best_mnist_mlp.pt`：验证准确率最高的 checkpoint。
- `outputs/training_curves.png`：训练 loss 和验证 accuracy 曲线。
- `outputs/sample_prediction.png`：单张样本推理结果。

## 重点概念

- `Dataset`：表示数据集，负责提供单条样本。
- `DataLoader`：批量读取数据，并支持 shuffle。
- `model.train()`：启用训练模式。
- `model.eval()`：启用验证/推理模式。
- `torch.no_grad()`：推理时不构建计算图，节省内存。
- `checkpoint`：保存模型参数和训练元信息。

