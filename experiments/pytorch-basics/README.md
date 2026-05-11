# Day 2：PyTorch 张量、自动微分与线性回归

运行：

```bash
uv run python experiments/pytorch-basics/linear_regression.py
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

