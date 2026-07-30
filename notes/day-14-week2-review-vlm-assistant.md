# Day 14 - Week2 复盘与 VLM 小助手设计

> 当前阶段：Day14 已按“学习型最小 MVP 设计”收口。真实 Demo 编码和 10 张图片评测延期到后续实现阶段。

## 学习目标

- 把 Day8-Day13 串成完整的 VLM 数据流。
- 从学习目标反推输入、输出、模型和失败边界。
- 形成可执行的 `vlm-visual-assistant` 最小技术方案。

## 主学习入口

- HTML 教学文档：`docs/day-14-week2-review-vlm-assistant/index.html`
- 技术方案：`demos/vlm-visual-assistant/design.md`

## Step 01：Week2 知识地图

```text
image -> patch tokens -> CLIP representation -> retrieval / alignment
      -> image tokens + question tokens -> LLM -> answer
```

- Day8：图片切 patch 并形成视觉 token 序列。
- Day9-Day11：CLIP 图文对齐、检索应用和论文依据。
- Day12：观察训练完成的 VLM 如何执行推理。
- Day13：观察视觉问答能力如何通过数据、loss 和参数更新训练出来。

## Step 02：需求契约

- 目标：用于学习和验证的通用视觉问答 Demo。
- 输入：单张图片路径 + 文本问题 + 可选任务模式。
- 任务：软件截图、基础图表、普通照片。
- 输出：回答 + 模型、设备、dtype、耗时和错误信息。
- 边界：暂不支持多图、视频、音频、外部动作和专业领域承诺。

## Step 03：模型与运行策略

- 模型：`HuggingFaceTB/SmolVLM-256M-Instruct`。
- 原因：Day12 已验证、权重较轻、适合本地反复实验。
- CUDA：优先 `bfloat16`。
- CPU：回退 `float32`。
- 默认离线，不静默下载权重。

## Step 04：评测计划

原计划为截图 4 张、图表 3 张、普通照片 3 张，并为每张图片记录问题、期望要点和失败条件。

该 10 张图片测试集尚未整理，明确延期到真实 Demo 实现阶段。本次不将其标记为已经完成。

## Step 05：最小技术方案

```text
validate input
-> build task-aware prompt
-> processor
-> model.generate
-> decode new tokens
-> normalize result
-> record runtime metadata and JSONL evaluation
```

完整模块和失败处理记录在 `demos/vlm-visual-assistant/design.md`。

## Day14 收口

- 已完成：Week2 复盘、需求定义、模型选择和最小 MVP 技术方案。
- 已延期：真实 Demo 编码、10 张测试图片和完整评测。
- 下一步：Day15 扩散模型基本原理。
