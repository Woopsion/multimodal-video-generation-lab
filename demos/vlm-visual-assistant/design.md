# VLM Visual Assistant - Minimum MVP Design

> 状态：Day14 最小 MVP 设计已收口。真实 Demo 编码和 10 张图片评测留到后续实现阶段。

## 1. 目标与边界

目标用户是学习者本人。这个 Demo 用于验证 VLM 的输入、推理、输出和失败边界，不是生产级服务，也不承诺专业领域准确性。

第一版支持：

- 单张图片 + 文本问题。
- 软件截图、基础图表、普通照片。
- 自然语言回答与运行元数据。
- 本地单用户运行。

第一版不支持：

- 多图对比、视频和音频。
- 自动操作界面或执行外部动作。
- 医疗、遥感、工业质检等专业可靠性承诺。
- 高并发、权限、计费和 SLA。

## 2. 输入契约

```text
image path + question text + optional task mode
```

- 图片路径与问题文本必填。
- `task_mode` 默认是 `auto`，评测时可使用 `screenshot`、`chart` 或 `photo`。
- 问题为空时不自动猜测任务。
- 图片无效或格式不支持时，在模型推理前返回错误。

## 3. 输出契约

```text
answer text + runtime metadata + optional failure note
```

输出至少包含：

- `answer`：自然语言回答。
- `model`：实际使用的模型。
- `device` / `dtype`：运行环境。
- `latency_ms`：推理耗时。
- `error`：输入、加载或推理错误。

视觉信息不足时明确说明无法确认，不编造确定答案。评测结果使用 JSONL 保存。

## 4. 模型选择

第一版复用 Day12 已验证的 `HuggingFaceTB/SmolVLM-256M-Instruct`。

选择理由：

- 权重较轻，适合本地学习和重复实验。
- 已跑通 Transformers 推理链路，减少无关环境问题。
- 支持图片与文本联合输入，满足三类基础视觉问答。
- 已知 OCR、图表完整回答和细粒度场景理解存在边界，适合用于学习评测。

运行策略：

- CUDA 可用时优先使用 `bfloat16`。
- CUDA 不可用时回退 CPU 与 `float32`。
- 默认不自动下载权重，需要显式允许下载。
- 第一版不引入量化、LoRA、服务化并发和模型路由。

只有当该模型无法满足学习目标时，才比较更大的 VLM。

## 5. 系统数据流

```text
image + question
-> input validation
-> task-aware prompt builder
-> processor
-> vision encoder
-> projector
-> LLM generation
-> decode new tokens
-> answer normalization
-> result + runtime metadata
-> JSONL evaluation record
```

## 6. 最小 MVP 模块

| 模块 | 职责 |
| --- | --- |
| Input validation | 检查图片、问题、任务模式和生成参数 |
| Prompt builder | 根据任务模式组织图片占位符与用户问题 |
| Model runner | 加载 processor / model，选择 device / dtype 并执行生成 |
| Result formatter | 返回回答、模型、设备、耗时和错误信息 |
| Evaluation logger | 将任务、期望要点和实际结果写入 JSONL |

## 7. 失败处理

- 图片不存在或格式错误：生成前返回输入错误。
- 模型权重不存在：说明离线状态，并提示显式允许下载。
- CUDA 不可用或显存不足：回退 CPU，或返回可诊断错误。
- 图片信息不足：回答中明确表达无法确认。
- 模型回答遗漏：保留原始结果，在评测记录中标记缺失要点。

## 8. 测试计划

原计划整理 10 张图片：截图 4 张、图表 3 张、普通照片 3 张。每张图片需要问题、期望要点和失败条件。

该测试集尚未实际整理。本项延期到真实 Demo 实现阶段，不作为 Day14 设计学习的完成条件。

## 9. 收口结论

- 已完成：Week2 知识复盘、需求边界、模型选择、运行策略和技术方案。
- 已延期：10 张测试图片整理、真实 Demo 编码和完整评测。
- 下一学习阶段：Day15 扩散模型基本原理。
