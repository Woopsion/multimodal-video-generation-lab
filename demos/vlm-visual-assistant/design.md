# VLM Visual Assistant - Design Draft

> 状态：Day14 Step 02 草稿。已确定通用多模态 Demo 的第一版边界，模型选择留到 Step 03。

## 1. 目标用户与核心问题

目标用户是学习者本人，目标是用一个最小但真实的 Demo 验证 VLM 的输入、推理、输出和失败边界。它不是生产级服务，也不承诺专业领域准确性。

第一版任务范围：

- 软件界面截图理解与操作状态说明。
- 柱状图、折线图等基础图表问答。
- 普通照片中的主体、动作、位置和场景描述。

## 2. 输入契约

第一版固定为单图 + 文本问题：图片模态提供视觉上下文，文本模态提供任务意图。

```text
image path + question text + optional task mode
```

第一版约束：

- 支持的图片格式与大小限制。
- 第一版不支持多图对比、视频和音频，避免扩大问题边界。
- 问题为空时暂不自动猜测任务，要求调用方提供文本问题。
- `task_mode` 默认使用 `auto`，评测时可显式指定 `screenshot`、`chart` 或 `photo`。

## 3. 输出契约

```text
answer text + runtime metadata + optional failure note
```

固定输出字段：

- 返回模型名称、耗时和设备信息，便于学习和定位性能边界。
- 视觉信息不足时明确说明无法确认，不用编造确定答案。
- 保存结构化 JSONL 评测记录，后续用于 Step 04 的测试集评估。

## 4. 模型选择

当前基线候选：`HuggingFaceTB/SmolVLM-256M-Instruct`。

选择模型时需要同时考虑：

- 任务能力：OCR、图表、普通场景。
- 本地显存、内存和推理速度。
- 下载体积与离线运行要求。
- Transformers 兼容性与代码维护成本。
- 已知失败边界是否可以被产品策略接受。

## 5. 系统数据流

```text
image + question
-> input validation
-> processor
-> vision encoder
-> projector
-> LLM generation
-> decode
-> answer normalization
-> result + runtime metadata
```

## 6. 测试集计划

计划整理 10 张图片：

| 类型 | 建议数量 | 重点能力 |
| --- | ---: | --- |
| 软件截图 | 4 | OCR、控件、状态、布局关系 |
| 图表 | 3 | 极值、趋势、比较、完整回答 |
| 普通照片 | 3 | 主体、动作、位置、场景关系 |

每张图片至少准备：问题、期望要点、允许表达、失败条件。

## 7. Step 02 决策结果

当前第一版契约：`image path + question text + optional task mode -> answer text + runtime metadata + optional failure note`。

## 8. Day14 后续决策

- Step 02：已完成通用 Demo 的用户、任务和输入输出边界。
- Step 03：确定模型、设备和降级策略。
- Step 04：定义 10 张测试图片与评分规则。
- Step 05：收敛为可以直接进入实现的技术方案。
