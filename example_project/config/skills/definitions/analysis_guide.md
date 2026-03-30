---
name: analysis_guide
version: "1.0.0"
type: knowledge
description: "数据分析与推理指南 — 指导 Agent 进行结构化分析"
when_to_use: >
  当任务涉及数据分析、对比、推理、评估时使用。
  帮助 Agent 采用结构化方法处理复杂分析任务。

triggers:
  keywords:
    - "分析"
    - "对比"
    - "评估"
    - "analyze"
    - "compare"
    - "evaluate"
  intents:
    - "data_analysis"
    - "comparison"

metadata:
  author: "example_project"
  created_at: "2026-03-24"
  tags: ["analysis", "reasoning", "methodology"]
  priority: 8
---

# 结构化分析指南

当进行分析或推理任务时，遵循以下方法论：

## 1. 明确分析目标

- 确定要回答的核心问题
- 识别关键维度和指标
- 界定分析范围

## 2. 信息收集

- 使用搜索工具收集相关数据
- 多来源交叉验证
- 注意数据的时效性和可靠性

## 3. 结构化对比

采用表格或矩阵形式展示对比：

| 维度 | 选项A | 选项B | 说明 |
|------|-------|-------|------|
| 维度1 | ... | ... | ... |
| 维度2 | ... | ... | ... |

## 4. 推理与结论

- 基于证据进行推理，避免无依据的假设
- 明确区分事实和推测
- 给出置信度评估
- 指出信息缺口和不确定性

## 5. 建议输出格式

- 先给结论，再展开论证
- 使用层级结构组织信息
- 关键数据用加粗标注
