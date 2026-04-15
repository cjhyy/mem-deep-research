# 技术 Roadmap

本文档只描述“当前阶段最值得投入的方向”，不再保留已经完成的历史计划清单。

## 当前阶段判断

项目已经进入“内核收敛期”。

当前更重要的不是继续横向扩功能，而是：

- 拆主链复杂度
- 提升公共入口可靠性
- 建立更可信的 benchmark / eval
- 清理兼容层与历史文档

## 已经比较稳的部分

- `AgentRuntime` 主路径已经成型
- context 管理链路完整
- sub-agent / todo / memory / transcript 已接入主循环
- tests 对子模块的覆盖比较扎实

## 当前最明显的瓶颈

### 1. 主循环过胖

`Orchestrator` 和 `MainLoopRunner` 承担了过多职责：

- mode policy
- builtin tool dispatch
- context lifecycle
- verify / summary policy
- checkpoint / resume

这是当前最大的结构性瓶颈。

### 2. 公共入口覆盖不足

相较于子模块测试，以下区域仍需要更强保护：

- `deep_research.py`
- `agent_factory.py`
- `pipeline.py`
- `orchestrator.py`
- `tool/manager.py`

### 3. Benchmark 仍偏 smoke

当前 `scripts/run_benchmark.py` 已经可以跑模式对比，但还不够当长期评测基准。

缺的不是“再加几个任务”，而是：

- 分层 benchmark
- 更稳定的任务集
- 质量指标

## 接下来建议的优先级

## P0：拆执行策略

建议目标：

```text
ModeResolver
  -> QuickRunner
  -> StandardRunner
  -> DeepRunner
```

至少先把下面几类逻辑从 `MainLoopRunner` 中拆出去：

- mode 解析与 mode profile
- builtin tool dispatch
- final summary policy
- deep verify policy

## P1：继续收口 runtime / lifecycle

建议继续推进：

- 清理剩余 global fallback
- 统一 message 写入入口
- 明确 tool result 从原始结果到 offload / restore 的统一生命周期
- 继续减少 legacy 文档和旧语义

## P2：测试前移到公共边界

优先补：

- `DeepResearch.run()` 端到端
- `from_project()` + hooks 加载
- `resume()` 完整性
- `AgentFactory.run_batch()`
- `ToolManager` transport 行为

## P3：把 benchmark 建成评估体系

建议拆三层：

### 离线回归集

只测框架行为，不依赖时效数据：

- route
- dedup
- compact / summarize / emergency
- offload / read_result / resume
- sub-agent / todo / transcript

### 在线稳定集

使用尽量稳定的任务和数据源，测：

- 任务完成率
- 事实正确率
- 引用覆盖率
- 成本 / 时延

### 新鲜度 smoke 集

保留实时搜索和最新信息题，只做能力 smoke，不纳入长期总分。

## 未来 4 到 6 周建议

1. 拆 `MainLoopRunner`
2. 补公共入口测试
3. 扩展 benchmark 指标与分层任务集
4. 继续清理兼容路径和历史文档

## 相关文档

- [13-execution-modes](./13-execution-modes.md)
- [16-dual-mode-execution-plan](./16-dual-mode-execution-plan.md)
- [17-repo-architecture-review](./17-repo-architecture-review.md)
