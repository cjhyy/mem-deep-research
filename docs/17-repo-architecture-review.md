# 仓库架构评估与演进建议

> 状态基线：基于当前 `main` 分支代码结构整理，目标是描述“现在是什么”，而不是只描述理想设计。

## 一句话判断

当前仓库已经从“功能探索期”进入“框架内核收敛期”。

它不是简单的 demo agent，而是一个有完整执行内核的研究型 Agent Framework。当前真实架构更接近：

```text
单主循环执行内核
  + 多个横切子系统（runtime / hooks / context / monitoring / transcript / memory / todo / skills）
```

优点是扩展点丰富、长任务能力完整；缺点是复杂度高度集中在 `Orchestrator` 和 `MainLoopRunner`。

## 当前分层

### 1. API 与装配层

- `mem_deep_research_core/deep_research.py`
  对外主入口，提供 `from_project()`、`run()`、`resume()`、`run_batch()`、`validate()`。
- `mem_deep_research_core/core/agent_factory.py`
  负责初始化组件、缓存 ToolManager、调度单任务和批量任务。
- `mem_deep_research_core/core/pipeline.py`
  负责一次任务运行的资源生命周期：`TaskTracer`、LLM client、`Orchestrator`。

### 2. 运行时隔离层

- `mem_deep_research_core/core/agent_runtime.py`
  负责 hooks 和 config loader 的实例级隔离。
- 当前已经具备 runtime 主路径，但仍保留少量兼容 fallback。

### 3. 编排层

- `mem_deep_research_core/core/orchestrator.py`
  负责把 prompt、skills、monitor、context manager、todo、transcript、sub-agent runner 等组件装配起来。
- 它本质上是“组合根”，不直接承担每轮状态推进，但承担了大量初始化逻辑。

### 4. 执行内核层

- `mem_deep_research_core/core/main_loop.py`
  当前框架最核心的状态机。
- 这里串起了：
  - `effective_mode` 路由
  - LLM 调用
  - 内置工具与普通工具执行
  - 子 Agent
  - dedup
  - context compaction / summarize / emergency reduction
  - checkpoint / resume
  - verify / final summary

### 5. 横切子系统

- Prompt / Skill
  - `core/prompt_builder.py`
  - `skills/inline_selector.py`
  - `skills/llm_selector.py`
- Tooling
  - `tool/manager.py`
  - `core/tool_executor.py`
  - `core/deferred_tools.py`
- Context / Reliability
  - `core/context_manager.py`
  - `core/window_strategy.py`
  - `core/monitoring.py`
- State / Observability
  - `core/memory.py`
  - `core/todo_tracker.py`
  - `core/transcript.py`
  - `mem_deep_research_logging/task_tracer.py`

## 主执行链

```text
DeepResearch.run()
  -> AgentFactory.run()
    -> execute_task_pipeline()
      -> create LLM clients / TaskTracer
      -> Orchestrator.run_main_agent()
        -> InputCompiler.compile()
        -> get tool definitions
        -> DeferredToolManager.apply()
        -> PromptBuilder.select_skills()
        -> PromptBuilder.build_system_prompt()
        -> MainLoopRunner.run()
          -> LLMRouter resolves effective_mode
          -> turn loop
            -> monitor pre-check
            -> microcompact
            -> LLM call
            -> monitor post-check
            -> inline skill processing
            -> dedup + tool execution
            -> register results / evidence / strategy
            -> context manage + summarize if needed
            -> checkpoint
            -> reflection / verify when needed
        -> post_process_final_answer()
        -> save transcript / perf metrics / checkpoints
```

## 当前最有价值的设计

### Runtime 隔离已经成型

- `AgentRuntime` 把 hooks / config loader 从全局单例迁移到了实例级。
- 这是多实例安全和项目级 hooks 正确隔离的基础。

### 长任务链路是闭环的

不是单独做了某一个 feature，而是形成了完整链路：

- `SessionMemory`
- `TodoTracker`
- `ContextManager`
- result offload
- `read_result`
- checkpoint / resume
- transcript / perf metrics

这使框架不仅能“跑起来”，还能支撑长任务调试、恢复和评估。

### 工具层工程化程度较高

- `ToolManager` 已支持 stdio / SSE / streamable-http / inprocess
- 有 persistent session
- 有 per-server lock
- 有自动纠错与重连

这部分已经具备框架底座价值。

### 评估基础设施已具雏形

- `TaskTracer.perf_metrics`
- `TaskTracer.checkpoints`
- `Transcript`
- `scripts/run_benchmark.py`

虽然 benchmark 方案还不成熟，但底层观测数据已经不是空白。

## 当前主要瓶颈

### 1. Orchestrator 和 MainLoopRunner 偏胖

问题不在“代码行数多”，而在于职责持续累积：

- 新能力大多还是往主循环叠
- mode policy、builtin tool dispatch、context lifecycle、summary policy 仍集中在一个文件里

结果是：

- 认知负担高
- 改动容易互相影响
- benchmark 与 regression 难以精确定位

### 2. 文档和实现之间曾出现多处偏移

例如：

- `auto` 路由逻辑
- `DeepResearch` API 签名
- dual-mode 的完成状态
- runtime / hook 的作用边界

这次文档同步就是在收这部分偏移。

### 3. 测试很多，但主链覆盖不均衡

当前子模块测试已经很丰富，但覆盖热点和编排主链并不完全重合。

相对更薄弱的区域包括：

- `deep_research.py`
- `agent_factory.py`
- `pipeline.py`
- `orchestrator.py`
- `tool/manager.py`

这意味着“局部模块稳定”不等于“对外入口稳定”。

### 4. 仍有兼容路径和历史路径并存

主要体现在：

- global fallback
- legacy 文档/配置语义
- 一些历史注释和旧路线仍在 repo 中共存

这会持续增加维护负担。

## 建议的演进顺序

### 第一优先级：拆执行策略，而不是继续叠条件分支

建议目标：

```text
ModeResolver
  -> QuickRunner
  -> StandardRunner
  -> DeepRunner
```

不是为了“设计更漂亮”，而是为了把以下策略从主循环里拆出来：

- quick 的轻量 fast path
- standard 的常规多轮工具流
- deep 的研究型阶段化流程

### 第二优先级：继续收口 runtime 与 message lifecycle

建议继续做：

- 清理剩余全局 fallback
- 统一 message 写入入口
- 明确 tool result 从原始结果到 offload / compact / restore 的统一生命周期

### 第三优先级：把测试从子模块推到公共边界

优先补这几类测试：

- `DeepResearch.run()` 端到端
- `from_project()` hooks 加载与 runtime 隔离
- `resume()` 恢复完整性
- `AgentFactory.initialize/run_batch`
- `ToolManager` 真实 transport 行为

### 第四优先级：把 benchmark 从 smoke 提升为评估体系

当前脚本已经可用，但更像 mode smoke test，还不是稳定 benchmark。

## Benchmark 建议

建议拆成三层。

### L1：离线回归集

不依赖外网和实时数据，只验证框架行为：

- route 是否符合预期
- quick / standard / deep 行为差异
- dedup / compact / summarize / emergency
- offload / read_result / resume
- sub-agent / todo / transcript

适合作为 CI 回归集。

### L2：在线稳定集

依赖真实模型和工具，但尽量使用冻结网页、固定语料或可复现数据源。

重点看：

- 任务完成率
- 事实正确率
- 引用覆盖率
- 证据与结论映射质量
- 成本 / 时延

### L3：新鲜度 smoke 集

保留“latest / most recent”类任务，只做能力 smoke，不纳入长期对比分数。

这类任务适合检测：

- 搜索是否仍可用
- provider / tool 生态是否正常
- 时效性回答是否退化

### 建议补充的指标

除了当前脚本已有的 `duration / turns / tool_calls / tokens / answer_length`，建议补：

- `effective_mode` 来源（hook / structural / router / default）
- LLM time vs tool time
- context compaction 次数
- loop escalation 次数
- verify 是否触发
- evidence / source 数量
- resume 成功率

这些数据大多可以直接从 `perf_metrics` 与 `transcript` 扩展得到。

## 当前建议

如果只做下一轮，我建议按这个顺序推进：

1. 拆 `MainLoopRunner`，先把 mode strategy 和 builtin tool dispatch 拆出去
2. 补 `DeepResearch` / `AgentFactory` / `Pipeline` 级别测试
3. 把 benchmark 拆成离线回归集和在线评测集
4. 继续清理历史兼容文档与旧语义

## 相关文档

- [00-architecture](./00-architecture.md)
- [13-execution-modes](./13-execution-modes.md)
- [14-api-reference](./14-api-reference.md)
- [15-technical-roadmap](./15-technical-roadmap.md)
- [16-dual-mode-execution-plan](./16-dual-mode-execution-plan.md)
