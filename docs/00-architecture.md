# 架构概览

Mem Deep Research 当前的真实架构不是 DAG/Graph 风格的编排器，而是：

```text
单主循环执行内核
  + 多个横切子系统
```

其中主循环负责推进状态，横切子系统负责 prompt、工具、context、monitoring、memory、todo、transcript 等能力。

## 当前分层

```text
DeepResearch API
  -> AgentFactory
    -> Pipeline
      -> AgentRuntime
      -> Orchestrator
        -> MainLoopRunner
          -> LLM / Tools / Context / Monitor / Memory / Todo / Transcript
```

### API 层

- `mem_deep_research_core/deep_research.py`
- 提供 `from_project()`、`run()`、`resume()`、`run_batch()`、`validate()`

### 任务装配层

- `core/agent_factory.py`
- `core/pipeline.py`

职责：

- 初始化 ToolManager / OutputFormatter / LLM client
- 创建 `TaskTracer`
- 控制单任务与批量任务生命周期

### 运行时层

- `core/agent_runtime.py`

职责：

- 为每个 `DeepResearch` 实例提供独立的 hooks / config loader
- 隔离项目级 hooks，避免多实例互相污染

### 编排层

- `core/orchestrator.py`

职责：

- 输入编译
- 获取工具定义
- deferred tools
- skill 选择
- 构建 system prompt
- 装配 `MainLoopContext`
- 调用 `MainLoopRunner`

### 执行内核层

- `core/main_loop.py`

职责：

- 路由 `effective_mode`
- turn loop
- LLM 调用
- tool dispatch
- sub-agent
- context lifecycle
- checkpoint / resume
- verify / final summary

## 主执行链

```text
DeepResearch.run(query)
  -> AgentFactory.run()
    -> execute_task_pipeline()
      -> create LLM clients + TaskTracer
      -> Orchestrator.run_main_agent()
        -> InputCompiler.compile()
        -> get tool definitions
        -> DeferredToolManager.apply()
        -> PromptBuilder.select_skills()
        -> PromptBuilder.build_system_prompt()
        -> MainLoopRunner.run()
          -> resolve effective_mode with LLMRouter
          -> turn loop
            -> monitor pre-check
            -> microcompact
            -> LLM call
            -> monitor post-check
            -> inline skill handling
            -> dedup + tool execution
            -> context manage / summarize / emergency reduction
            -> checkpoint / reflection / verify
        -> post_process_final_answer()
        -> save transcript / perf metrics
```

## 横切子系统

### Prompt 与 Skill

- `core/prompt_builder.py`
- `skills/matcher.py`
- `skills/inline_selector.py`
- `skills/llm_selector.py`

### Tooling

- `tool/manager.py`
- `core/tool_executor.py`
- `core/deferred_tools.py`

### Context 与可靠性

- `core/context_manager.py`
- `core/window_strategy.py`
- `core/monitoring.py`

### 状态与观测

- `core/memory.py`
- `core/todo_tracker.py`
- `core/transcript.py`
- `mem_deep_research_logging/task_tracer.py`

## 当前架构特点

- 运行时隔离已经成型：`AgentRuntime` 是实例级主路径
- 长任务链路完整：`SessionMemory`、`TodoTracker`、offload、`read_result`、resume、transcript 已形成闭环
- mode 已经不是纯配置开关：`auto` 通过 `LLMRouter` 解析成 `effective_mode`
- 主循环很强，但也偏胖：大量新能力仍在往 `MainLoopRunner` 聚集

## 推荐阅读顺序

- [03-core-modules](./03-core-modules.md)
- [13-execution-modes](./13-execution-modes.md)
- [14-api-reference](./14-api-reference.md)
- [15-technical-roadmap](./15-technical-roadmap.md)
- [17-repo-architecture-review](./17-repo-architecture-review.md)
