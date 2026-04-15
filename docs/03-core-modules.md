# 核心模块

本页只描述当前代码中的模块职责边界，不重复逐个方法签名。

## 生命周期与装配

### `deep_research.py`

对外主入口。

负责：

- `from_project()` / `from_config_dir()`
- `run()` / `resume()` / `run_batch()`
- 读取最终日志中的 perf metrics 和 checkpoints

### `core/agent_factory.py`

负责：

- 初始化 ToolManager 与 OutputFormatter
- 缓存工具定义
- 执行单任务和批量任务

### `core/pipeline.py`

负责一次任务运行的资源生命周期：

- 创建 `TaskTracer`
- 初始化主/子/路由 LLM client
- 创建 `Orchestrator`
- 在 finally 中关闭资源

### `core/agent_runtime.py`

负责实例级 runtime：

- hooks
- config loader
- 项目 hooks 加载
- 默认 hook 注册

## 编排与主循环

### `core/orchestrator.py`

组合根。

负责：

- 初始化 stream / interceptor / monitor / context manager / prompt builder
- 获取工具定义并注入内置工具
- 输入编译与 prompt 构建
- 创建并运行 `MainLoopRunner`
- final answer 后处理

### `core/main_loop.py`

执行内核。

负责：

- `effective_mode` 路由
- turn loop
- LLM 调用与循环检测
- built-in tool 与普通工具 dispatch
- sub-agent 协调
- context compaction / summarize / emergency reduction
- checkpoint / resume
- verify / final summary

## Prompt / Skill / 输入

### `core/prompt_builder.py`

负责：

- system prompt 构建
- hint 生成
- skill 选择与注入
- 静态 prompt section cache

### `core/input_compiler.py`

负责：

- URL 提取
- `@file` 展开
- `on_query_compile` hook

### `skills/*`

当前支持三类 skill 选择方式：

- `rules`
- `llm`
- `inline`

## Tooling

### `tool/manager.py`

MCP 工具总入口。

负责：

- stdio / sse / streamable-http / inprocess transport
- persistent session
- per-server call lock
- tool definition 缓存
- 工具调用纠错与重试

### `core/tool_executor.py`

负责：

- `on_tool_start` / `on_tool_end`
- SecureContext 占位符还原
- 工具调用流式事件
- 工具结果后处理

### `core/sub_agent_runner.py`

负责：

- 显式配置的子 Agent
- builtin `spawn_agent`
- 复用 `MainLoopRunner`，但隔离上下文

### `core/deferred_tools.py`

负责：

- 工具数过多时只暴露摘要
- builtin `tool_search`
- 按需恢复完整 schema

## Context / Reliability

### `core/context_manager.py`

负责：

- tool dedup
- source registry
- result offload / restore
- 驱动窗口压缩策略

### `core/window_strategy.py`

默认三层策略：

- Observation Masking
- LLM Summarize
- Binary Reduction

### `core/monitoring.py`

负责：

- timeout
- stall detection
- response loop detection
- escalation policy

## State / Observability

### `core/memory.py`

负责：

- `SessionMemory`
- `LongTermMemory`
- evidence ledger

### `core/todo_tracker.py`

负责：

- builtin `update_todo`
- 任务注入消息
- 独立于 message history 的 todo 状态

### `core/transcript.py`

负责结构化事件流，适合：

- replay
- debug
- benchmark 扩展

### `mem_deep_research_logging/task_tracer.py`

负责：

- step logs
- perf metrics
- turn checkpoints
- resume 所需最小状态

## 其他关键模块

### `core/llm_call_handler.py`

负责：

- 统一 LLM 调用
- guardrail hooks
- summary retry

### `core/message_interceptor.py`

负责：

- stream 中 reasoning / text / tool call 的拦截与清洗

### `core/answer_handler.py`

负责：

- final answer 后处理
- boxed answer / final summary 封装

## 模块关系

```text
DeepResearch
  -> AgentFactory
    -> Pipeline
      -> AgentRuntime
      -> Orchestrator
        -> PromptBuilder / ContextManager / Monitor / StreamHandler
        -> MainLoopRunner
          -> LLMCallHandler
          -> ToolExecutor
          -> SubAgentRunner
          -> TodoTracker / SessionMemory / Transcript
```
