# 架构概览

Mem Deep Research 是一个可扩展的 AI Agent 框架，专注于深度研究任务。基于 MCP (Model Context Protocol) 工具协议，支持多 LLM 提供商。

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      DeepResearch API                       │
│              (deep_research.py / AgentFactory)              │
├─────────────────────────────────────────────────────────────┤
│                    Pipeline (pipeline.py)                    │
│        任务编排 · 组件初始化 · 错误处理 · 日志追踪          │
├─────────────────────────────────────────────────────────────┤
│                 Orchestrator (orchestrator.py)               │
│      Prompt 构建 · Skill 选择 · 语言检测 · 摘要生成         │
├─────────────────────────────────────────────────────────────┤
│                MainLoopRunner (main_loop.py)                 │
│   主循环 · 工具执行 · 上下文管理 · 监控 · 反思检查点        │
├──────────┬──────────┬──────────┬──────────┬─────────────────┤
│  LLM     │  Tool    │ Context  │ Monitor  │   Skill         │
│  Client  │ Executor │ Manager  │          │   System        │
│          │          │          │          │                 │
│ 多Provider│ MCP 协议 │ 三级压缩  │ 循环检测 │ rules/llm/     │
│ 流式输出  │ SecureCtx│ 去重     │ 超时保护 │ inline          │
├──────────┴──────────┴──────────┴──────────┴─────────────────┤
│                     Hook System (hooks.py)                   │
│          全生命周期钩子 · 项目级自定义 · 优先级链             │
├─────────────────────────────────────────────────────────────┤
│                   Configuration Layer                        │
│        Pydantic 验证 · OmegaConf/Hydra · YAML 配置          │
└─────────────────────────────────────────────────────────────┘
```

## 核心执行流程

```
DeepResearch.run(query)
  │
  ├─ 1. Pipeline.execute_task_pipeline()
  │     ├─ 创建 TaskTracer（结构化日志）
  │     ├─ 初始化 LLM Client（主 Agent + 子 Agent）
  │     ├─ 创建 Orchestrator
  │     └─ 调用 orchestrator.run_main_agent()
  │
  ├─ 2. Orchestrator.run_main_agent()
  │     ├─ 输入处理（hints 生成、任务指导）
  │     ├─ 获取工具定义（ToolManager）
  │     ├─ LLM Skill 选择（可选）
  │     ├─ 构建 System Prompt（工具 + Skill + SecureContext）
  │     └─ 创建 MainLoopRunner 并执行
  │
  ├─ 3. MainLoopRunner.run()  ← 核心循环
  │     while turn < max_turns:
  │       ├─ Hook: on_turn_start
  │       ├─ Monitor: pre_turn_check（超时/卡死检测）
  │       ├─ LLM 调用（via LLMCallHandler）
  │       ├─ Monitor: post_turn_check（循环检测）
  │       ├─ 升级处理（WARN → INJECT_HINT → TERMINATE）
  │       ├─ Inline Skill: 解析 <next_skills> 标签
  │       ├─ 工具调用去重（ContextManager.filter_duplicate_calls）
  │       ├─ 执行工具（ToolExecutor / SubAgentRunner）
  │       ├─ 注册结果（ContextManager.register_tool_results）
  │       ├─ 上下文管理（L1 Masking → L2 Summarize → L3 Reduction）
  │       ├─ Hook: on_turn_end
  │       └─ 反思检查点（按间隔注入）
  │
  └─ 4. 后处理
        ├─ 注入引用摘要（SourceRegistry）
        ├─ 生成最终摘要（SummaryHandler）
        └─ 返回 ResearchResult
```

## 模块依赖关系

```
DeepResearch
  └─ AgentFactory
       ├─ ConfigLoader (external_loader.py)
       │    ├─ 加载 YAML 配置
       │    ├─ 加载工具配置
       │    └─ 加载 Skill 定义
       ├─ Pipeline
       │    ├─ ToolManager (MCP 工具管理)
       │    ├─ LLMClient (Provider 工厂)
       │    └─ Orchestrator
       │         ├─ AgentPrompt (Prompt 生成)
       │         ├─ StreamHandler (SSE 事件流)
       │         └─ MainLoopRunner
       │              ├─ LLMCallHandler
       │              ├─ ToolExecutor
       │              ├─ SubAgentRunner
       │              ├─ ContextManager
       │              │    └─ WindowStrategyPipeline
       │              ├─ ExecutionMonitor
       │              ├─ InlineSkillSelector
       │              └─ TaskPlanner
       └─ HookRegistry (全局单例)
```

## 设计原则

| 原则 | 说明 |
|------|------|
| **全异步** | 基于 asyncio，LLM 调用、工具执行、子 Agent 均为异步 |
| **无状态框架** | 所有定制通过项目级 config + hooks.py 注入，框架本身不保存状态 |
| **渐进式降级** | 上下文管理三级压缩，监控三级升级，均为渐进式 |
| **依赖注入** | MainLoopRunner 等核心类通过构造函数接收所有依赖 |
| **可插拔策略** | WindowStrategy、LLM Provider、Skill 选择等均支持替换 |
| **MCP 优先** | 工具系统完全基于 MCP 协议，支持 stdio/HTTP/SSE 三种传输 |
| **配置驱动** | Pydantic 校验 + OmegaConf 运行时配置，YAML 声明式定义 |

## 后续演进

下一阶段技术演进规划见 [15-technical-roadmap.md](./15-technical-roadmap.md)。
