# 核心模块

## 模块总览

```
mem_deep_research_core/core/
├── orchestrator.py          # Agent 编排器
├── main_loop.py             # 主执行循环
├── pipeline.py              # 任务执行管道
├── agent_factory.py         # Agent 工厂
├── context_manager.py       # 上下文管理
├── window_strategy.py       # 窗口压缩策略
├── monitoring.py            # 执行监控
├── hooks.py                 # 钩子系统
├── secure_context.py        # 隐私数据保护
├── tool_executor.py         # 工具执行器
├── llm_call_handler.py      # LLM 调用处理
├── sub_agent_runner.py      # 子 Agent 运行器
├── stream_handler.py        # SSE 流式输出
├── task_planner.py          # 任务分解
├── message_interceptor.py   # 消息拦截
├── answer_handler.py        # 答案提取
├── user_context.py          # 用户上下文构建
├── constants.py             # 框架常量 + 工具函数
├── prompt_builder.py        # Prompt 构建（system prompt + skill 注入 + hint）
├── memory.py                # SessionMemory + LongTermMemory
├── todo_tracker.py          # TodoTracker 任务追踪
├── message_utils.py         # 消息工具函数
└── tool_result_formatter.py # 工具结果格式化
```

## Orchestrator — Agent 编排器

**文件**: `core/orchestrator.py`

Orchestrator 是单次任务执行的总协调者，负责：
- 输入预处理（hints 生成、语言检测）
- 工具定义获取和子 Agent 暴露
- System Prompt 构建（组合工具、Skill、SecureContext）
- 创建并运行 MainLoopRunner
- 后处理（最终摘要生成）

```python
class Orchestrator:
    async def run_main_agent(
        self,
        task_description: str,
        task_file_name: str = "",
        task_id: str = "",
        history: list = None,
    ) -> tuple[str, str]:
        """执行主 Agent 任务，返回 (final_answer, boxed_answer)"""
```

### 关键方法

| 方法 | 说明 |
|------|------|
| `_get_tool_definitions()` | 获取工具定义 + 子 Agent 工具 |
| `_build_system_prompt()` | 构建完整 System Prompt |
| `_select_skills()` | LLM Skill 选择（method=llm 时） |
| `_generate_hints()` | 生成任务提示（可选） |
| `_detect_language()` | 检测响应语言 |

## MainLoopRunner — 主执行循环

**文件**: `core/main_loop.py`

核心 turn-by-turn 执行循环，所有依赖通过构造函数注入：

```python
class MainLoopRunner:
    async def run(
        self,
        system_prompt: str,
        message_history: list,
        tool_definitions: list,
        max_turns: int,
        max_tool_calls_per_turn: int,
        keep_tool_result: int,
    ) -> tuple[str, str]:
        """执行主循环，返回 (final_answer, boxed_answer)"""
```

### 每轮执行流程

```
┌──────────────────────────────────────────────────┐
│ Turn N                                           │
├──────────────────────────────────────────────────┤
│ 1. Hook: on_turn_start                           │
│ 2. Monitor: pre_turn_check (超时/卡死)           │
│ 3. LLM 调用 → 获取响应文本 + 工具调用            │
│ 4. Monitor: post_turn_check (循环检测)           │
│ 5. 升级处理 (WARN/INJECT_HINT/TERMINATE)         │
│ 6. Inline Skill: 解析 <next_skills>              │
│ 7. 工具调用去重 (ContextManager)                  │
│ 8. 执行工具 (ToolExecutor / SubAgentRunner)       │
│ 9. 注册结果 (ContextManager)                      │
│ 10. 上下文管理 (L1→L2→L3)                        │
│ 11. Hook: on_turn_end                             │
│ 12. 反思检查点（按间隔注入）                       │
└──────────────────────────────────────────────────┘
```

## Pipeline — 任务管道

**文件**: `core/pipeline.py`

Pipeline 是任务执行的外壳，负责组件初始化和生命周期管理：

```python
async def execute_task_pipeline(
    cfg, task_name, task_id, task_description, task_file_name,
    main_tool_manager, sub_tool_managers, output_formatter,
    history=None, context=None, stream_queue=None, ground_truth=None,
) -> tuple[str, str, Path]:
    """返回 (final_answer, boxed_answer, log_path)"""
```

流程：创建 TaskTracer → 初始化 LLM Client → 创建 Orchestrator → 执行 → 清理

## AgentFactory — Agent 工厂

**文件**: `core/agent_factory.py`

提供多种创建方式：

```python
class AgentFactory:
    @classmethod
    def from_project_dir(cls, project_dir, config_name="agent") -> "AgentFactory"

    @classmethod
    def from_config_file(cls, config_path, logs_dir=None) -> "AgentFactory"

    @classmethod
    def from_config(cls, cfg, logs_dir=None) -> "AgentFactory"

    async def initialize(self) -> None
    async def run(self, task, task_id=None, context=None, stream_queue=None) -> TaskResult
    async def run_batch(self, tasks, parallel=False, max_concurrent=5) -> list[TaskResult]
```

便捷函数：

```python
from mem_deep_research_core.core.agent_factory import run_agent, run_agent_from_project

result = await run_agent("研究任务", config_path="config/agent.yaml")
result = await run_agent_from_project("研究任务", project_dir="./my_project")
```

## ToolExecutor — 工具执行器

**文件**: `core/tool_executor.py`

工具执行引擎，集成 Hook 和 SecureContext：

```python
class ToolExecutor:
    async def execute_tool_calls(
        self, tool_calls, max_tool_calls, agent_name="main",
    ) -> tuple[list, list, bool]:
        """返回 (tool_calls_data, tool_results_with_id, exceeded)"""
```

执行流程：
1. `on_tool_start` Hook（可修改参数）
2. SecureContext 占位符替换（`[SECURE:field]` → 真实值）
3. ToolManager 执行工具
4. 结果后处理（截断过长结果）
5. `on_tool_end` Hook（可修改结果）

## LLMCallHandler — LLM 调用处理

**文件**: `core/llm_call_handler.py`

统一 LLM 调用接口，处理日志、错误检测和上下文超限：

```python
class LLMCallHandler:
    async def handle_llm_call(
        self, system_prompt, message_history, tool_definitions,
        step_id="", purpose="", agent_type="main", stream_callback=None,
    ) -> tuple[str, bool, list]:
        """返回 (response_text, should_break, tool_calls_info)"""
```

**SummaryHandler**: 处理最终摘要生成，支持上下文超限重试（最多 10 次）：

```python
class SummaryHandler:
    async def handle_summary_with_retry(
        self, system_prompt, agent_prompt, message_history, ...
    ) -> tuple[str, str]:
        """返回 (final_answer, boxed_answer)"""
```

重试策略：
1. 替换旧工具结果为占位符
2. 二进制缩减（保留一半中间消息）
3. 兜底：提取最后一条有效响应

## SubAgentRunner — 子 Agent 运行器

**文件**: `core/sub_agent_runner.py`

管理子 Agent 的完整生命周期：

```python
class SubAgentRunner:
    async def run(
        self, sub_agent_name, task_description, keep_tool_result=-1,
    ) -> str:
        """执行子 Agent，返回最终答案"""
```

子 Agent 与主 Agent 共享 Hook 系统，但有独立的：
- 消息历史
- 工具定义
- Prompt 配置
- 上下文管理器

## StreamHandler — SSE 流式输出

**文件**: `core/stream_handler.py`

通过 asyncio.Queue 发送 SSE 协议事件：

| 事件类型 | 说明 |
|---------|------|
| `start_of_workflow` | 工作流开始 |
| `end_of_workflow` | 工作流结束 |
| `start_of_agent` | Agent 开始执行 |
| `end_of_agent` | Agent 结束执行 |
| `message` | 增量文本消息 |
| `tool_call` | 工具调用事件 |
| `reasoning` | 推理过程事件 |
| `usage_info` | 用量统计 |
| `show_error` | 错误展示 |

## Constants — 框架常量

**文件**: `core/constants.py`

所有硬编码值的单一来源（27+ 常量），包括：
- Token 阈值和比例
- 工具调用限制
- 超时设置
- 消息格式模板
- 推理标签定义

同时提供工具函数如 `generate_message_id()`、`reasoning_tags()` 等。

## PromptBuilder — Prompt 构建

**文件**: `core/prompt_builder.py`

从 Orchestrator 中提取的 Prompt 构建逻辑，负责：
- System Prompt 组装（基础模板 + 工具描述 + SecureContext）
- Skill 内容注入
- Hint 和 Guidance 追加
- 语言指令注入

```python
class PromptBuilder:
    def build_system_prompt(
        self, tool_definitions, skills, context, response_language,
    ) -> str:
        """构建完整的 system prompt"""
```

## Memory — 记忆系统

**文件**: `core/memory.py`

双层记忆架构：

| 类 | 作用域 | 持久化 |
|----|--------|--------|
| `SessionMemory` | 单次运行 | 否 |
| `LongTermMemory` | 跨 session | 是（文件系统） |

**SessionMemory** 自动追踪：
- 关键发现 (findings)
- 已使用的策略 (strategies)
- 失败的尝试 (failed_attempts)

**LongTermMemory** 跨 session 积累：
- 领域知识
- 有效策略模式
- 常见错误规避

## TodoTracker — 任务追踪

**文件**: `core/todo_tracker.py`

内置 `update_todo` 工具，LLM 可通过工具调用管理任务列表：

特性：
- 独立于 message_history，不受 context 压缩影响
- 每轮自动注入当前任务状态到 context
- 支持任务的创建、更新、完成标记

## MessageUtils — 消息工具函数

**文件**: `core/message_utils.py`

消息处理相关的工具函数集合，提供消息格式化、ID 生成、内容提取等通用功能。
