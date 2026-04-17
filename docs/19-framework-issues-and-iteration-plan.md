# 当前框架问题与后续迭代建议

> 状态基线：基于 2026-04-16 仓库现状整理，目标是回答“当前框架最需要解决什么，以及应该按什么顺序迭代”，而不是重复已有的架构介绍或远期愿景。

## 一句话判断

项目已经具备研究型 Agent Runtime 的核心骨架，但现在最主要的问题不是功能缺失，而是运行时 contract 还没有完全收口。

当前阶段最值得投入的，不是继续横向叠 feature，而是先把下面四件事做扎实：

- 多任务 / 多上下文运行时隔离
- 主 Agent 与子 Agent 的结果生命周期一致性
- 配置、入口 API、发布元数据的单一契约
- 主链复杂度与公共边界测试

## 当前优势

从代码结构上看，这个项目已经明显超过“prompt + tools wrapper”阶段，几个核心能力是成型的：

- `MainLoopRunner`、`ContextManager`、`ExecutionMonitor`、`SummaryHandler` 已经构成可运行的长任务内核
- `AgentRuntime`、`HookRegistry`、`ConfigLoader` 提供了实例级扩展点
- `ToolManager` 已支持 `stdio`、`sse`、`streamable-http`、`inprocess` 四类 MCP transport
- `SessionMemory`、`TodoTracker`、`Transcript`、`TaskTracer` 让长任务的恢复和观测具备框架价值
- `PromptBuilder`、`Skill`、`DeferredToolManager` 说明框架已经开始考虑成本、可扩展性和 prompt cache 命中

问题在于，这些能力目前还更像“并列存在”，而不是完全统一到一个稳定 contract 下。

## 当前最需要解决的问题

## P0：运行时隔离与 contract 仍然存在真实风险

### 1. MCP 会话复用和任务上下文隔离不完全一致

`ToolManager` 会缓存 server 级持久会话，而 stdio transport 的上下文注入发生在 `_get_or_create_session()` 首次创建 session 时。  
这意味着同一个 `ToolManager` 被多个任务复用时，后续任务虽然会传入新的 `context`，但子进程环境变量可能仍然沿用首次创建 session 时的旧值。

这个问题在以下场景里都可能出现：

- 同一个 `DeepResearch` 实例连续运行多个任务
- `AgentFactory.run_batch(parallel=True)` 并发跑多个不同 context 的任务
- 依赖环境变量读取 `USER_ID`、`ORG_ID`、`TIMEZONE` 的 MCP server

这类问题的危险不是“报错”，而是静默串号，属于框架级高优先级风险。

### 2. 子 Agent / spawn Agent 的结果生命周期和主链不完全一致

主 Agent 的结果会经过：

```text
tool result
  -> format
  -> backup_large_result
  -> offload marker / read_result
  -> context compaction
  -> resume rebuild
```

但 `SubAgentRunner.run()` 和 `SubAgentRunner.spawn()` 内部各自创建了新的 `ContextManager()`，并没有完整继承主链的配置和语义。

当前主要问题包括：

- 子 Agent 的 `ContextManager` 不是从主配置构建，很多行为退回默认值
- offload 虽然可能发生，但子 Agent 自己未必拥有与主链一致的 `read_result` 能力
- spawned agent 复用了父级 `ToolExecutor`，却没有完全复用父级结果生命周期管理

结果是：主 Agent 和子 Agent 看似共用一套能力，实际 contract 并不完全对齐。

### 3. 配置契约已经分裂，而且启动阶段不是 fail-fast

仓库里现在至少存在两套配置预期：

- schema / example / provider client 倾向于把 key 放在 `main_agent.llm.*`
- `PromptBuilder.generate_hints()`、`ConfigLoader.get_llm_skill_selector()` 又直接读取 `main_agent.openai_api_key`

这会导致一些功能不是显式失败，而是静默失效：

- hint generation
- LLM skill selector
- 某些 OpenAI 相关辅助路径

更关键的是，`DeepResearch._validate_config()` 当前只记录日志，不阻断初始化。  
对于“核心字段路径错了”的情况，这会让框架继续运行，但行为偏离配置作者的预期。

## P1：边界与正确性问题还需要继续收口

### 4. 输入编译链的 `@file` 能直接读取本机文件

`InputCompiler` 会在进入工具层之前直接解析 `@file` 并读取文件内容，然后把文件内容拼进 prompt。

这带来两个问题：

- 绕过了工具层的边界控制与审计
- 如果任务输入不完全可信，可能读取项目外的敏感文件，如 `.env`、SSH 配置、本地凭证文件

如果框架的定位只是本地单用户研究工具，这个问题的优先级可以下调；但如果希望做成更通用的 runtime，这里需要 project-root allowlist、显式配置开关，或者更严格的 attachment policy。

### 5. 最终直出答案和内部清洗逻辑之间还存在状态漂移

`MainLoopRunner` 里已经有对 `<response_language>` 和 `<evidence>` 等内部标签的清洗逻辑，但 `last_assistant_text` 与 `assistant_response_text` 的同步并不总是严格一致。

这会带来一个边缘但真实的问题：

- message history 里的 assistant 文本已经被清理
- summary 又因为 `is_simple_response` 被跳过
- 最终返回给用户的仍可能是清洗前的旧文本

这类问题通常只会在少量路径触发，但一旦触发，会直接影响用户看到的最终答案质量。

### 6. `tool_definitions` 数据形态混用，正在持续增加主链复杂度

当前主链里同时存在两类工具定义：

- MCP server 风格：`{"name": server, "tools": [...]}`
- builtin tool 单体 dict：直接 append 到 `tool_definitions`

因此在 `Orchestrator`、`MainLoopRunner`、`PromptBuilder`、native/xml tool 处理逻辑里，都存在针对两种 shape 的特殊分支。

这不是“立刻会炸”的问题，但会持续提高：

- 新功能接入成本
- 类型不确定性
- 回归测试复杂度

后面如果再继续增加 builtin tool、deferred tool、sub-agent tool 变体，这个问题会越来越明显。

## P2：工程质量和对外边界还不够稳

### 7. 对外入口的测试覆盖仍然偏薄

当前测试数量不少，但分布更偏子模块行为。  
相对更薄弱的，是对外真正承诺给用户的公共边界：

- `DeepResearch.run()`
- `DeepResearch.from_project()`
- `DeepResearch.resume()`
- `AgentFactory.initialize()` / `run_batch()`
- `ToolManager` 在不同 transport 下的行为差异

这意味着“局部模块都测了”并不等价于“用户入口足够稳定”。

### 8. 版本与发布元数据已经出现漂移

当前仓库里至少存在以下不一致：

- `pyproject.toml` 为 `1.2.1`
- `mem_deep_research/__init__.py` 为 `1.2.1`
- `mem_deep_research_core/__init__.py` 为 `1.2.0`
- `docs/15-technical-roadmap.md` 又以 `v1.2.2` 作为基线

这类问题不影响单次运行，但会影响：

- 发布可信度
- 问题定位
- CHANGELOG 对齐
- 用户对“当前版本到底是什么”的判断

对于框架项目，这属于必须尽快清理的工程卫生问题。

## 建议的迭代顺序

## 第一阶段：运行时修复版

目标：先修“会影响正确性和隔离性”的问题，不新增大 feature。

### 建议优先完成

- 改造 `ToolManager` 的 context 注入策略
- 明确 stdio session 是“按任务隔离”还是“按 server 复用但可热更新 context”
- 把 `SubAgentRunner` 的 `ContextManager` 初始化统一到主配置
- 对齐主 Agent / 子 Agent / spawned agent 的 offload 与 `read_result` contract
- 统一配置字段读取路径，至少先把 `openai_api_key` 的读取入口收口
- 让配置校验对核心字段错误 fail-fast，而不是只打日志
- 修复最终答案路径里 `last_assistant_text` 和清洗逻辑不同步的问题

### 完成标准

- 不同任务的 context 不会在 MCP server 之间串号
- 子 Agent 的结果生命周期与主链保持一致
- 配置写错核心字段时，初始化明确失败
- 最终答案不再泄漏内部标签或旧状态

## 第二阶段：contract 收敛版

目标：把几个已经存在但尚未统一的 runtime 语义收口成一个明确模型。

### 建议重点

- 统一 `tool_definitions` 的单一数据结构
- 显式定义结果生命周期：

```text
raw result
  -> formatted result
  -> backup
  -> offloaded marker
  -> restore / read_result
  -> resume rebuild
```

- 把 `resume` 从“message_history + 少量附加字段”升级为更明确的 runtime snapshot
- 明确 `TOOL_RESULT`、`OFFLOADED`、`SESSION_MEMORY`、`SUMMARY_PROMPT` 等消息类型的写入规则

### 完成标准

- `read_result`、offload、restore、resume 对同一份结果使用统一语义
- 主链不再到处判断工具定义 shape
- checkpoint 恢复的语义可以被清楚描述和测试验证

## 第三阶段：主链瘦身版

目标：把主循环从“能工作的大状态机”收敛成“边界清晰的执行内核”。

### 优先拆出的模块

- `ModeResolver`
- `BuiltinToolDispatcher`
- `ResultLifecycleManager`
- `SummaryPolicy`

建议不是为了机械拆文件，而是为了减少一个地方同时承担这些职责：

- route 决策
- builtin tool 分派
- 结果生命周期
- verify / summary 策略
- checkpoint / resume 细节

### 完成标准

- `MainLoopRunner` 只保留回合推进和状态编排
- route、builtin tools、summary policy 不再和 context lifecycle 强耦合
- 新功能接入时不再优先修改主循环本体

## 第四阶段：公共边界与评估体系版

目标：让框架从“内部设计不错”进入“可持续对外使用”的阶段。

### 建议补齐

- 公共入口端到端测试
- transport 差异测试
- 离线回归 benchmark
- 版本单一来源
- 发布前检查脚本

### 最值得补的测试

- `DeepResearch.run()` 基本成功路径
- `from_project()` + hooks/runtime 隔离
- `resume()` 恢复完整性
- `AgentFactory.run_batch()` 在不同 context 下的行为
- `ToolManager` 在 `stdio` / `sse` / `streamable-http` 下的上下文一致性
- 子 Agent / spawn + offload + `read_result` 回归测试

## 一个更实际的版本顺序

如果按未来两个版本来排，我会建议这样落：

### 下一版：稳定性与 contract 修复

- 先不加新模式和新能力
- 只修运行时隔离、配置契约、子 Agent 生命周期、最终答案清洗、版本元数据
- 补与这些问题直接对应的回归测试

### 下下版：主链收敛与评估体系

- 拆 `MainLoopRunner` 的执行策略
- 统一结果生命周期模型
- 把 benchmark 和 release 工程补齐

## 当前阶段不建议优先做的事

在上述问题没有收口之前，不建议优先投入以下方向：

- 再扩更多 prompt preset
- 再引入新的执行模式分支
- 再增加更多 builtin tool 变体
- 大规模扩张 skill 选择逻辑

这些方向都可能有价值，但现在继续加，会让主链复杂度和 contract 分裂继续扩大。

## 总结

这个项目现在最值得珍惜的，是它已经拥有一个真正有潜力的研究型执行内核。

下一步的关键，不是证明“还能继续加什么功能”，而是把现有能力收敛成一个更可信、更可恢复、更可测试、更可发布的 runtime。  
只要把运行时隔离、结果生命周期、配置契约、主链复杂度这几件事处理好，后面的功能扩展成本会明显下降，框架也会更像一个稳定底座，而不是一组强功能模块的集合。
