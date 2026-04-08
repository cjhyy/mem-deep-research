# 技术 Roadmap

本文档基于当前代码库状态更新，不再只描述理想方向，而是明确：

- 哪些基础能力已经落地
- 哪些改造正在进行中
- 下一阶段最该投入的工作是什么

当前判断：项目已经从“功能探索期”进入“内核收敛期”。`AgentRuntime`、结构化消息类型、checkpoint/resume、token budget、transcript 等基础能力已经开始成型，但仍处在“新路径与旧兼容路径并存”的阶段。接下来最重要的不是继续扩面，而是完成内核收口。

## 当前状态评估

### 已经落地的基础

- 已引入 `AgentRuntime`，开始把 hooks / config loader 从全局单例迁移到实例级 runtime
- 已引入结构化消息类型 `MT.*` 和 `_type` 字段
- 已有 `make_tool_result_msg()` / `make_tool_result_msg_native()`，开始统一 tool result 写入入口
- `microcompact` 和 `ObservationMasking` 已从“黑名单排除”切换到“白名单只处理 `MT.TOOL_RESULT`”
- `LongTermMemory` 的首次写入死锁问题已通过 `RLock` 修复
- 已有 turn-level checkpoint、`resume()`、`get_resumable_state()` 基础链路
- 已有 token budget warning / terminate 基础能力
- 已有 transcript、task tracer、citation summary 基础设施

### 正在进行中的改造

- hooks 注入链路已经扩展到多个模块，但仍保留 global fallback
- message typing 已进入主循环、provider 和上下文管理，但 user/tool role 与 `_type` 语义还没有完全对齐
- context compact / microcompact / offload / resume 已具备基础能力，但恢复链路和压缩链路还没有完全收口
- 新的测试文件已经开始覆盖 runtime isolation、hook injection、message types、session memory

### 当前仍然明显的风险

- 新 runtime 路径与旧全局单例路径并存，长期会增加维护成本
- `keep_tool_result` 仍有一层“按最后 N 条消息截断历史”的逻辑，和“只保留最近 N 个工具结果”的配置语义不完全一致
- resume 当前只恢复了部分 checkpoint 状态，`todo_state` 与 offloaded content 还没有完整接回
- GPT 原生 tool call 路径使用 `role="tool"`，但压缩链大多仍按 `role="user"` 遍历，导致部分真实 tool result 不会进入 compact / microcompact
- `on_query_compile` 虽然在 InputCompiler 中被调用，但 hook 注册表与 runtime 传递链还没有完全打通
- 配置校验目前以“记录 warning”为主，fail-fast 还不够强
- `except Exception` 仍然较多，关键模块的错误边界不够明确
- 端到端可靠性验证仍偏少，尤其是 resume / offload / sub-agent 组合场景

## 总体原则

技术演进分为四个阶段：

1. 先收内核：完成 runtime、message model、context lifecycle 收口
2. 再补可靠性：把测试、校验、异常边界和恢复能力补扎实
3. 后做生产化：在已有 checkpoint、budget、streaming 基础上补全生产能力
4. 最后做差异化与生态：放大 citation、transcript、secure context、tool ecosystem 的优势

## Phase 1：完成内核收口

目标：把已经开始的架构改造做完整，避免新旧路径长期并存。

### 1. Runtime 全量收口

现状：

- `AgentRuntime` 已落地
- `DeepResearch`、`AgentFactory`、`Orchestrator` 已接入 runtime
- 多个模块支持注入 hooks

下一步：

- 清理仍然依赖全局 hooks 的剩余路径
- 明确哪些模块允许 fallback，哪些模块必须显式使用 runtime
- 将默认 hook 注册统一放到 runtime 初始化阶段，减少模块 import 时副作用
- 为多实例并行运行补充更强的集成测试

### 2. Message Model 收口

现状：

- `MT.*`、`PROTECTED_MESSAGE_TYPES`、`make_msg()` 已存在
- 主循环里多个系统注入消息已带 `_type`
- provider 已开始通过统一 helper 写入 tool result

下一步：

- 统一所有 message_history 写入入口，优先使用 `make_msg()`
- 补齐 tool result、offloaded content、system injection 的结构化 metadata
- 统一 `role="user"` / `role="tool"` 两条 tool result 路径的压缩、恢复和裁剪语义
- 让 `compact`、`masking`、`microcompact`、`resume` 优先依赖 `_type`，仅在 legacy 场景使用兼容 fallback
- 清理 `keep_tool_result` 中仍然按“最后 N 条消息”裁剪历史的旧逻辑

### 3. Context Lifecycle 收口

现状：

- 已有 `microcompact`、`offload_large_result()`、`restore_offloaded_content()`、resume 机制

下一步：

- 定义 tool result 从“原始结果 → offload → compact → resume restore”的统一生命周期
- 明确哪些消息可压缩、可卸载、可恢复、可永久保护
- 在 resume 时恢复 `todo_state`、offloaded content、session memory，而不是只恢复 message history 文本
- 补充 offload + resume + compact 组合测试
- 统一 transcript 中对这些阶段的事件记录

### 4. Runtime Hook 收口

现状：

- 大部分核心模块已经支持注入 runtime-scoped hooks
- `InputCompiler` 仍有一条绕过 runtime 的路径

下一步：

- 让 `InputCompiler` 使用实例级 hooks，而不是模块级全局 hooks
- 将 `on_query_compile` 纳入正式支持的 hook 列表，并补对应测试
- 统一项目 hooks 加载和 runtime hooks 传递方式，减少“临时替换全局 hooks”的兼容代码

### Phase 1 交付结果

- runtime 成为唯一主路径
- message typing 成为默认约束，而不是可选增强
- context lifecycle 行为稳定、可预测

## Phase 2：补齐可靠性与验证

目标：把框架从“架构方向正确”提升到“可放心迭代”。

### 1. 核心链路测试补强

优先补齐以下场景：

- MainLoopRunner → Orchestrator → Pipeline 端到端
- runtime isolation / hook injection
- compact + offload + resume
- checkpoint + resume continuation
- todo_state 恢复
- GPT native tool call → tool result write → compact / microcompact
- sub-agent spawn + return + failure
- provider 请求构造、响应解析、错误映射

### 2. 配置与错误边界治理

- 将关键配置问题从 warning 提升为 fail-fast
- 建立“schema 合法”和“运行时可执行”两层校验
- 缩小关键模块中的 `except Exception` 范围
- 为常见错误建立更稳定的异常分类与错误消息

### 3. Session / Memory 正确性

- 完善 SessionMemory 并发与快照一致性测试
- 校验 LongTermMemory 的恢复、更新、截断和异常路径
- 明确记忆注入与 message typing / compact 策略之间的契约

### Phase 2 交付结果

- 核心路径具备回归保护
- 配置错误能更早暴露
- 恢复链路和记忆链路更可信

## Phase 3：生产化能力补全

目标：在现有基础设施上补齐真正的生产运行能力。

### 1. Checkpoint / Resume 升级

现状：

- 已有 checkpoint 和 resume 基础实现

下一步：

- 补全 resume 后的 tool/session/memory 恢复一致性
- 增加更细粒度 checkpoint 策略
- 支持 resume 后的状态校验和告警

### 2. Token / Cost 治理

现状：

- 已有 token budget warning / terminate 基础能力

下一步：

- 增加 per-agent / per-sub-agent 消耗统计
- 把 token budget 扩展为更完整的成本治理模型
- 将 token、耗时、工具调用量纳入统一观测指标

### 3. Provider Reliability

- 增加主 provider 失败后的 fallback / retry 策略
- 统一 provider 能力差异的降级行为
- 补齐 provider 级别测试矩阵

### 4. Streaming 与服务接口

- 在现有 stream queue 基础上补 HTTP SSE / WebSocket 端点
- 补容器化、API server、部署文档
- 让运行状态、工具进展、引用来源对外可消费

### Phase 3 交付结果

- 长任务恢复更稳定
- 成本与预算可跟踪
- provider 故障更可控
- 更接近生产环境部署要求

## Phase 4：差异化与生态扩展

目标：放大项目真正有优势的方向，而不是变成通用功能清单。

### 1. Citation / Research Quality

现状：

- 已有 SourceRegistry 和 citation summary 基础

下一步：

- 引用绑定到具体结论，而不只是最终汇总
- 增加 confidence / evidence strength 评分
- 增加研究结果结构化导出格式

### 2. Transcript → Eval → Training Data

- 基于 transcript 构建 replay、评估、数据导出能力
- 对接 DeepResearch Bench、DRACO 等评测方式
- 支持轨迹导出为 SFT / preference 数据

### 3. Secure Context 与企业场景

- 扩展 SecureContext 到更完整的企业私有数据链路
- 增加更清晰的 on-prem / local model / private tool 接入模式
- 放大“可编程 + 隐私友好”的框架定位

### 4. 工具与 Provider 生态

- 扩展更多 provider、本地模型和常用研究工具
- 提供 MCP server 模式，让本框架被其他 agent 调用
- 在文档和模板层面降低接入成本

### Phase 4 交付结果

- 形成可区别于通用 research agent 的核心优势
- 让 transcript、citation、secure context 成为框架卖点
- 逐步形成生态能力

## 未来 4 到 6 周优先级

如果只看接下来 4 到 6 周，建议优先做这几件事：

1. 修正 `keep_tool_result` 语义，让它只处理 `MT.TOOL_RESULT`，不再截断任意历史
2. 完成 tool result 生命周期收口，统一 `role="user"` / `role="tool"` 的压缩与恢复逻辑
3. 补齐 resume 恢复链路，把 `todo_state` 和 offloaded content 真正接回来
4. 完成 runtime / hooks 主路径收口，修正 `InputCompiler` 和 `on_query_compile`
5. 补端到端、恢复链路、GPT native tool path 的关键测试
6. 将配置校验升级为 fail-fast，并逐步缩小关键模块中的广义异常捕获

## 下一步建议

如果只做下一轮迭代，我建议按这个顺序执行：

1. 先修 `keep_tool_result`
   这是当前最容易造成“行为和配置不一致”的点，修复收益高，改动边界也相对清晰。
2. 再修 resume 恢复完整性
   把 `todo_state`、offloaded content、session memory 恢复接完整，避免长任务中断后状态失真。
3. 然后统一 GPT native tool result 压缩链
   让 `role="tool"` 的 `MT.TOOL_RESULT` 也进入 compact / microcompact，避免 provider 行为分叉。
4. 最后收 runtime hooks
   把 `InputCompiler` 和 `on_query_compile` 这条链接回 runtime-scoped hooks，减少残余全局状态。

## 定位建议

不要把它做成又一个终端产品形态的“Deep Research clone”。

更适合这个项目的定位是：

- 可编程的深度研究引擎
- 面向开发者和企业的研究工作流内核
- 以 hooks、skills、secure context、execution modes 为核心差异化能力

## 一句话总结

当前 roadmap 的重点，已经不是“提出新方向”，而是把已经开始的架构升级做完整。

短期先收内核和可靠性，中期再补生产化，长期放大 citation、transcript、secure context 这些真正有辨识度的优势。
