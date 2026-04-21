# Mem Deep Research Framework Roadmap

> 状态基线：2026-04-21
> 当前版本：v1.2.5
> 文档定位：当前权威 Roadmap，用于回答“项目下一阶段做什么、为什么做、做到什么算完成”。
> 配套阅读：`docs/21-industry-framework-analysis.md`（业界对比 + 长期方向）。

## 一句话判断

项目已经完成“研究型 Agent 执行内核”的第一阶段建设。下一步的战略重点**不是继续做一个更强的 research agent，而是把 deep research 从"框架本体"降级为"框架上的一个高级执行 profile"，把底座从研究型内核升级为通用 Agent Runtime。**

## 定位转向

过去一年的演进路径是"围绕研究任务做得更深"。随着 `MainLoopRunner` / `ContextManager` / `ToolManager` / sub-agent / hook / skill / memory / resume 都已就位，继续沿着"研究单点"方向优化的边际收益在下降：主循环已经承担了大量研究专属逻辑，任何新场景（automation / coding / ops / enterprise workflow）都会继续往主链里堆条件分支。

因此定位调整如下：

- **框架本体** = 通用 Agent Runtime（主循环、context、tool、sub-agent、resume、observability、guardrails、workflow）
- **Deep Research** = 框架之上的一个高级执行 profile（聚合了反思、规划、offload、evidence extraction、summary 等研究专属策略）

这不是放弃 deep research，而是**让研究能力**作为 profile **在底座之上**清晰表达，同时给 workflow / automation / coding 等其他 profile 留出生长空间。

## 当前阶段

截至 2026-04-20，框架已经具备这些核心能力：

- `MainLoopRunner` + `ContextManager` + `ToolManager` 构成可运行的执行内核
- `quick / standard / deep / auto` 四种执行模式已经可用
- 子 Agent、Hook、Skill、Memory、Todo、Transcript 均已接入主链
- 上下文压缩、结果 offload、resume、监控与循环检测已经形成闭环
- v1.2.5 已移除 grace turn，主循环退出对齐 Claude Code `stop_reason` 语义

这意味着项目已经越过“功能证明可行”阶段，进入“框架可信度 + 通用化演进”阶段。

## 北极星目标

面向未来 2 到 3 个版本，Roadmap 聚焦五个方向（前两项是定位转向的直接动作，后三项是支撑底座可信的基础工作）：

1. **把 deep research 从主链逻辑降级为 profile**。保留 `quick / standard / deep`，但把 deep 明确为一种 profile；新增更通用的 workflow / automation / coding profile 概念；把研究专属逻辑逐步从主循环抽离到 policy / profile 层。
2. **补 workflow layer**。业界已经清晰分成 workflow（开发者定义结构化控制流）+ agent（局部节点开放式推理）两层。当前 agent loop 很强但缺 workflow，新增 `Flow` / `Process` / `TaskGraph` 抽象，允许 Agent 作为 workflow 节点。
3. 让主循环行为更可预测，降低不同 provider / mode / tool transport 的行为漂移。
4. 让长任务生命周期更统一，尤其是 `compact -> offload -> read_result -> resume` 这条链，把 durable execution 做成正式 contract。
5. 让框架具备更强的对外可用性，包括测试、文档、版本发布、observability 和 eval 体系。

## 版本路线图总览

| 版本 | 目标窗口 | 主题 | 核心结果 |
|------|---------|------|---------|
| v1.3.0 | 2026-05 | Runtime Contract 收敛 | 统一结果生命周期、配置契约、子 Agent 行为边界；为 profile 拆分准备契约基础 |
| v1.4.0 | 2026-06 | Profile 拆分 + Workflow Layer 雏形 | deep research 从主循环降为 profile；引入 `Flow` / `TaskGraph` 抽象；主循环瘦身 |
| v1.5.0 | 2026-07 至 2026-08 | 质量、Observability、对外可用性 | 建立评测体系、trace schema 收口、稳定入口 API、规范发布流程 |

> **说明**：v1.4.0 是定位转向的关键版本。此前版本的 "执行架构整理" 目标现在明确了方向 —— 不只是拆 mode 策略文件，而是把主循环里的研究专属逻辑（reflection / verify / task planner / evidence extraction / summary policy）全部收到 DeepResearchProfile 里，主循环本体只保留通用 agent runtime 能力。

## v1.3.0：Runtime Contract 收敛

### 目标

把“已经能跑”的运行时能力收口成清晰、稳定、可验证的 contract。

### 重点任务

#### 1. 统一结果生命周期

收口下面这条主链语义：

```text
tool result
  -> format
  -> offload / marker
  -> compact / summarize
  -> read_result / restore
  -> resume rebuild
```

需要明确：

- 什么消息会被视为标准 `tool_result`
- 什么状态下结果允许再次压缩
- `read_result(ref=...)` 能恢复到什么粒度
- `resume()` 需要重建哪些运行时状态

#### 2. 收口配置契约

继续把配置从“多入口可用”收口到“单一来源可信”：

- 核心字段必须 fail-fast
- provider / skill selector / prompt hint 的字段读取路径保持一致
- `example_project`、`README`、schema、运行时默认值保持同步

#### 3. 对齐主 Agent / 子 Agent 行为

重点解决“看起来共享能力，实际行为不完全一致”的问题：

- 子 Agent 默认继承主链 context 管理策略
- offload registry、dedup、resume 相关语义在主链和子链保持一致
- named sub-agent 与动态 spawn agent 的结果回收和清理逻辑一致

#### 4. 补运行时回归测试

优先补齐这些高价值回归场景：

- `resume()` 后继续执行
- `offload + read_result + compact` 联动
- 多任务 / 多 context 下 MCP session 隔离
- 子 Agent 返回结果重新进入主链后的生命周期

### 完成标准

- 结果生命周期可以用一张图讲清楚，且实现与文档一致
- `resume`、`read_result`、offload 的关键场景具备端到端回归测试
- 子 Agent 和主 Agent 的上下文管理行为不再静默漂移
- 配置写错核心字段时初始化明确失败，不再“带病运行”

## v1.4.0：Profile 拆分 + Workflow Layer 雏形

### 目标

让框架从 "一个研究型主循环" 升级为 "通用 Agent Runtime + 可插拔 profile + 雏形 workflow 层"。**这是项目定位转向的关键版本。**

### 重点任务

#### 1. Deep research 从主链逻辑降级为 profile

把目前散落在 `MainLoopRunner` / `Orchestrator` 里的研究专属逻辑，收到独立的 `DeepResearchProfile` 里：

- reflection 检查点（目前由 `task_engine_cfg` 驱动）
- verify checkpoint（deep 模式前的证据覆盖/冲突检测）
- task planner（LLM 任务分解）
- evidence extraction（`<evidence>` tag 解析）
- summary policy（deep + 用过工具强制生成 summary）
- context 压缩策略的研究向偏好

主循环本体只保留：turn loop、tool dispatch、stop_reason 判退、基础 context 管理、sub-agent orchestration。

形态参考：

```text
ModeResolver
  -> QuickProfile      (直答场景)
  -> StandardProfile   (通用工具调用 agent)
  -> DeepResearchProfile  (研究场景，含反思/规划/evidence/summary)
  -> (future) AutomationProfile / CodingProfile / WorkflowNodeProfile
```

#### 2. 引入 Workflow Layer 雏形

新增 `Flow` / `Process` / `TaskGraph` 抽象（任选一个命名），支持：

- 顺序、并行、路由、等待、人工确认、子流程节点
- Agent 作为 workflow 节点运行（而非把所有复杂性塞进单 agent loop）
- 与现有 `resume` / offload / checkpoint 机制集成

这一版不需要做到 LangGraph 级完整度，目标是让"复杂任务可以用 workflow 组合多个 agent 节点"这条路在框架层走得通。

#### 3. 缩小主链对象职责

优先抽出的模块：

- `RouteController`（mode 决策 + profile 选择）
- `BuiltinToolDispatcher`（spawn_agent / read_result / update_todo 等内置工具分发）
- `ResultLifecycleManager`（tool result format → offload → compact → restore 生命周期）
- `SummaryPolicy`（当前由 `generate_summary` + mode 隐式决定，抽成显式策略）

目的是降低 `MainLoopRunner` 和 `Orchestrator` 的认知负担，让后续功能不再默认往主链堆。

#### 4. 统一工具定义形态

减少 builtin tool、MCP tool、deferred tool 在数据 shape 上的分叉，建立统一 `ToolDefinition` 中间表示，让 prompt 构建、工具选择、工具执行和 profile 层使用一致契约。

#### 5. 收口公共入口语义

把以下入口的行为边界补清楚并文档化：

- `DeepResearch.run()` / `from_project()` / `resume()` / `run_batch()`
- 新增 `DeepResearch(profile=...)` 或等价显式 profile 指定 API
- Workflow 入口的调用 / 注册方式

### 完成标准

- `MainLoopRunner` 主循环不再包含 reflection / verify / evidence / summary policy 等研究专属分支
- `DeepResearchProfile` 可以独立单测，能拼装到其他入口
- quick / standard / deep 的行为差异通过 profile 定义表达，可解释、可测试、可替换
- Workflow layer 至少支持 "顺序 + 并行 + 单 agent 节点" 三个最小场景，且能在长任务中 resume
- 新功能（如新 profile）接入时，不需要修改 `MainLoopRunner`

## v1.5.0：质量与对外可用性

### 目标

让项目从“内部架构成熟”进入“可长期维护、可对外稳定使用”的阶段。

### 重点任务

#### 1. 建立分层评测体系

建议拆成三层：

- 离线回归集：验证 runtime contract，不依赖实时世界状态
- 在线稳定集：验证完成率、引用覆盖率、时延与成本
- 新鲜度 smoke 集：只做实时搜索类能力冒烟，不纳入核心长期分数

#### 2. 补公共边界测试

补齐更贴近用户使用方式的测试：

- `from_project()` + hooks 自动加载
- `run_batch()` 在不同 context 下的隔离
- 不同 transport 的 `ToolManager` 行为差异
- provider 差异对工具调用与退出语义的影响

#### 3. 规范版本与发布流程

目标是让版本不再漂移、发布不再依赖人工记忆：

- 版本号单一来源
- CHANGELOG 与发布标签自动校验
- 发布前 smoke test
- example project 作为最小可用验证链路

#### 4. 收口文档体系

重点不是“再写更多文档”，而是让文档、实现和示例保持一致：

- roadmap 与 changelog 对齐
- 配置文档与 schema 对齐
- mode / resume / offload 契约文档明确
- hooks 和扩展点给出推荐用法

### 完成标准

- 不同版本之间的运行时表现可通过 benchmark 进行横向比较
- 公共入口的主要使用路径有稳定测试覆盖
- 发布流程具备自动化检查，版本号不再多处漂移
- 文档能支撑新用户从接入到扩展的完整路径

## 成功指标

为了避免 roadmap 变成纯叙事，建议按下面几类指标跟踪：

### 运行时指标

- 长任务中断后 `resume()` 成功恢复比例
- context compact / summarize / offload 触发后的成功续跑比例
- 多任务场景下 tool session 串号问题数量

### 质量指标

- 关键主链端到端测试覆盖率
- benchmark 任务通过率
- 每个版本回归 bug 数量

### 可维护性指标

- 主循环核心文件复杂度变化趋势
- 新增功能需要修改主链核心文件的比例（目标：v1.4.0 后新 profile 接入 = 0 主链改动）
- 文档与实现不一致问题数
- 研究专属逻辑在主循环中的占比（目标：v1.4.0 后收敛到 0，全部进入 `DeepResearchProfile`）

### 对外可用性指标

- example project 首次跑通成功率
- 用户接入常见问题是否能在文档中直接找到答案
- 发布后版本号、CHANGELOG、文档之间的一致性

## 当前优先级排序

2026-04-21 之后 4 到 8 周内最值得投入的事情，顺序如下：

1. 收口 `resume + offload + read_result + compact` 的统一 contract，为 profile 拆分准备契约基础
2. 对齐主 Agent / 子 Agent 的 context 与结果生命周期
3. **设计 `DeepResearchProfile` 抽象**，为 v1.4.0 的主循环瘦身做准备（可以先作为内部概念，不对外暴露）
4. 补公共入口、profile 切换、运行时隔离相关测试
5. 规范版本、发布与文档同步流程

## 明确不优先做的事情

在定位转向完成前，以下方向不建议作为主线投入：

- 继续优化 deep research 的 prompt 细节和 reflection 策略（这些将在 v1.4.0 被重构到 profile 里）
- 为研究场景往主循环里塞新 heuristics
- 大量新增内置工具
- 在 workflow 抽象缺位时过早强化多 Agent 花样
- 继续扩大 prompt 模板分支

原因：**当前项目的稀缺性不在"能力更多"，而在"底座更稳、更通用、更容易演进到其他场景"**。任何在主循环里新增的研究专属分支，都是 v1.4.0 拆分时需要偿还的债。

## 结论

这个项目已经具备成为**通用 Agent Runtime 底座 + 以 research 为强项**的潜力。接下来的关键，不是继续证明它"能做一个更强的 research agent"，而是证明它"有能力不被 research 绑定，能扩展到 workflow / automation / coding 等更多场景"。

因此，这份 Roadmap 的主旨是：

**先把运行时 contract 收紧，然后把 deep research 从主循环降级为 profile，补上 workflow layer 雏形，最后把质量、observability 和发布体系补齐。**
