# 技术 Roadmap（v1.2.2 之后）

> 状态基线：本文基于当前 `main` 分支的 `v1.2.2` 仓库状态整理，目标是回答“下一阶段最值得做什么”，而不是重复历史计划。

## 一句话判断

项目已经从“功能扩张期”进入“运行时收敛期”。

当前最重要的，不是继续横向加 feature，而是把下面三件事做扎实：

- 运行时 contract 收敛
- resume / context / offload 语义做实
- 测试、benchmark、发布工程补到可以长期演进的程度

## 当前阶段评价

和常见开源 Agent 框架相比，这个项目的优势不是“工作流 DSL”，而是完整的研究型执行内核：

- `MainLoopRunner`、`ContextManager`、`SessionMemory`、`TodoTracker`、`Transcript` 已经形成闭环
- `Hook`、`Prompt`、`Skill`、`ToolManager` 都具备框架级扩展能力
- 长任务能力明显强于大多数只做 prompt + tool wrapper 的项目

但和业界成熟 runtime 相比，当前差距主要在：

- 主链职责仍然集中
- 路由、恢复、压缩的契约还没有完全统一
- 发布与质量门槛还不够严格

## 总体目标

未来 2 到 3 个版本的目标不是“功能更多”，而是让框架变成一个更可信的底座：

1. 主循环行为可预测
2. 恢复语义可验证
3. 不同 provider / mode / tool transport 的行为边界清晰
4. 文档、测试、版本号、发布产物保持一致

## 当前最需要解决的问题

### P0：运行时 contract 不够统一

当前 `auto` 路由、`resume()`、`offload/read_result`、`context compaction` 各自都已经有实现，但组合起来还会出现行为漂移。

典型风险包括：

- `auto` 路由与 `router_model` / hook 的契约不完全一致
- 恢复后的大结果重新进入上下文后，无法回到原本的压缩生命周期
- `resume()` 无法完整恢复 dedup / turn result registry

### P1：主链文件职责持续累积

`MainLoopRunner`、`Orchestrator`、`ContextManager` 仍然承担了过多横切逻辑：

- mode policy
- builtin tool dispatch
- verify / summary policy
- offload lifecycle
- checkpoint / resume

这会让后续功能继续往主链堆积，回归风险越来越高。

### P2：质量门槛还没有收紧到框架级标准

当前测试基础已经不错，但还需要把质量关口继续前移：

- `pytest` 已经可以当回归网
- `ruff` 还没有成为硬门槛
- benchmark 已有雏形，但还没有形成长期评估体系
- release metadata 与版本管理还不够严谨

## 分阶段路线图

## v1.2.3：稳定性修复版

目标：把 `v1.2.2` 暴露出的核心行为问题收口，先把 runtime 语义补齐。

### 核心任务

- 修复 `auto` 路由链路，确保 `router_model`、`on_route_classify`、`on_route_apply` 的行为和文档一致
- 修复 resume 后 offloaded 大结果重新进入历史但无法再次压缩的问题
- 为 `resume()` 持久化或重建 `_call_registry`、`_dedup_cache`、offload registry 所需信息
- 统一 `pyproject.toml`、`mem_deep_research/__init__.py` 与 release tag 的版本号

### 测试补强

- 增加 `router_model` 生效链路测试
- 增加 `resume() -> continue execution` 端到端测试
- 增加 `read_result(ref="turn:N")` 在 resume 场景下的回归测试
- 增加版本一致性测试或 release 检查脚本

### 工程门槛

- 让 `ruff check` 进入默认 CI 成功条件
- 清理当前可直接修复的 import / unused / hygiene 问题

### 完成标准

- `auto` 路由、`resume`、`offload/read_result` 的行为与文档一致
- `pytest` 与 `ruff` 均默认通过
- 发布产物版本号不再漂移

## v1.3：运行时收敛版

目标：把“能工作”的主链重构成“更稳定、更清晰”的执行内核。

### 1. 拆执行策略

建议把当前主循环中的模式策略显式拆开：

```text
ModeResolver
  -> QuickProfile
  -> StandardProfile
  -> DeepProfile
```

优先拆出的逻辑：

- mode 解析与 route apply
- quick / standard / deep 的能力边界
- deep verify 与 final summary policy
- builtin tool dispatch policy

### 2. 收口结果生命周期

把 tool result 生命周期整理成单一语义链：

```text
raw tool result
  -> formatted result
  -> optional backup/offload
  -> compacted or offloaded marker
  -> restore/read_result
  -> resume rebuild
```

需要统一的不是实现细节，而是 contract：

- 什么状态下消息应视为 `TOOL_RESULT`
- 什么状态下应视为 `OFFLOADED`
- restore 之后是否允许再次压缩
- `read_result` 能保证恢复到什么程度

### 3. 升级 resume 模型

建议从“message history 快照 + 少量附加字段”升级为更明确的 runtime snapshot：

- message history
- session memory
- todo state
- call registry
- dedup cache
- offload registry
- effective mode / current turn / pending state

### 4. 缩小 God object

目标不是机械拆文件，而是降低主链认知负担。

建议优先抽出：

- `RouteController`
- `BuiltinToolDispatcher`
- `ResultLifecycleManager`
- `SummaryPolicy`

### 完成标准

- `MainLoopRunner` 不再同时承担 route / result lifecycle / summary policy 的全部细节
- `resume()` 恢复语义具备明确 contract 和测试覆盖
- `read_result`、offload、compact 行为保持一致

## v1.4：质量与评估体系版

目标：让框架从“架构不错”进入“可长期演进、可对外稳定使用”的阶段。

### 1. 建立分层 benchmark

建议拆成三层：

#### 离线回归集

只验证框架行为，不依赖实时世界状态：

- route
- dedup
- compact / summarize / emergency reduction
- offload / restore / read_result / resume
- sub-agent / todo / transcript

#### 在线稳定集

使用相对稳定的数据源，衡量：

- 完成率
- 事实正确率
- 引用覆盖率
- 成本 / 时延

#### 新鲜度 smoke 集

保留实时搜索类任务，只做能力 smoke，不纳入长期核心分数。

### 2. 补公共入口测试

优先补齐以下边界：

- `DeepResearch.run()`
- `DeepResearch.from_project()`
- `DeepResearch.resume()`
- `AgentFactory.run_batch()`
- `ToolManager` 在 stdio / SSE / streamable-http 下的行为差异

### 3. 规范 release 工程

建议补：

- 版本号单一来源
- CHANGELOG / release note 自动化
- 发布前检查脚本
- example project smoke test

### 4. 文档收口

重点不是再写更多文档，而是保证“实现、文档、配置名”三者一致。

优先清理：

- mode 相关文档
- resume / offload / read_result 契约说明
- hook 生命周期与推荐扩展点
- benchmark / release 流程说明

### 完成标准

- benchmark 能稳定比较不同版本的 runtime 行为
- 对外入口具备更高置信度
- release 流程不再依赖人工记忆

## 优先级排序

如果只看“投入产出比”，建议按这个顺序推进：

1. 修 `v1.2.2` 暴露出的 runtime 语义问题
2. 收口 `resume + offload + read_result + dedup`
3. 拆 `MainLoopRunner` 的策略职责
4. 把 `ruff`、版本管理、release 检查纳入硬门槛
5. 建 benchmark 分层体系

## 未来 4 到 8 周建议

### 第 1 阶段

- 完成 `v1.2.3` 修复项
- 清掉当前 lint 噪音
- 补 resume / route / read_result 的关键回归测试

### 第 2 阶段

- 拆 route / result lifecycle / summary policy
- 设计新的 runtime snapshot
- 把主循环恢复逻辑从“补丁式恢复”改成“显式恢复”

### 第 3 阶段

- 上线 benchmark 分层
- 补 release 工具链
- 把文档收敛到“可以指导外部用户”的程度

## 不建议当前阶段优先做的事

在主链 contract 还没稳定前，不建议优先投入：

- 更多执行模式分支
- 更多内置工具
- 更复杂的 planning DSL
- 更重的 UI / 可视化层

这些方向不是不重要，而是现在继续扩功能会放大主链复杂度。

## 相关文档

- [17-repo-architecture-review](./17-repo-architecture-review.md)
- [16-dual-mode-execution-plan](./16-dual-mode-execution-plan.md)
- [18-offload-evidence-optimization](./18-offload-evidence-optimization.md)
- [13-execution-modes](./13-execution-modes.md)
