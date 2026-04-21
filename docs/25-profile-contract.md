# Profile 契约设计

> 状态基线：2026-04-21
> 文档定位：v1.4.0 "deep research 从主链降级为 profile" 的抽象接口与迁移路径
> 配套阅读：`docs/20-roadmap.md`（版本路线图）、`docs/22-profile-boundary.md`（Runtime/Profile 边界盘点）、`docs/21-industry-framework-analysis.md`（定位转向依据）

## 目标

把当前散落在 `MainLoopRunner` 和 `Orchestrator` 里的 "研究场景专属逻辑"（约 29 处 `is_deep_mode` / `is_quick_mode` / `task_engine_cfg` / `generate_summary` / `<evidence>` 分支）收敛成一个独立的 `Profile` 抽象对象，主循环只调 profile 钩子，不感知内部是研究、自动化、编码还是 workflow。

**这不是重写功能，是重组结构**：迁移前后 `DeepResearchProfile` 的行为必须与当前 deep 模式 100% 等价。

## 设计原则

1. **主循环无 profile 知识**：主循环代码里不再出现 `is_deep_mode` / `is_quick_mode` 的功能性分支（只保留 `quick_mode max_turns` 等性能优化分支）
2. **StandardProfile 是空实现试金石**：如果 StandardProfile 的方法都是 `pass`，说明 Runtime 边界划分正确
3. **Breaking change 允许**：`execution_mode` 配置字段可以迁移到 `profile` 字段，不做别名兼容
4. **Profile 内聚，Runtime 通用**：Profile 内部可持有自己的 handler（SummaryHandler / TaskPlanner / reflection prompt builder），不侵入 runtime 层

## Profile 抽象接口

### 核心契约

```python
from abc import ABC
from typing import Protocol

class ProfileContext(Protocol):
    """Profile 方法可访问的 runtime 状态（只读视图）"""
    turn_number: int
    message_history: list[dict]
    assistant_response_text: str
    tool_calls: list[dict]
    tool_calls_executed: int
    completed_tool_results: list[tuple[str, dict]]
    task_description: str
    session_memory: Any
    todo_tracker: Any
    context_manager: Any
    llm_client: Any
    # Runtime 内部句柄（profile 需要调用时）
    hooks: Any

class Profile(ABC):
    """执行策略的抽象基类。
    
    每个 profile 代表一种 agent 执行风格（research / standard / automation / ...）。
    主循环在固定生命周期点调 profile 钩子，profile 返回决策或执行副作用。
    
    所有钩子都提供默认实现（= StandardProfile 的行为），子类按需覆盖。
    """
    
    name: str  # "standard" / "deep_research" / ...
    
    # ========== 启动阶段 ==========
    
    async def on_agent_start(self, ctx: ProfileContext) -> None:
        """Agent 启动时调一次。
        
        典型用途：DeepResearchProfile 在此注入 task plan。
        """
        pass
    
    async def build_initial_system_prompt(
        self,
        base_prompt: str,
        ctx: ProfileContext,
    ) -> str:
        """可选：修改 initial system prompt。
        
        典型用途：注入 profile-specific 指令（如 "你是研究助手，请在答案中用 <evidence> 标记引用"）。
        """
        return base_prompt
    
    # ========== 每轮生命周期 ==========
    
    async def on_turn_start(self, ctx: ProfileContext) -> None:
        """每轮开始时调。
        
        典型用途：
        - DeepResearchProfile 在 reflection interval 到时决定是否注入反思 prompt（见 should_inject_reflection）
        """
        pass
    
    async def should_inject_reflection(self, ctx: ProfileContext) -> bool:
        """每轮开始后调，决定是否注入反思 prompt。
        
        StandardProfile 返回 False。
        DeepResearchProfile 按 turn_number % reflection_interval == 0 决定。
        """
        return False
    
    async def build_reflection_prompt(self, ctx: ProfileContext) -> str | None:
        """返回反思 prompt 文本；None 表示不注入。
        
        仅在 should_inject_reflection 返回 True 时调用。
        """
        return None
    
    # ========== LLM 响应后 ==========
    
    async def on_llm_response(
        self,
        assistant_text: str,
        ctx: ProfileContext,
    ) -> str:
        """LLM 响应后调，返回可能被修改的 assistant_text。
        
        典型用途：
        - DeepResearchProfile 抽取 <evidence> tag 到 session_memory，返回清理后的 text
        """
        return assistant_text
    
    # ========== 工具执行前后（与 HITL 无关，HITL 是 Runtime 能力）==========
    
    async def on_before_tools(
        self,
        tool_calls: list[dict],
        ctx: ProfileContext,
    ) -> list[dict]:
        """工具批次执行前调，返回可能被修改/重排的 tool_calls。
        
        主循环已有的 on_tool_filter hook 是通用扩展点；这里是 profile 专属决策。
        StandardProfile 直接返回原 list。
        """
        return tool_calls
    
    async def on_after_tools(
        self,
        results: list[tuple[str, dict]],
        ctx: ProfileContext,
    ) -> None:
        """工具结果收齐后调。
        
        典型用途：
        - DeepResearchProfile 更新 source registry / evidence binding
        """
        pass
    
    # ========== 验证检查点（deep 专属）==========
    
    async def should_run_verify(self, ctx: ProfileContext) -> bool:
        """决定是否执行 verify checkpoint。
        
        StandardProfile 返回 False。
        DeepResearchProfile 按配置 enable_verify 决定。
        """
        return False
    
    async def run_verify(self, ctx: ProfileContext) -> dict | None:
        """执行 verify checkpoint，返回检查结果（None 表示无需处理）。
        
        仅在 should_run_verify 返回 True 时调用。
        """
        return None
    
    # ========== 最终答案 ==========
    
    async def build_final_answer(
        self,
        last_assistant_text: str,
        message_history: list[dict],
        ctx: ProfileContext,
    ) -> str:
        """决定如何生成最终答案。
        
        StandardProfile：直接返回 last_assistant_text。
        DeepResearchProfile：当有 tool 调用时跑 SummaryHandler 生成结构化 summary。
        """
        return last_assistant_text
    
    # ========== 配置 ==========
    
    @classmethod
    def default_config(cls) -> dict:
        """返回 profile 默认配置。"""
        return {}
    
    def validate_config(self, config: dict) -> dict:
        """验证并返回规范化的配置。无效时抛 ConfigValidationError。"""
        return config
```

### ProfileContext 的两种形态

- **只读视图**：大多数 profile 方法收到的是只读 `ProfileContext`，不能直接改 message_history（要改走 `build_reflection_prompt` 返回值之类的显式路径）
- **可写副作用**：少数方法（`on_llm_response` 返回值、`on_after_tools`）允许通过返回值或显式 API 写入 session_memory 等 profile-owned 数据结构

这样避免 profile 变成"隐式修改 runtime 状态"的黑盒。

## 当前研究专属逻辑到 Profile 方法的映射

基于 `docs/22-profile-boundary.md` 和代码盘点：

| 当前位置（`main_loop.py`） | 现状分支 | Profile 方法 | 迁移方式 |
|--------------------------|---------|-------------|---------|
| ~line 715 `task_planner.create_plan` + 注入 | `not is_quick_mode and self.task_planner.enabled` | `on_agent_start` | 只在 DeepResearchProfile 实现 |
| ~line 1045 `is_deep_mode` reasoning_effort 同步 | `is_deep_mode` | 移到 adaptive 分类结果处理（非 profile 职责） | - |
| ~line 1077 inline skill selector 注入 | `not is_quick_mode and self.inline_skill_selector` | `on_llm_response`（skill 注入作为 profile 决策）| StandardProfile 默认调用 selector；quick 不调 |
| ~line 1369-1373 evidence tag 抽取 | 永远执行（但只有 deep 产出 evidence） | `on_llm_response` | DeepResearchProfile override，StandardProfile 直接 return |
| ~line 1165-1170 `_reflection_pending` 豁免 | 永远执行（deep 才有反思） | Runtime 保留（反思轮标志位由 profile 通过 `build_reflection_prompt` 触发，豁免处理在主循环）| Runtime 识别 MT.REFLECTION 消息 |
| ~line 1524-1545 reflection 注入 | `not is_quick_mode and turn_counter.should_inject_reflection()` | `on_turn_start` + `should_inject_reflection` + `build_reflection_prompt` | DeepResearchProfile override |
| ~line 1628-1635 `_run_verify_checkpoint` | `is_deep_mode and task_engine_cfg.enable_verify` | `should_run_verify` + `run_verify` | DeepResearchProfile override |
| ~line 1641-1651 `generate_summary` 判定 | `deep + tool_calls > 0` 强制 true | `build_final_answer` 内部判定 | DeepResearchProfile override |
| ~line 368 `_clean_last_assistant(<evidence>)` | 永远执行 | `on_llm_response` | DeepResearchProfile 在抽取 evidence 后清理 tag |

**迁移后 `MainLoopRunner` 删除的字段**：
- `is_deep_mode` / `is_quick_mode`（替换为 profile）
- `task_engine_cfg` 参数（传入 profile 构造，runtime 不感知）
- `task_planner` / `inline_skill_selector` 字段（移到 profile 内部持有）

**迁移后 `MainLoopRunner` 保留的字段**：
- `_reflection_pending`（runtime 循环状态，不是 profile 专属）
- `effective_mode`（只用于 adaptive 路由的元数据，不驱动分支）
- `quick_mode max_turns` 限制（纯性能优化，通过 profile `default_config` 表达）

## 执行模式 vs Profile 的关系

**重要澄清**：`quick` / `standard` / `deep` 不是 profile 的别名，它们是**执行资源等级**，和 profile 正交。

| 维度 | 含义 | 举例 |
|------|------|------|
| **Profile** | 做什么类型的任务 | research / automation / coding |
| **Mode** | 投入多少资源 | quick（少轮快速）/ standard（中等）/ deep（多轮高投入）|

原来耦合的原因：deep research 的 "deep 模式" 同时意味着 "研究类型 + 高投入资源"。拆分后：

```yaml
# 迁移前（v1.3.x）
main_agent:
  execution_mode: deep          # 混合概念

# 迁移后（v1.4.0）
main_agent:
  profile: deep_research        # 研究类型
  mode: deep                    # 资源等级
```

默认组合：
- `profile: deep_research` + `mode: deep` = 当前默认研究行为
- `profile: deep_research` + `mode: quick` = "研究 profile 的快速版"（skip verify、shorter max_turns）
- `profile: standard` + `mode: standard` = 通用 agent，中等资源
- `profile: workflow_node` + `mode: quick` = 作为 workflow 节点，每次只做一步

Profile 方法内可以读 `ctx.mode` 做进一步决策（比如 `DeepResearchProfile.should_run_verify` 在 mode=quick 时返回 False）。

## 内置 Profile

### `StandardProfile`（默认）

全部方法空实现或 pass-through。对应当前 `execution_mode=standard` 且无 task_engine 的行为。

```python
class StandardProfile(Profile):
    name = "standard"
    # 所有方法用基类默认实现
```

### `DeepResearchProfile`

聚合所有当前研究专属逻辑。内部持有：

- `TaskPlanner`（原 `core/task_planner.py`）
- `SummaryHandler`（从 `llm_call_handler.py` 移入 profile）
- `EvidenceExtractor`（封装当前 `_extract_evidence_tags` 函数）
- `ReflectionPromptBuilder`（封装当前 `generate_reflection_prompt`）
- `VerifyChecker`（封装当前 `_run_verify_checkpoint`）

```python
class DeepResearchProfile(Profile):
    name = "deep_research"
    
    def __init__(self, config: dict):
        self.config = self.validate_config(config)
        self.task_planner = TaskPlanner(config.get("task_planner", {}))
        self.summary_handler = SummaryHandler(config.get("summary", {}))
        self.evidence_extractor = EvidenceExtractor()
        self.reflection_builder = ReflectionPromptBuilder(config)
        self.verify_checker = VerifyChecker(config.get("verify", {}))
    
    @classmethod
    def default_config(cls) -> dict:
        return {
            "task_planner": {"enabled": False},
            "reflection_interval": 5,
            "enable_verify": True,
            "generate_summary": True,
            "summary": {...},
            "evidence_extraction": True,
        }
    
    async def on_agent_start(self, ctx):
        if self.task_planner.enabled and ctx.mode != "quick":
            plan = await self.task_planner.create_plan(ctx.task_description, ...)
            # 通过 ctx 提供的显式 API 注入
            ctx.inject_system_message(plan_text, msg_type=MT.TASK_PLAN)
    
    async def should_inject_reflection(self, ctx):
        if ctx.mode == "quick":
            return False
        return ctx.turn_number > 0 and ctx.turn_number % self.config["reflection_interval"] == 0
    
    async def build_reflection_prompt(self, ctx):
        return self.reflection_builder.build(ctx)
    
    async def on_llm_response(self, assistant_text, ctx):
        # 抽取 <evidence> tag
        if self.config["evidence_extraction"]:
            assistant_text = self.evidence_extractor.extract(
                assistant_text, ctx.turn_number, ctx.session_memory
            )
        return assistant_text
    
    async def should_run_verify(self, ctx):
        return ctx.mode == "deep" and self.config["enable_verify"] and ctx.tool_calls_executed > 0
    
    async def run_verify(self, ctx):
        return await self.verify_checker.run(ctx)
    
    async def build_final_answer(self, last_assistant_text, message_history, ctx):
        # 当前逻辑：deep + 有 tool 调用 → 强制 summary
        need_summary = ctx.mode == "deep" and ctx.tool_calls_executed > 0
        if not need_summary or not self.config["generate_summary"]:
            return last_assistant_text
        return await self.summary_handler.generate(message_history, ctx)
```

### 未来预留（不在本版实现）

- `AutomationProfile`：订单处理、任务执行类，轻量 summary、强 guardrails
- `CodingProfile`：代码生成/审查，专注 `<file>` / `<diff>` tag 解析
- `WorkflowNodeProfile`：作为 workflow 节点运行，产出可被上层 workflow 消费的结构化结果

## API 变化

### 配置

```yaml
# 迁移前
main_agent:
  execution_mode: deep
  task_engine:
    enabled: true
    reflection_interval: 5
    enable_verify: true
  generate_summary: true

# 迁移后
main_agent:
  profile: deep_research          # profile 名称或 import path
  mode: deep                      # 资源等级
  profile_config:                 # profile 内部配置（结构由 profile.default_config 决定）
    task_planner:
      enabled: true
    reflection_interval: 5
    enable_verify: true
    generate_summary: true
```

### 代码入口

```python
# 迁移前
dr = DeepResearch(
    execution_mode="deep",
    ...,
)

# 迁移后：字符串方式（内置 profile）
dr = DeepResearch(
    profile="deep_research",
    mode="deep",
    profile_config={...},
)

# 迁移后：对象方式（自定义 profile）
dr = DeepResearch(
    profile=MyCustomProfile(my_config),
    mode="standard",
)
```

### Breaking Change 清单

| 旧 | 新 | 迁移路径 |
|----|----|---------|
| `execution_mode: deep` + `task_engine` 配置 | `profile: deep_research` + `mode: deep` + `profile_config` | 自动迁移脚本可选 |
| `execution_mode: standard` | `profile: standard` + `mode: standard` | 显式配置 |
| `execution_mode: quick` | `profile: standard` + `mode: quick` | 显式配置（quick 不是 profile） |
| `execution_mode: auto` | `profile: deep_research` + `mode: auto` | auto 是 mode 层行为，adaptive 分类只选 mode 不选 profile |
| `generate_summary` 顶层字段 | `profile_config.generate_summary` | 迁移到 profile_config |
| `DeepResearch(execution_mode=...)` | `DeepResearch(profile=..., mode=...)` | 用户代码需改 |

## Runtime 改造

### 主循环删除的代码

`MainLoopRunner` 中删除约 29 处分支，典型如：

```python
# 删除
if is_deep_mode:
    turn_counter.reflection_enabled = True
    if hasattr(self.llm_client, "reasoning_effort"):
        self.llm_client.reasoning_effort = adaptive_result.reasoning_effort

# 删除
if self.task_planner.enabled and not is_quick_mode:
    plan = await self.task_planner.create_plan(...)
    message_history.insert(...)

# 删除
cleaned = _extract_evidence_tags(assistant_response_text, turn_count, self.session_memory)
assistant_response_text = cleaned

# 删除
if not is_quick_mode and turn_counter.should_inject_reflection():
    reflection_prompt = generate_reflection_prompt(...)
    message_history.append(...)
    _reflection_pending = True

# 删除
if is_deep_mode and task_engine_cfg and task_engine_cfg.get("enable_verify", True):
    await self._run_verify_checkpoint(...)

# 删除
generate_summary = self.cfg.main_agent.get("generate_summary", False)
if effective_mode == EXECUTION_MODE_DEEP and total_tool_calls_executed > 0:
    generate_summary = True
```

### 主循环新增的代码

```python
# Agent 启动
await self.profile.on_agent_start(ctx)

# 每轮开始
await self.profile.on_turn_start(ctx)
if await self.profile.should_inject_reflection(ctx):
    prompt = await self.profile.build_reflection_prompt(ctx)
    if prompt:
        message_history.append(make_msg("user", prompt, MT.REFLECTION))
        _reflection_pending = True

# LLM 响应后
assistant_response_text = await self.profile.on_llm_response(
    assistant_response_text, ctx
)

# 工具执行前后
tool_calls = await self.profile.on_before_tools(tool_calls, ctx)
# ... execute tools ...
await self.profile.on_after_tools(results, ctx)

# 验证检查点（代替 _run_verify_checkpoint）
if await self.profile.should_run_verify(ctx):
    verify_result = await self.profile.run_verify(ctx)

# 最终答案（代替 is_simple_response + generate_summary + handle_summary）
final_answer = await self.profile.build_final_answer(
    last_assistant_text, message_history, ctx
)
```

### `MainLoopContext` 变化

```python
@dataclass
class MainLoopContext:
    # 删除
    # task_engine_cfg: dict | None
    # 其他原在主链的研究专属字段
    
    # 新增
    profile: Profile
    mode: str  # "quick" / "standard" / "deep" / "auto"
```

### Runtime 保留但调整的模块

- `llm_call_handler.py` 的 `SummaryHandler` → 移入 `core/profiles/deep_research/summary.py`
- `llm_call_handler.py` 的 `generate_reflection_prompt` → 移入 `core/profiles/deep_research/reflection.py`
- `main_loop.py` 的 `_extract_evidence_tags` → 移入 `core/profiles/deep_research/evidence.py`
- `main_loop.py` 的 `_run_verify_checkpoint` → 移入 `core/profiles/deep_research/verify.py`
- `task_planner.py` → 移到 `core/profiles/deep_research/task_planner.py`

## 新增目录结构

```
mem_deep_research_core/core/
├── profiles/
│   ├── __init__.py              # registry：name → Profile class
│   ├── base.py                  # Profile ABC + ProfileContext Protocol
│   ├── standard.py              # StandardProfile
│   └── deep_research/
│       ├── __init__.py          # DeepResearchProfile
│       ├── summary.py           # SummaryHandler（从 llm_call_handler 迁入）
│       ├── reflection.py        # generate_reflection_prompt + builder
│       ├── evidence.py          # EvidenceExtractor
│       ├── verify.py            # VerifyChecker
│       └── task_planner.py      # TaskPlanner（从 core/task_planner.py 迁入）
```

## 与 HITL / Runtime Snapshot 的关系

### 与 HITL（`docs/23`）
HITL 完全独立：`wait_for_human` 等是 Runtime 能力，任何 profile 都可以通过 hook 触发 HITL。Profile 可以在默认配置里**推荐**某些 tool 自动触发审批（写到 profile 文档里），但不能阻止 hook 层的业务逻辑。

### 与 Runtime Snapshot（`docs/24`，待写）
Snapshot 覆盖 profile 自身的状态：

```python
@dataclass
class RuntimeSnapshot:
    # ... runtime 字段 ...
    profile_name: str                      # 当前 profile 名称
    profile_state: dict                    # profile 自定义 state（通过 profile.snapshot() / restore()）
```

Profile 基类提供默认 `snapshot()` / `restore()`：

```python
class Profile(ABC):
    def snapshot(self) -> dict:
        """返回 profile 内部状态，用于 checkpoint。默认空。"""
        return {}
    
    def restore(self, state: dict) -> None:
        """从 snapshot 恢复 profile 内部状态。默认 no-op。"""
        pass
```

DeepResearchProfile 需要 snapshot 的典型状态：verify checkpoint 历史、task plan 当前节点。

## 分阶段实施

### Phase 1：接口落地 + StandardProfile

- 新增 `core/profiles/base.py` + `core/profiles/__init__.py` registry
- 实现 `StandardProfile`（全空实现）
- `MainLoopContext` / `DeepResearch` 入口接受 `profile` 参数，默认 `StandardProfile`
- 主循环在固定生命周期点调 profile 钩子（钩子当前都是 no-op，不改行为）
- 测试：StandardProfile 下主循环行为等价于 v1.2.5

**验收**：
- 543 个现有测试全过（行为零变化）
- 新增测试：`on_agent_start` / `on_turn_start` / `on_llm_response` 等钩子确实被调用
- `MainLoopRunner` 字段减少（`is_deep_mode` / `is_quick_mode` 先保留，下阶段移除）

### Phase 2：DeepResearchProfile 迁移

- 新增 `core/profiles/deep_research/` 目录结构
- 把研究专属代码**物理搬家**（不改逻辑）到 profile 子模块
- 实现 `DeepResearchProfile`，钩子 override 调 profile 子模块
- 主循环删除 29 处 `is_deep_mode` / `is_quick_mode` 功能性分支
- 配置入口接受 `profile: "deep_research"` 字符串和 Profile 对象两种形态
- 保留 `execution_mode` 字段的兼容映射（`deep` → `profile=deep_research, mode=deep`），CHANGELOG 标记为 deprecated

**验收**：
- 原 `execution_mode=deep` 的测试用例在 `profile=deep_research, mode=deep` 下行为等价
- 原 `execution_mode=quick` 的用例在 `profile=standard, mode=quick` 和 `profile=deep_research, mode=quick` 下各自的预期行为
- 主循环代码删减 ≥ 200 行
- `main_loop.py` 里 `is_deep_mode` / `is_quick_mode` 功能性分支数 = 0

### Phase 3：配置迁移 + 文档 + 废弃旧字段

- CHANGELOG 写明 Breaking change + 迁移路径
- 更新 `example_project` 使用新配置
- 删除 `execution_mode` 字段（v1.5.0 或下一个大版本），或保留 deprecation warning 一个版本
- 文档：`docs/profiles-guide.md` + 每个内置 profile 的独立文档
- 开放 `load_custom_profile(path)` 支持用户项目自定义 profile

**验收**：
- Example project 使用新 API 跑通
- 每个内置 profile 有独立文档
- 用户可写自定义 profile 扩展框架

## 关键决策

| # | 决策 | 结论 |
|---|------|------|
| 1 | Profile 和 Mode 的关系 | **正交**。Profile 决定"做什么类型任务"，Mode 决定"投入多少资源"。`profile=deep_research, mode=quick` 是合法组合 |
| 2 | Profile 是否可以修改 message_history | **不能直接改**。通过返回值（`on_llm_response` 返回修改后的 text）或显式 API（`ctx.inject_system_message`）。主循环拥有 message_history 写权 |
| 3 | SummaryHandler / TaskPlanner 归属 | 迁移到 `core/profiles/deep_research/` 内部。Runtime 层不再持有研究专属 handler |
| 4 | `execution_mode` 字段迁移策略 | Phase 2 保留兼容映射（带 deprecation warning），Phase 3/v1.5.0 删除 |
| 5 | `_reflection_pending` 归属 | Runtime 保留（是主循环状态机标志，不是 profile 专属数据）。Profile 通过 `build_reflection_prompt` 返回值触发反思，主循环识别并设置 pending 状态 |
| 6 | Hook 系统和 Profile 的关系 | 完全独立。Hook 是**业务扩展点**（用户项目注册）；Profile 是**执行策略**（框架内置 + 可扩展）。Hook 先于 profile 方法调用 |
| 7 | adaptive 路由是否被 profile 控制 | 否。Adaptive 路由选的是 **mode**（quick/standard/deep），不选 profile。Profile 由用户在启动时指定 |
| 8 | 自定义 profile 的加载方式 | 字符串（内置 registry）+ 对象（用户项目直接传入）。不提供 dynamic import path（避免任意代码执行风险） |

## 非目标（本版不做）

- **不做 workflow layer**：那是 v1.4.0 profile 拆分稳定后的下一步
- **不做 profile 热切换**：任务运行中不能切 profile；resume 时 profile 必须和 checkpoint 时一致（通过 `snapshot.profile_name` 校验）
- **不做 profile 组合**：一个任务只能用一个 profile。未来如果要"research + automation 混合"，通过 workflow layer 的多节点实现
- **不做 profile 级 hook 系统**：Hook 系统保持全局，profile 不引入独立 hook 命名空间
- **不做 RuntimeSnapshot 完整设计**：这是 `docs/24-runtime-snapshot-design.md` 的范围

## 关键风险

| 风险 | 严重度 | 缓解 |
|------|-------|------|
| 迁移后行为漂移（与 v1.2.5 deep 模式不等价）| 高 | Phase 1 / 2 的验收都要求"等价回归"；大量 e2e 测试对比 |
| Profile 接口设计不足，后续要补钩子 | 中 | 先按 22-profile-boundary 盘点的 29 个分支点定钩子；允许 v1.4.0 中期小幅扩展 |
| `_reflection_pending` 状态机拆分后主循环和 profile 不同步 | 中 | 状态机仍在主循环，profile 只出 prompt；`on_turn_start` 接口收到 `pending_reflection` 字段 |
| 用户项目的 `execution_mode` 配置批量失败 | 中 | Phase 2 保留一个版本的兼容映射 + deprecation warning |
| SummaryHandler 迁出后 context_limit 重试逻辑断裂 | 中 | 迁移时确保 LLMCallHandler 仍提供 `SummaryHandler` 所需的底层 LLM 调用能力，仅"策略层"移到 profile |
| Profile 持有 handler 导致状态污染 | 低 | Profile 实例与 DeepResearch 实例同生命周期；每次 `dr.run()` profile 不持久化可变状态（除非走 snapshot 路径） |

## 测试清单

### Phase 1
- StandardProfile 下 543 个现有测试全过
- Profile 钩子调用顺序测试（on_agent_start → 每轮 on_turn_start → on_llm_response → ...）
- Profile 对象传入 `DeepResearch(profile=...)` 正确接收
- 字符串 `profile="standard"` 从 registry 解析

### Phase 2
- `execution_mode=deep` 的所有现有用例在 `profile=deep_research, mode=deep` 下等价
- `profile=deep_research, mode=quick` 的新组合行为（跳过 verify、reflection_interval 不生效）
- Evidence 抽取从 profile 触发，main_loop 不调用 `_extract_evidence_tags`
- Reflection 注入从 profile 决定，main_loop 仅识别 `MT.REFLECTION` 消息设置 pending
- Summary 生成仅由 profile 决定，`generate_summary` 顶层字段已废弃但兼容
- 主循环代码中 `is_deep_mode` / `is_quick_mode` 引用数 = 0（grep 验证）

### Phase 3
- Example project 使用新配置跑通
- 用户自定义 Profile 加载
- Deprecation warning 在 `execution_mode` 使用时触发
