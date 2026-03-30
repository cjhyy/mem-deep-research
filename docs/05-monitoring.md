# 执行监控

ExecutionMonitor 提供运行时保护，防止 Agent 陷入无限循环、卡死或超时。

## 三级升级机制

```
┌─────────┐     ┌──────────────┐     ┌───────────┐
│  WARN   │ ──→ │ INJECT_HINT  │ ──→ │ TERMINATE │
│ 日志警告 │     │ 注入提示到上下文│     │  强制终止  │
└─────────┘     └──────────────┘     └───────────┘
```

| 级别 | 动作 | 说明 |
|------|------|------|
| `WARN` | 记录日志警告 | 通知监控系统，不影响执行 |
| `INJECT_HINT` | 注入提示消息 | 向消息历史注入"换策略"提示，同时提升 temperature |
| `TERMINATE` | 强制终止循环 | 终止主循环，进入摘要生成阶段 |

## 检测维度

### 1. 超时检测

```yaml
monitoring:
  max_total_time: 600.0      # 最大总执行时间（秒）
```

超过 `max_total_time` 后触发 TERMINATE。

### 2. 卡死检测

```yaml
monitoring:
  stall_detection_threshold: 120.0   # 无进展阈值（秒）
```

- 超过阈值但在 `stall_terminate_multiplier`（默认 2x）范围内 → WARN
- 超过阈值 × multiplier → TERMINATE

### 3. 响应循环检测

检测 LLM 反复生成相似响应的情况：

```yaml
monitoring:
  enable_loop_detection: true
  response_hash_window_size: 8       # 滑动窗口大小
  response_hash_repeat_threshold: 3  # 重复阈值
  loop_escalation_terminate_threshold: 3  # 升级到 TERMINATE 的次数
```

**原理**: 对每次 LLM 响应的前 N 字符计算 hash，维护滑动窗口。当窗口内相同 hash 出现次数超过阈值时，判定为循环。

**升级逻辑**:
- 第 1 次检测到 → WARN
- 第 2 次检测到 → INJECT_HINT（注入提示 + 提升 temperature）
- 第 3 次检测到 → TERMINATE

### 4. 空响应检测

```yaml
monitoring:
  max_consecutive_empty_turns: 3    # 最大连续空响应
```

LLM 连续返回空响应超过阈值后终止。

### 5. 工具循环检测

```yaml
monitoring:
  max_tool_loop_retries: 2          # 工具循环重试限制
```

同一工具调用模式重复出现时的升级处理。

## INJECT_HINT 行为

当升级到 INJECT_HINT 时，系统会：

1. **提升 temperature**: 增加 `temperature_boost`（默认 0.3），上限 `temperature_boost_cap`（默认 1.0）
2. **注入提示**: 向消息历史插入循环打破提示

循环打破提示内容（`get_loop_break_hint()`）：
- 列出已尝试过的策略（最多 10 条）
- 列出最近使用的工具（建议避免重复）
- 要求 Agent 综合已有信息 / 尝试新方法 / 承认信息不足

## ExecutionMonitor API

```python
class ExecutionMonitor:
    def __init__(self, config: MonitoringConfig, stream_reasoning_callback)

    # 集成入口（每轮调用）
    async def pre_turn_check(self) -> EscalationAction
    async def post_turn_check(self, response_text, llm_call_failed) -> EscalationAction

    # 独立检测
    async def check_timeout(self) -> EscalationAction
    async def check_stall(self) -> EscalationAction
    def record_progress(self, response_text) -> EscalationAction
    def record_empty_response(self) -> bool
    def record_tool_loop_warning(self) -> EscalationAction

    # 升级处理
    async def handle_loop_detected(self, action: EscalationAction) -> None
    def get_loop_break_hint(self, recent_tool_names: list) -> str

    # 状态查询
    def get_elapsed_time(self) -> float
    def get_status_summary(self) -> dict
    def reset(self) -> None
```

## TurnCounter

辅助类，跟踪轮次进度：

```python
class TurnCounter:
    def __init__(self, max_turns, reflection_enabled=False, reflection_interval=5)

    def increment(self) -> int              # 递增并返回当前轮次
    def is_max_reached(self) -> bool        # 是否达到最大轮数
    def should_inject_reflection(self) -> bool  # 是否需要反思检查点
    def get_progress_percentage(self) -> float  # 进度百分比 0-100
```

## 配置参考

```yaml
monitoring:
  stall_detection_threshold: 120.0         # 卡死检测阈值（秒）
  max_total_time: 600.0                    # 最大总时间（秒）
  max_consecutive_empty_turns: 3           # 最大连续空响应
  enable_loop_detection: true              # 启用循环检测
  loop_detection_text_length: 500          # hash 取前 N 字符
  loop_escalation_terminate_threshold: 3   # TERMINATE 升级阈值
  response_hash_window_size: 8             # 滑动窗口大小
  response_hash_repeat_threshold: 3        # 重复阈值
  max_tool_loop_retries: 2                 # 工具循环重试
  stall_terminate_multiplier: 2.0          # 卡死硬超时乘数
  temperature_boost: 0.3                   # 循环时 temperature 提升
  temperature_boost_cap: 1.0               # temperature 上限
```
