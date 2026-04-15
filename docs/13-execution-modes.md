# 执行模式与语言控制

## 执行模式

```yaml
main_agent:
  execution_mode: auto   # auto | quick | standard | deep
```

当前框架使用两层概念：

- `execution_mode`
  用户配置的目标模式
- `effective_mode`
  实际执行模式，由 `auto` 路由后得到

## 四种模式

| 模式 | 工具 | 反思 | 子 Agent | verify | 典型场景 |
|------|------|------|---------|--------|---------|
| `quick` | 是 | 否 | 否 | 否 | 简单问答、翻译、轻量任务 |
| `standard` | 是 | 否 | 否 | 否 | 一般多步工具任务 |
| `deep` | 是 | 是 | 是 | 是 | 调研、报告、复杂分析 |
| `auto` | 路由后决定 | 路由后决定 | 路由后决定 | 路由后决定 | 默认模式 |

## `auto` 当前路由逻辑

`auto` 不再是“`task_engine.enabled=true` 就 deep，否则 standard”的旧逻辑。

当前优先级是：

1. `on_route_classify` hook
2. 结构信号
3. LLM 分类
4. 默认 `standard`

### 结构信号

- 配置了显式 `sub_agents`：优先路由到 `deep`
- 没有任何工具：优先路由到 `quick`

### LLM 分类

- 配置了 `main_agent.llm.router_model` 时，用轻量模型分类
- 未配置时，回退用主模型分类

### 默认值

如果前面都没有给出结果，回到 `standard`

## Quick 模式

Quick 模式是一个真正的 fast path，而不只是“少跑几轮”。

当前行为：

- 最大轮次被限制为 quick 上限
- 移除重型内置工具：
  - `spawn_agent`
  - `update_todo`
  - `read_result`
- 动态注入 `presets/quick.md`
- 不做 reflection
- 不走深度研究型 verify

适用：

- 简单问答
- 轻量转换
- 小型事实查找

## Standard 模式

标准多轮工具循环。

包含：

- 正常工具调用
- dedup
- context management
- monitoring
- skill

但默认不启用 deep 特有的研究流增强。

## Deep 模式

Deep 模式仍然建立在同一个主循环上，但会打开研究型能力：

- reflection checkpoint
- 子 Agent
- final summary 强制收尾
- verify checkpoint
- 更适合长任务的 todo / session memory 链路

需要注意：

- deep 能力主要由 `effective_mode == deep` 驱动
- `task_engine` 更像 deep 模式的参数区，而不是 auto 路由本身

## `task_engine` 的当前语义

```yaml
main_agent:
  task_engine:
    enabled: false
    reflection_interval: 5
    auto_planning: false
    enable_verify: true
```

当前建议理解为：

- `execution_mode`
  决定走哪条模式路径
- `task_engine`
  决定 deep 相关能力如何表现

尤其是：

- `auto_planning` 只有在任务规划启用时才生效
- `reflection_interval` 控制 deep 模式下注入反思的频率
- `enable_verify` 控制 summary 前 verify

## 内置工具与模式的关系

框架内置工具包括：

- `spawn_agent`
- `update_todo`
- `read_result`
- `tool_search`

其中：

- `auto` 模式会先注入完整集合
- `quick` 会在进入主循环后裁剪掉重型工具

## 语言控制

```yaml
main_agent:
  response_language: auto   # auto | Chinese | English | Japanese | ...
```

### 规则

| 值 | 行为 |
|---|------|
| `auto` | 根据 query 自动检测语言 |
| `Chinese` | 强制中文 |
| `English` | 强制英文 |
| 其他语言名 | 传给 prompt 作为目标语言 |

### 向后兼容

旧配置：

```yaml
main_agent:
  chinese_context: true
```

等同于：

```yaml
main_agent:
  response_language: Chinese
```

## 相关文件

- `core/llm_router.py`
- `core/main_loop.py`
- `core/orchestrator.py`
- `config_schema.py`
