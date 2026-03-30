# 配置系统

框架使用 Pydantic 进行配置验证，OmegaConf (Hydra) 进行运行时配置管理。所有配置通过 YAML 文件声明。

## 配置加载优先级

```
项目级配置 (project_dir/config/)  >  框架默认配置 (config/)
```

## 完整配置参考

```yaml
main_agent:
  # ─── Prompt 配置 ───
  prompt:
    agent_type: main              # main | worker
    tool_format: xml              # xml | native
    presets: []                   # 预设模板，如 [research, time_sensitive]
    custom_system_template: null  # 自定义 system prompt 模板名
    custom_summarize_template: null  # 自定义摘要模板名
    templates_dir: null           # 自定义模板目录

  # ─── LLM 配置 ───
  llm:
    provider_class: "ClaudeOpenRouterClient"  # Provider 类名
    model_name: "anthropic/claude-sonnet-4"   # 模型标识
    temperature: 0.3              # 采样温度
    top_p: null                   # nucleus 采样
    top_k: null                   # top-k 采样
    min_p: null                   # 最小概率采样
    max_tokens: 32000             # 最大输出 token
    timeout: 120                  # 请求超时（秒）
    max_context_length: 128000    # 最大上下文长度，-1 = 不限制
    keep_tool_result: 5           # 保留工具结果数，-1 = 全部保留
    enable_streaming: true        # 启用流式输出
    disable_cache_control: false  # 禁用缓存控制
    # 重试配置
    retry_max_attempts: 3         # 最大重试次数
    retry_strategy: "exponential" # exponential | fixed
    retry_multiplier: 1.0         # 指数退避乘数
    retry_wait_seconds: 2.0       # 固定等待时间
    # Provider 特定配置
    openrouter_api_key: "${oc.env:OPENROUTER_API_KEY,}"
    anthropic_api_key: "${oc.env:ANTHROPIC_API_KEY,}"
    openai_api_key: "${oc.env:OPENAI_API_KEY,}"

  # ─── 工具配置 ───
  tool_config:                    # 工具列表（YAML 文件名，不含扩展名）
    - tool-searching-serper
    - tool-calculator
  max_tool_calls_per_turn: 10     # 每轮最大工具调用数

  # ─── Agent 参数 ───
  max_turns: 20                   # 最大执行轮数
  execution_mode: auto            # auto | quick | standard | deep
                                  # auto: 框架根据任务复杂度自动选择
                                  # quick: 简单问答，少轮执行
                                  # standard: 需要工具的常规任务
                                  # deep: 深度研究模式
  max_concurrent_subagents: 3     # 并发子 Agent 数量上限（asyncio.Semaphore）
  response_language: auto         # auto | Chinese | English | Japanese | ...
                                  # auto: 从 query 自动检测语言
                                  # 替代旧的 chinese_context 配置
  chinese_context: false          # （已废弃，向后兼容）等同 response_language: Chinese

  # ─── TodoTracker ───
  todo_tracker:
    enabled: true                 # 启用任务追踪（内置 update_todo 工具）
                                  # 独立于 message_history，不受 context 压缩影响

  # ─── Skill 选择 ───
  skill_selection:
    enabled: true                 # 启用 Skill 系统
    method: inline                # rules | llm | inline
    max_skills: 3                 # 最大选择 Skill 数
    model: null                   # LLM 选择时使用的模型
    progressive: true             # 渐进式加载：首轮只加载 Skill 目录
                                  # 后续通过 <next_skills> 按需加载完整内容

  # ─── 上下文管理 ───
  context_manager:
    enable_dedup: true            # 启用工具调用去重
    enable_compact: true          # 启用上下文压缩
    compact_at_ratio: 0.6         # L1 触发阈值（token 占比）
    summarize_at_ratio: 0.8       # L2 触发阈值
    compact_keep_recent: 3        # 保留最近 N 轮不压缩
    result_offload_threshold: 5000  # 工具结果超过此字符数时卸载到文件系统
                                    # context 中只保留摘要引用

  # ─── 执行监控 ───
  monitoring:
    stall_detection_threshold: 120.0   # 卡死检测阈值（秒）
    max_total_time: 600.0              # 最大总执行时间（秒）
    max_consecutive_empty_turns: 3     # 最大连续空响应次数
    enable_loop_detection: true        # 启用循环检测
    loop_escalation_terminate_threshold: 3  # 循环升级终止阈值
    response_hash_window_size: 8       # 滑动窗口大小
    response_hash_repeat_threshold: 3  # 重复阈值

  # ─── 任务引擎（深度研究） ───
  task_engine:
    enabled: false                # 启用深度研究模式
    reflection_interval: 5        # 反思检查点间隔（轮数）
    auto_planning: false          # 自动任务分解

# ─── 子 Agent 配置 ───
sub_agents:
  worker:
    prompt:
      agent_type: worker
      tool_format: xml
    llm:
      provider_class: "ClaudeOpenRouterClient"
      model_name: "anthropic/claude-sonnet-4"
      temperature: 0.3
      max_tokens: 16000
    tool_config: [tool-searching-serper]
    max_turns: 10

# ─── 输出配置 ───
output_dir: "logs"                # 日志输出目录
```

## 配置验证

框架使用 Pydantic 模型 (`config_schema.py`) 验证配置：

```python
from mem_deep_research_core.config_schema import validate_agent_config

config_dict = {...}
validated = validate_agent_config(config_dict)  # 抛出 ValidationError
```

### 主要 Pydantic 模型

| 模型 | 说明 |
|------|------|
| `AgentConfig` | 顶层配置（包含 main_agent + sub_agents） |
| `MainAgentConfig` | 主 Agent 完整配置 |
| `LLMConfig` | LLM Provider 配置 |
| `PromptConfig` | Prompt 系统配置 |
| `SkillSelectionConfig` | Skill 选择策略配置 |
| `ContextManagerConfig` | 上下文管理配置 |
| `MonitoringConfigSchema` | 执行监控配置 |
| `TaskEngineConfig` | 任务引擎配置（深度研究 + 反思） |

## 环境变量引用

配置中支持 OmegaConf 环境变量语法：

```yaml
# 必须存在的环境变量
api_key: "${oc.env:API_KEY}"

# 带默认值（空字符串）
api_key: "${oc.env:API_KEY,}"

# 带默认值
api_key: "${oc.env:API_KEY,default-value}"
```

## 配置加载器

`ConfigLoader` (`utils/external_loader.py`) 负责配置加载：

```python
from mem_deep_research_core.utils.external_loader import config_loader

# 设置项目目录
config_loader.set_project_dir("./my_project")

# 加载工具配置（项目级 → 框架级）
tool_cfg = config_loader.load_tool_config("tool-searching-serper")

# 获取 Skill 注入器
injector = config_loader.get_skill_injector()

# 获取 Inline Skill 选择器
selector = config_loader.get_inline_skill_selector(cfg)
```

搜索顺序：
1. `project_dir/config/tool/` — 项目级（优先）
2. `框架 config/tool/` — 框架默认
