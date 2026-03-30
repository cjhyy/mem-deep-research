"""
配置验证 Schema

使用 Pydantic 定义配置结构和验证规则。
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class LLMConfig(BaseModel):
    """LLM 配置验证"""

    provider_class: str = Field(
        ...,
        description="LLM provider 类名",
        examples=["ClaudeOpenRouterClient", "ClaudeAnthropicClient", "GPTOpenAIClient"],
    )
    model_name: str = Field(..., description="模型名称")
    async_client: bool = Field(default=True, description="是否使用异步客户端")

    # Model parameters
    temperature: float = Field(default=0.3, ge=0.0, le=2.0, description="采样温度")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Top-p 采样")
    top_k: int = Field(default=-1, ge=-1, description="Top-k 采样，-1 表示禁用")
    min_p: float = Field(default=0.0, ge=0.0, le=1.0, description="Min-p 采样")
    max_tokens: int = Field(default=32000, ge=1, description="最大生成 token 数")

    # Timeout and retry
    timeout: int = Field(default=300, ge=1, description="请求超时时间（秒）")
    retry_max_attempts: int = Field(default=3, ge=1, description="最大重试次数")
    retry_strategy: Literal["exponential", "fixed"] = Field(
        default="exponential", description="重试策略"
    )
    retry_multiplier: int = Field(default=2, ge=1, description="重试间隔乘数")
    retry_wait_seconds: int = Field(default=5, ge=1, description="基础重试等待时间")

    # API configuration (optional, can come from env vars)
    openrouter_api_key: str | None = Field(default=None, description="OpenRouter API Key")
    openrouter_base_url: str | None = Field(default=None, description="OpenRouter Base URL")
    openrouter_provider: str | None = Field(default="", description="OpenRouter Provider 偏好")
    anthropic_api_key: str | None = Field(default=None, description="Anthropic API Key")
    api_key: str | None = Field(default=None, description="通用 API Key")
    base_url: str | None = Field(default=None, description="通用 Base URL")

    # Streaming and caching
    enable_streaming: bool = Field(default=True, description="是否启用流式输出")
    disable_cache_control: bool = Field(default=False, description="是否禁用缓存控制")

    # Tool handling
    oai_tool_thinking: bool = Field(default=False, description="OpenAI 工具思考模式")

    # Context management
    max_context_length: int = Field(
        default=-1, description="模型最大上下文长度(tokens)，-1 表示不限制"
    )

    @field_validator("provider_class")
    @classmethod
    def validate_provider_class(cls, v: str) -> str:
        """验证 provider_class 是有效的 Python 标识符"""
        if not v.isidentifier():
            raise ValueError(f"provider_class must be a valid Python identifier, got: {v}")
        return v

    class Config:
        extra = "allow"


class PromptConfig(BaseModel):
    """新版统一 Prompt 配置"""

    agent_type: Literal["main", "worker"] = Field(default="main", description="Agent 类型")
    tool_format: Literal["xml", "native"] = Field(default="xml", description="工具格式")
    presets: list[str] = Field(default_factory=list, description="预设模块列表")
    templates_dir: str | None = Field(default=None, description="自定义模板目录")
    custom_system_template: str | None = Field(default=None, description="自定义系统提示词模板")
    custom_summarize_template: str | None = Field(default=None, description="自定义总结模板")


class DeepResearchConfig(BaseModel):
    """Deep Research 协议配置"""

    enabled: bool = Field(default=False, description="是否启用深度研究模式")
    reflection_interval: int = Field(default=5, ge=1, description="反思间隔轮次")
    require_explicit_planning: bool = Field(default=True, description="是否需要显式规划")
    auto_planning: bool = Field(default=False, description="是否启用 LLM 自动任务分解")


class InputProcessConfig(BaseModel):
    """输入处理配置"""

    hint_generation: bool = Field(default=False, description="是否启用提示生成")
    hint_llm_base_url: str | None = Field(default=None, description="提示生成 LLM Base URL")


class OutputProcessConfig(BaseModel):
    """输出处理配置"""

    final_answer_extraction: bool = Field(default=False, description="是否提取最终答案")
    final_answer_llm_base_url: str | None = Field(default=None, description="答案提取 LLM Base URL")
    final_answer_model: str | None = Field(default=None, description="答案提取模型")


class SkillSelectionConfig(BaseModel):
    """Skill 选择配置"""

    enabled: bool = Field(default=True, description="是否启用 Skill 选择")

    # 选择方式
    method: Literal["rules", "llm", "inline"] = Field(
        default="rules",
        description=(
            "Skill 选择方式: "
            "rules=仅规则匹配, "
            "llm=额外 LLM 调用选择, "
            "inline=LLM 在回复中声明下一轮 skill（零额外开销）"
        ),
    )

    # LLM 选择配置（method=llm 时使用）
    model: str = Field(default="gpt-4o-mini", description="用于 Skill 选择的模型")
    fallback_to_rules: bool = Field(default=True, description="LLM 失败时是否降级到规则匹配")

    # Inline 渐进加载（method=inline 时使用）
    progressive: bool = Field(
        default=True,
        description=(
            "渐进加载模式（仅 method=inline 生效）: "
            "True=第一轮仅注入 catalog，LLM 按需声明后再加载完整内容; "
            "False=第一轮用规则匹配注入完整 skill 内容（传统模式）"
        ),
    )

    # 通用配置
    max_skills: int = Field(default=3, ge=0, le=10, description="最大选择 Skill 数量")


class MonitoringConfigSchema(BaseModel):
    """执行监控配置"""

    # 停滞检测
    stall_detection_threshold: float = Field(
        default=120.0, ge=0.0, description="停滞检测阈值（秒）"
    )
    stall_terminate_multiplier: float = Field(default=2.0, ge=1.0, description="停滞终止倍数")
    max_total_time: float = Field(default=600.0, ge=0.0, description="最大总运行时间（秒）")

    # 空响应检测
    max_consecutive_empty_turns: int = Field(default=3, ge=1, description="连续空响应终止阈值")

    # 响应循环检测
    enable_loop_detection: bool = Field(default=True, description="是否启用重复响应检测")
    loop_detection_text_length: int = Field(default=500, ge=100, description="循环检测文本截取长度")
    loop_escalation_terminate_threshold: int = Field(
        default=3, ge=1, description="响应循环终止阈值"
    )
    response_hash_window_size: int = Field(default=8, ge=2, description="滑动窗口大小")
    response_hash_repeat_threshold: int = Field(default=3, ge=2, description="窗口内 hash 重复阈值")

    # 工具循环
    max_tool_loop_retries: int = Field(default=2, ge=0, description="工具循环最大重试次数")

    # 温度提升（响应循环升级时使用）
    temperature_boost: float = Field(
        default=0.3, ge=0.0, le=1.0, description="循环检测时温度提升量"
    )
    temperature_boost_cap: float = Field(default=1.0, ge=0.0, le=2.0, description="温度提升上限")

    # 工具结果限制
    scrape_max_length: int = Field(default=20000, ge=1000, description="Scrape 结果最大长度")


class TodoTrackerConfig(BaseModel):
    """任务追踪配置"""
    enabled: bool = Field(default=False, description="是否启用任务追踪")


class ContextManagerConfig(BaseModel):
    """Context Manager 配置

    三级 Context 管理策略:
      Level 1 (Compact): token-aware 摘要替换，零 LLM 成本
      Level 2 (Summarize): LLM 压缩旧历史为一条摘要
      Level 3 (Emergency): 二分删除中间消息
    """

    enable_dedup: bool = Field(default=True, description="是否启用跨轮次 tool call 去重")
    enable_compact: bool = Field(default=True, description="是否启用 Level 1 摘要替换")

    # Level 1: 摘要替换
    compact_at_ratio: float = Field(
        default=0.6, ge=0.0, le=1.0, description="token 占比超过此值时触发 compact"
    )
    compact_keep_recent: int = Field(default=3, ge=1, description="至少保留最近 N 轮完整结果")

    # Level 2: LLM 压缩
    summarize_at_ratio: float = Field(
        default=0.8, ge=0.0, le=1.0, description="token 占比超过此值时触发 LLM 压缩"
    )

    # Dedup cache
    max_dedup_cache_size: int = Field(
        default=200, ge=1, description="Maximum entries in dedup cache"
    )

    # Result offloading
    result_offload_threshold: int = Field(
        default=5000, ge=0,
        description="工具结果超过此字符数时卸载到文件，0=禁用"
    )
    result_offload_dir: str = Field(default="", description="卸载文件目录，空=使用 output_dir")

    # Token 估算
    chars_per_token: float = Field(
        default=3.5, gt=0.0, description="无 tiktoken 时的 fallback 估算比例"
    )

    # 兼容旧配置
    mask_after_n_turns: int = Field(
        default=5, ge=1, description="[已废弃] 旧配置，被 compact_keep_recent 取代"
    )
    enable_masking: bool = Field(
        default=True, description="[已废弃] 旧配置，被 enable_compact 取代"
    )

    @model_validator(mode='after')
    def warn_deprecated(self) -> 'ContextManagerConfig':
        import logging
        _logger = logging.getLogger("mem_deep_research")
        if self.mask_after_n_turns != 5:  # non-default means user set it
            _logger.warning(
                "Config 'mask_after_n_turns' is deprecated, use 'compact_keep_recent' instead"
            )
        if not self.enable_masking:  # non-default means user set it
            _logger.warning(
                "Config 'enable_masking' is deprecated, use 'enable_compact' instead"
            )
        return self


class InterceptorConfig(BaseModel):
    """消息拦截器配置"""

    preset: str = Field(default="default", description="拦截器预设名称")

    class Config:
        extra = "allow"


class MemoryConfig(BaseModel):
    """记忆系统配置"""

    enabled: bool = Field(default=False, description="是否启用记忆系统")
    storage: str = Field(default="file", description="存储方式: file | sqlite | custom")
    storage_path: str = Field(default="memory/", description="存储路径")
    max_entries: int = Field(default=1000, ge=1, description="最大记忆条目数")


class MainAgentConfig(BaseModel):
    """主 Agent 配置"""

    # 新版 prompt 配置
    prompt: PromptConfig = Field(default_factory=PromptConfig, description="Prompt 配置")

    llm: LLMConfig = Field(..., description="LLM 配置")
    tool_config: list[str] = Field(default_factory=list, description="工具配置列表")
    tool_blacklist: list = Field(default_factory=list, description="工具黑名单")

    # Execution limits
    max_turns: int = Field(default=20, ge=1, description="最大对话轮次")
    max_tool_calls_per_turn: int = Field(default=10, ge=1, description="每轮最大工具调用数")
    keep_tool_result: int = Field(default=-1, description="保留工具结果数")
    execution_mode: str = Field(
        default="auto",
        description="执行模式: 'auto' 自动判断, 'flash' 单轮直接回答, 'standard' 多轮工具调用, 'deep' 多轮+反思+子agent"
    )
    max_concurrent_subagents: int = Field(default=3, ge=1, description="最大并行子 Agent 数")

    # Deep Research
    deep_research: DeepResearchConfig = Field(
        default_factory=DeepResearchConfig, description="Deep Research 配置"
    )

    # TodoTracker
    todo_tracker: TodoTrackerConfig = Field(
        default_factory=TodoTrackerConfig, description="任务追踪配置"
    )

    # Processing
    input_process: InputProcessConfig = Field(
        default_factory=InputProcessConfig, description="输入处理配置"
    )
    output_process: OutputProcessConfig = Field(
        default_factory=OutputProcessConfig, description="输出处理配置"
    )

    # Interceptor
    interceptor: InterceptorConfig = Field(
        default_factory=InterceptorConfig, description="消息拦截器配置"
    )

    # Skill Selection
    skill_selection: SkillSelectionConfig = Field(
        default_factory=SkillSelectionConfig, description="Skill LLM 选择配置"
    )

    # Context Manager
    context_manager: ContextManagerConfig = Field(
        default_factory=ContextManagerConfig, description="Context Manager 配置"
    )

    # Monitoring
    monitoring: MonitoringConfigSchema = Field(
        default_factory=MonitoringConfigSchema, description="执行监控配置"
    )

    # Memory
    memory: MemoryConfig = Field(
        default_factory=MemoryConfig, description="记忆系统配置"
    )

    # Language
    response_language: str = Field(
        default="auto",
        description="响应语言: 'auto' 从 query 自动检测, 或指定语言如 'Chinese', 'English', 'Japanese' 等"
    )
    add_message_id: bool = Field(default=True, description="是否添加消息 ID")
    chinese_context: bool = Field(default=False, description="[已废弃] 使用 response_language 代替。设为 true 等同 response_language='Chinese'")

    class Config:
        extra = "allow"


class BenchmarkConfig(BaseModel):
    """Benchmark 配置"""

    name: str = Field(default="custom", description="Benchmark 名称")


class AgentConfig(BaseModel):
    """完整的 Agent 配置"""

    main_agent: MainAgentConfig = Field(..., description="主 Agent 配置")
    sub_agents: dict[str, Any] | None = Field(default=None, description="子 Agent 配置")
    benchmark: BenchmarkConfig = Field(
        default_factory=BenchmarkConfig, description="Benchmark 配置"
    )
    output_dir: str = Field(default="logs/", description="输出目录")

    @field_validator("sub_agents")
    @classmethod
    def validate_sub_agents(cls, v):
        if v is None:
            return v
        for name, cfg in v.items():
            if not name.startswith("agent-"):
                raise ValueError(f"Sub-agent name must start with 'agent-': {name}")
            if not isinstance(cfg, dict):
                raise ValueError(f"Sub-agent '{name}' config must be a dict")
            if "llm" not in cfg and "max_turns" not in cfg:
                raise ValueError(f"Sub-agent '{name}' must have at least 'llm' or 'max_turns' config")
        return v

    class Config:
        extra = "allow"  # 允许额外字段（如 defaults, env 等 Hydra 特定字段）


class ToolConfig(BaseModel):
    """工具配置"""

    name: str = Field(..., description="工具名称")
    tool_command: str = Field(..., description="工具启动命令")
    args: list[str] = Field(default_factory=list, description="命令参数")
    env: dict[str, str] = Field(default_factory=dict, description="环境变量")

    @field_validator("tool_command")
    @classmethod
    def validate_tool_command(cls, v: str) -> str:
        """验证工具命令不为空"""
        if not v.strip():
            raise ValueError("tool_command cannot be empty")
        return v


def validate_agent_config(config_dict: dict[str, Any]) -> AgentConfig:
    """
    验证 Agent 配置字典。

    Args:
        config_dict: 配置字典（通常从 YAML 加载）

    Returns:
        验证后的 AgentConfig 对象

    Raises:
        pydantic.ValidationError: 配置验证失败
    """
    return AgentConfig.model_validate(config_dict)


def validate_tool_config(config_dict: dict[str, Any]) -> ToolConfig:
    """
    验证工具配置字典。

    Args:
        config_dict: 配置字典

    Returns:
        验证后的 ToolConfig 对象

    Raises:
        pydantic.ValidationError: 配置验证失败
    """
    return ToolConfig.model_validate(config_dict)
