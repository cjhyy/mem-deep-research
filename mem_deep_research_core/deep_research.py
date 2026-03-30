"""
DeepResearch - 主入口类

提供简洁的 API 用于配置和运行深度研究 Agent。

Usage:
    from mem_deep_research import DeepResearch

    # 方式 1: 从配置目录加载
    dr = DeepResearch.from_config_dir("./config")
    result = await dr.run("研究一下 AI Agent 的最新进展")

    # 方式 2: 从项目目录加载 (自动查找 config/ 目录)
    dr = DeepResearch.from_project("./my_research_project")
    result = await dr.run("你的研究任务")

    # 方式 3: 代码内配置
    dr = DeepResearch(
        llm_provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        api_key="your-api-key",
    )
    result = await dr.run("你的任务")

    # 方式 4: 同步运行
    result = dr.run_sync("你的任务")
"""

import asyncio
import logging
import pathlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig, OmegaConf

from mem_deep_research_core.utils.external_loader import load_env_file, load_yaml_config

logger = logging.getLogger("mem_deep_research")


@dataclass
class ResearchResult:
    """研究结果"""

    task_id: str
    answer: str
    boxed_answer: str = ""
    status: str = "completed"  # completed, failed
    duration_seconds: float = 0.0
    log_path: pathlib.Path | None = None
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.status == "completed" and self.error is None


class DeepResearch:
    """
    深度研究 Agent

    提供简洁的接口用于运行深度研究任务。
    支持从配置文件加载或代码内配置。
    """

    def __init__(
        self,
        # LLM 配置
        llm_provider: str = "anthropic",
        model: str = "claude-3-5-sonnet-20241022",
        api_key: str | None = None,
        base_url: str | None = None,
        # Agent 配置
        max_turns: int = 20,
        max_tool_calls_per_turn: int = 10,
        temperature: float = 0.3,
        # 工具配置
        tools: list[str] | None = None,
        tool_blacklist: list | None = None,
        # 输出配置
        logs_dir: str | pathlib.Path = "logs",
        chinese_context: bool = False,
        # 拦截器
        interceptor_preset: str = "default",
        # 输入/输出处理
        hint_generation: bool = False,
        final_answer_extraction: bool = False,
        # 高级配置
        config: DictConfig | None = None,
    ):
        """
        初始化 DeepResearch

        Args:
            llm_provider: LLM 提供商 ("anthropic", "openai", "openrouter", "deepseek")
            model: 模型名称
            api_key: API 密钥 (也可通过环境变量设置)
            base_url: API 基础 URL (可选)
            max_turns: 最大对话轮数
            max_tool_calls_per_turn: 每轮最大工具调用数
            temperature: 采样温度
            tools: 工具列表，如 ["tool-searching-serper", "tool-scraping"]
            tool_blacklist: 工具黑名单
            logs_dir: 日志目录
            chinese_context: 是否使用中文上下文
            interceptor_preset: 消息拦截器预设 ("default", "verbose", "minimal", "debug")
            hint_generation: 是否启用输入提示生成
            final_answer_extraction: 是否启用最终答案提取
            config: 完整的 OmegaConf 配置 (覆盖其他参数)
        """
        self.logs_dir = pathlib.Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # 如果提供了完整配置，直接使用
        if config is not None:
            self._cfg = config
        else:
            # 构建配置
            self._cfg = self._build_config(
                llm_provider=llm_provider,
                model=model,
                api_key=api_key,
                base_url=base_url,
                max_turns=max_turns,
                max_tool_calls_per_turn=max_tool_calls_per_turn,
                temperature=temperature,
                tools=tools or ["tool-searching-serper"],
                tool_blacklist=tool_blacklist or [],
                chinese_context=chinese_context,
                interceptor_preset=interceptor_preset,
                hint_generation=hint_generation,
                final_answer_extraction=final_answer_extraction,
            )

        # 验证配置
        self._validate_config()

        # 延迟初始化的组件
        self._factory = None
        self._initialized = False

    @classmethod
    def from_config_dir(
        cls,
        config_dir: str | pathlib.Path,
        config_name: str = "agent",
        logs_dir: str | pathlib.Path | None = None,
        env_file: str | pathlib.Path | None = None,
    ) -> "DeepResearch":
        """
        从配置目录加载

        Args:
            config_dir: 配置目录路径
            config_name: 配置文件名 (不含 .yaml 后缀)
            logs_dir: 日志目录 (默认: config_dir/../logs)
            env_file: .env 文件路径 (可选，默认查找 config_dir/../.env)

        Returns:
            DeepResearch 实例
        """
        config_dir = pathlib.Path(config_dir)
        config_path = config_dir / f"{config_name}.yaml"

        # 加载 .env 文件
        if env_file is None:
            env_file = config_dir.parent / ".env"
        else:
            env_file = pathlib.Path(env_file)

        load_env_file(env_file)

        # 加载配置
        cfg = load_yaml_config(config_path)

        if logs_dir is None:
            logs_dir = config_dir.parent / "logs"

        return cls(config=cfg, logs_dir=logs_dir)

    @classmethod
    def from_project(
        cls,
        project_dir: str | pathlib.Path,
        config_name: str = "agent",
        logs_dir: str | pathlib.Path | None = None,
    ) -> "DeepResearch":
        """
        从项目目录加载

        自动查找:
        - {project_dir}/config/{config_name}.yaml
        - {project_dir}/config/tool/*.yaml (自定义工具)
        - {project_dir}/tools/*.py (工具实现)
        - {project_dir}/logs/ (默认日志目录，可通过 logs_dir 覆盖)

        Args:
            project_dir: 项目目录
            config_name: 配置文件名
            logs_dir: 日志目录 (默认: project_dir/logs/)

        Returns:
            DeepResearch 实例
        """
        project_dir = pathlib.Path(project_dir).resolve()
        config_dir = project_dir / "config"
        if logs_dir is None:
            logs_dir = project_dir / "logs"

        # 设置项目目录，用于加载项目级别的工具配置
        from mem_deep_research_core.utils.external_loader import external_loader

        external_loader.set_project_dir(project_dir)

        # 加载项目钩子
        from mem_deep_research_core.core.hooks import load_project_hooks

        load_project_hooks(str(project_dir))

        return cls.from_config_dir(config_dir, config_name, logs_dir)

    def _validate_config(self) -> None:
        """验证配置是否符合 schema，仅警告不阻断"""
        try:
            from mem_deep_research_core.config_schema import validate_agent_config

            config_dict = OmegaConf.to_container(self._cfg, resolve=False)
            validate_agent_config(config_dict)
        except Exception as e:
            logger.warning(f"Config validation warning: {e}")

    def _build_config(
        self,
        llm_provider: str,
        model: str,
        api_key: str | None,
        base_url: str | None,
        max_turns: int,
        max_tool_calls_per_turn: int,
        temperature: float,
        tools: list[str],
        tool_blacklist: list,
        chinese_context: bool,
        interceptor_preset: str,
        hint_generation: bool,
        final_answer_extraction: bool,
    ) -> DictConfig:
        """构建配置"""
        # 映射 provider 到 client class
        provider_map = {
            "anthropic": (
                "ClaudeAnthropicClient",
                "ANTHROPIC_API_KEY",
                "https://api.anthropic.com",
            ),
            "openai": ("GPTOpenAIClient", "OPENAI_API_KEY", "https://api.openai.com/v1"),
            "openrouter": (
                "ClaudeOpenRouterClient",
                "OPENROUTER_API_KEY",
                "https://openrouter.ai/api/v1",
            ),
            "deepseek": (
                "DeepSeekOpenRouterClient",
                "DEEPSEEK_API_KEY",
                "https://api.deepseek.com",
            ),
        }

        if llm_provider not in provider_map:
            raise ValueError(
                f"Unknown LLM provider: {llm_provider}. Supported: {list(provider_map.keys())}"
            )

        provider_class, env_key, default_base_url = provider_map[llm_provider]

        # 构建 LLM 配置
        llm_config = {
            "provider_class": provider_class,
            "model_name": model,
            "temperature": temperature,
            "top_p": 0.95,
            "max_tokens": 32000,
            "timeout": 1800,
            "retry_max_attempts": 5,
            "retry_strategy": "exponential",
            "retry_multiplier": 5,
            "retry_wait_seconds": 10,
            "enable_streaming": True,
            "disable_cache_control": False,
            "keep_tool_result": -1,
            "oai_tool_thinking": False,
        }

        # 设置 API key
        if api_key:
            if llm_provider == "anthropic":
                llm_config["anthropic_api_key"] = api_key
            elif llm_provider == "openrouter":
                llm_config["openrouter_api_key"] = api_key
            else:
                llm_config["api_key"] = api_key
        else:
            # 使用环境变量占位符
            if llm_provider == "anthropic":
                llm_config["anthropic_api_key"] = f"${{oc.env:{env_key}}}"
            elif llm_provider == "openrouter":
                llm_config["openrouter_api_key"] = f"${{oc.env:{env_key}}}"

        # 设置 base URL
        if base_url:
            if llm_provider == "openrouter":
                llm_config["openrouter_base_url"] = base_url
            else:
                llm_config["base_url"] = base_url

        config = {
            "main_agent": {
                "prompt": {
                    "agent_type": "main",
                    "tool_format": "xml",
                    "presets": ["research"],
                },
                "llm": llm_config,
                "tool_config": tools,
                "tool_blacklist": tool_blacklist,
                "max_turns": max_turns,
                "max_tool_calls_per_turn": max_tool_calls_per_turn,
                "keep_tool_result": -1,
                "deep_research": {
                    "enabled": True,
                    "reflection_interval": 3,
                    "require_explicit_planning": True,
                },
                "add_message_id": True,
                "chinese_context": chinese_context,
                "interceptor": {
                    "preset": interceptor_preset,
                },
                "input_process": {
                    "hint_generation": hint_generation,
                    "hint_llm_base_url": "",
                },
                "output_process": {
                    "final_answer_extraction": final_answer_extraction,
                    "final_answer_llm_base_url": "",
                    "final_answer_model": "",
                },
            },
            "sub_agents": None,
            "benchmark": {
                "name": "custom",
            },
            "output_dir": str(self.logs_dir),
        }

        return OmegaConf.create(config)

    async def _ensure_initialized(self) -> None:
        """确保已初始化"""
        if self._initialized:
            return

        from mem_deep_research_core.core.agent_factory import AgentFactory

        self._factory = AgentFactory.from_config(
            cfg=self._cfg,
            logs_dir=self.logs_dir,
        )
        await self._factory.initialize()
        self._initialized = True

    async def run(
        self,
        task: str,
        context: dict[str, Any] | None = None,
        on_progress: Callable[[str, Any], None] | None = None,
    ) -> ResearchResult:
        """
        运行研究任务

        Args:
            task: 任务描述
            context: 用户上下文 (可选)
            on_progress: 进度回调 (可选)

        Returns:
            ResearchResult: 研究结果
        """
        await self._ensure_initialized()

        result = await self._factory.run(
            task=task,
            context=context,
            on_progress=on_progress,
        )

        return ResearchResult(
            task_id=result.task_id,
            answer=result.final_answer,
            boxed_answer=result.boxed_answer,
            status=result.status,
            duration_seconds=result.duration_seconds,
            log_path=result.log_path,
            error=result.error,
        )

    def run_sync(
        self,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> ResearchResult:
        """
        同步运行研究任务

        Args:
            task: 任务描述
            context: 用户上下文 (可选)

        Returns:
            ResearchResult: 研究结果
        """
        return asyncio.run(self.run(task, context))

    async def run_batch(
        self,
        tasks: list[str],
        parallel: bool = False,
        max_concurrent: int = 3,
    ) -> list[ResearchResult]:
        """
        批量运行研究任务

        Args:
            tasks: 任务列表
            parallel: 是否并行执行
            max_concurrent: 最大并发数

        Returns:
            结果列表
        """
        await self._ensure_initialized()

        results = await self._factory.run_batch(
            tasks=tasks,
            parallel=parallel,
            max_concurrent=max_concurrent,
        )

        return [
            ResearchResult(
                task_id=r.task_id,
                answer=r.final_answer,
                boxed_answer=r.boxed_answer,
                status=r.status,
                duration_seconds=r.duration_seconds,
                log_path=r.log_path,
                error=r.error,
            )
            for r in results
        ]

    @property
    def config(self) -> DictConfig:
        """获取当前配置"""
        return self._cfg
