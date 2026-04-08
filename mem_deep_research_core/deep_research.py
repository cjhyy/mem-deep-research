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
        model="claude-sonnet-4-20250514",
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

from mem_deep_research_core.core.agent_runtime import AgentRuntime
from mem_deep_research_core.utils.external_loader import load_env_file, load_yaml_config

logger = logging.getLogger("mem_deep_research")


def _classify_error(error_str: str | None) -> str | None:
    """Classify error string into error type."""
    if not error_str:
        return None
    e = error_str.lower()
    if "timeout" in e or "timed out" in e:
        return "timeout"
    if "context limit" in e or "context length" in e:
        return "context_limit"
    if "tool" in e and ("failed" in e or "error" in e):
        return "tool_error"
    if "config" in e or "validation" in e or "not found" in e:
        return "config_error"
    if "api" in e or "llm" in e or "authentication" in e or "rate limit" in e:
        return "llm_error"
    return "unknown"


# Provider registry — maps short names to (client_class, env_key, default_base_url)
PROVIDER_REGISTRY: dict[str, tuple[str, str, str]] = {
    "anthropic": ("ClaudeAnthropicClient", "ANTHROPIC_API_KEY", "https://api.anthropic.com"),
    "openai": ("GPTOpenAIClient", "OPENAI_API_KEY", "https://api.openai.com/v1"),
    "openrouter": (
        "ClaudeOpenRouterClient",
        "OPENROUTER_API_KEY",
        "https://openrouter.ai/api/v1",
    ),
    "deepseek": ("DeepSeekOpenRouterClient", "DEEPSEEK_API_KEY", "https://api.deepseek.com"),
}


@dataclass
class TaskResult:
    """研究结果"""

    task_id: str
    answer: str
    boxed_answer: str = ""
    status: str = "completed"  # completed, failed
    duration_seconds: float = 0.0
    log_path: pathlib.Path | None = None
    error: str | None = None

    # v0.3: Execution details
    turns: int = 0
    tool_calls: int = 0
    error_type: str | None = None  # "llm_error", "tool_error", "config_error", "timeout"
    perf_metrics: dict | None = None
    checkpoints: list | None = None  # Turn-level progress checkpoints

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
        llm_provider: str | None = None,
        model: str | None = None,
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
        response_language: str = "auto",
        chinese_context: bool = False,
        # 拦截器
        interceptor_preset: str = "default",
        # 输入/输出处理
        hint_generation: bool = False,
        final_answer_extraction: bool = False,
        # Execution mode
        execution_mode: str = "auto",
        # 高级配置
        config: DictConfig | None = None,
        # 运行时隔离
        runtime: AgentRuntime | None = None,
    ):
        """
        初始化 DeepResearch

        Args:
            llm_provider: LLM 提供商 ("anthropic", "openai", "openrouter", "deepseek")。
                为 None 时根据 api_key 前缀自动检测，无 key 时默认 "openrouter"
            model: 模型名称。为 None 时根据 provider 自动选择默认模型
            api_key: API 密钥 (也可通过环境变量设置)
            base_url: API 基础 URL (可选)
            max_turns: 最大对话轮数
            max_tool_calls_per_turn: 每轮最大工具调用数
            temperature: 采样温度
            tools: 工具列表，如 ["tool-searching-serper", "tool-scraping"]
            tool_blacklist: 工具黑名单
            logs_dir: 日志目录
            response_language: 响应语言 ("auto" 自动检测, 或 "Chinese", "English", "Japanese" 等)
            chinese_context: [已废弃] 使用 response_language 代替。设为 True 等同 response_language='Chinese'
            interceptor_preset: 消息拦截器预设 ("default", "verbose", "minimal", "debug")
            hint_generation: 是否启用输入提示生成
            final_answer_extraction: 是否启用最终答案提取
            config: 完整的 OmegaConf 配置 (覆盖其他参数)
        """
        self.logs_dir = pathlib.Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Auto-detect provider from API key prefix when not explicitly set
        if config is None and llm_provider is None:
            if api_key:
                llm_provider = self._detect_provider(api_key)
            else:
                llm_provider = "openrouter"  # default when no key given

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
                tools=tools or [],
                tool_blacklist=tool_blacklist or [],
                response_language=response_language,
                chinese_context=chinese_context,
                interceptor_preset=interceptor_preset,
                hint_generation=hint_generation,
                final_answer_extraction=final_answer_extraction,
                execution_mode=execution_mode,
            )

        # 验证配置
        self._validate_config()

        # 运行时隔离：每个实例持有独立的 hooks + config_loader
        self._runtime = runtime or AgentRuntime()

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
        runtime: AgentRuntime | None = None,
    ) -> "DeepResearch":
        """
        从配置目录加载

        Args:
            config_dir: 配置目录路径
            config_name: 配置文件名 (不含 .yaml 后缀)
            logs_dir: 日志目录 (默认: config_dir/../logs)
            env_file: .env 文件路径 (可选，默认查找 config_dir/../.env)
            runtime: AgentRuntime 实例 (可选)

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

        return cls(config=cfg, logs_dir=logs_dir, runtime=runtime)

    @classmethod
    def from_project(
        cls,
        project_dir: str | pathlib.Path,
        config_name: str = "agent",
        logs_dir: str | pathlib.Path | None = None,
        runtime: AgentRuntime | None = None,
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
            runtime: AgentRuntime 实例 (可选，多实例时传入独立 runtime)

        Returns:
            DeepResearch 实例
        """
        project_dir = pathlib.Path(project_dir).resolve()
        config_dir = project_dir / "config"
        if logs_dir is None:
            logs_dir = project_dir / "logs"

        # 创建独立 runtime（如果未提供）
        rt = runtime or AgentRuntime()

        # 设置项目目录到实例级 config_loader
        rt.set_project_dir(str(project_dir))

        # 加载项目钩子到实例级 hook_registry
        from mem_deep_research_core.core.hooks import load_project_hooks

        load_project_hooks(str(project_dir), hook_registry=rt.hooks)

        # 同时更新全局单例以保持向后兼容（单实例场景）
        from mem_deep_research_core.utils.external_loader import external_loader

        external_loader.set_project_dir(project_dir)

        return cls.from_config_dir(config_dir, config_name, logs_dir, runtime=rt)

    @staticmethod
    def _detect_provider(api_key: str) -> str:
        """Auto-detect LLM provider from API key prefix."""
        if api_key.startswith("sk-ant-"):
            return "anthropic"
        elif api_key.startswith("sk-or-"):
            return "openrouter"
        elif api_key.startswith("sk-"):
            return "openai"
        else:
            return "openrouter"  # default fallback

    def _validate_config(self) -> None:
        """验证配置是否符合 schema，严重错误阻断，轻微问题仅警告"""
        try:
            from mem_deep_research_core.config_schema import validate_agent_config

            config_dict = OmegaConf.to_container(self._cfg, resolve=False)
            validate_agent_config(config_dict)
        except (ValueError, TypeError) as e:
            # Schema validation errors — log as error but don't block (may have optional fields)
            logger.error(f"Config validation failed: {e}")
        except ImportError as e:
            logger.warning(f"Config validation skipped (schema not available): {e}")
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
        response_language: str,
        chinese_context: bool,
        interceptor_preset: str,
        hint_generation: bool,
        final_answer_extraction: bool,
        execution_mode: str = "auto",
    ) -> DictConfig:
        """构建配置"""
        # Default model per provider
        if model is None:
            default_models = {
                "anthropic": "claude-sonnet-4-20250514",
                "openrouter": "anthropic/claude-sonnet-4",
                "openai": "gpt-4o",
                "deepseek": "deepseek-chat",
            }
            model = default_models.get(llm_provider, "claude-sonnet-4-20250514")

        # 映射 provider 到 client class
        provider_map = PROVIDER_REGISTRY

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
            "top_p": 1.0,
            "max_tokens": 32000,
            "timeout": 300,
            "retry_max_attempts": 3,
            "retry_strategy": "exponential",
            "retry_multiplier": 2,
            "retry_wait_seconds": 5,
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
                "execution_mode": execution_mode,
                "task_engine": {
                    "enabled": True,
                    "reflection_interval": 3,
                    "require_explicit_planning": True,
                },
                "response_language": response_language,
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
            runtime=self._runtime,
        )
        await self._factory.initialize()
        self._initialized = True

    async def run(
        self,
        task: str,
        context: dict[str, Any] | None = None,
        on_progress: Callable[[str, Any], None] | None = None,
        stream_queue: Any | None = None,
    ) -> TaskResult:
        """
        运行研究任务

        Args:
            task: 任务描述
            context: 用户上下文 (可选)
            on_progress: 进度回调 (可选)
            stream_queue: 流式输出队列 (可选，asyncio.Queue)

        Returns:
            TaskResult: 研究结果
        """
        await self._ensure_initialized()

        result = await self._factory.run(
            task=task,
            context=context,
            on_progress=on_progress,
            stream_queue=stream_queue,
        )

        # Read perf metrics from log if available
        _perf = {}
        _turns = 0
        _tool_calls = 0
        _checkpoints = []
        if result.log_path and result.log_path.exists():
            try:
                import json

                _log_data = json.loads(result.log_path.read_text())
                _perf = _log_data.get("perf_metrics", {})
                _turns = _perf.get("main_loop_turns", {}).get("value", 0)
                _tool_calls = _perf.get("main_loop_tool_calls", {}).get("value", 0)
                _checkpoints = _log_data.get("checkpoints", [])
            except Exception as e:
                logger.debug(f"[DeepResearch] Failed to parse log metrics: {e}")

        return TaskResult(
            task_id=result.task_id,
            answer=result.final_answer,
            boxed_answer=result.boxed_answer,
            status=result.status,
            duration_seconds=result.duration_seconds,
            log_path=result.log_path,
            error=result.error,
            turns=_turns,
            tool_calls=_tool_calls,
            error_type=_classify_error(result.error) if result.error else None,
            perf_metrics=_perf if _perf else None,
            checkpoints=_checkpoints if _checkpoints else None,
        )

    async def resume(
        self,
        log_path: str | pathlib.Path,
        context: dict[str, Any] | None = None,
        stream_queue: Any | None = None,
    ) -> TaskResult:
        """从之前中断的任务恢复执行

        Args:
            log_path: 之前任务的日志文件路径
            context: 用户上下文 (可选)
            stream_queue: 流式输出队列 (可选)

        Returns:
            TaskResult: 恢复执行的结果
        """
        from mem_deep_research_core.mem_deep_research_logging.task_tracer import TaskTracer

        tracer = TaskTracer.load_from_log(log_path)
        state = tracer.get_resumable_state()

        task_description = state.get("task_description", "")
        if not task_description:
            raise ValueError(f"Cannot resume: no task_description found in log {log_path}")

        await self._ensure_initialized()

        result = await self._factory.run(
            task=task_description,
            context=context,
            stream_queue=stream_queue,
            resume_from=state,
        )

        # Reuse the same result building logic
        _perf = {}
        _turns = 0
        _tool_calls = 0
        _checkpoints = []
        if result.log_path and result.log_path.exists():
            try:
                import json

                _log_data = json.loads(result.log_path.read_text())
                _perf = _log_data.get("perf_metrics", {})
                _turns = _perf.get("main_loop_turns", {}).get("value", 0)
                _tool_calls = _perf.get("main_loop_tool_calls", {}).get("value", 0)
                _checkpoints = _log_data.get("checkpoints", [])
            except Exception as e:
                logger.debug(f"[DeepResearch] Failed to parse log metrics: {e}")

        return TaskResult(
            task_id=result.task_id,
            answer=result.final_answer,
            boxed_answer=result.boxed_answer,
            status=result.status,
            duration_seconds=result.duration_seconds,
            log_path=result.log_path,
            error=result.error,
            turns=_turns,
            tool_calls=_tool_calls,
            error_type=_classify_error(result.error) if result.error else None,
            perf_metrics=_perf if _perf else None,
            checkpoints=_checkpoints if _checkpoints else None,
        )

    def resume_sync(
        self,
        log_path: str | pathlib.Path,
        context: dict[str, Any] | None = None,
    ) -> TaskResult:
        """同步恢复执行"""

        async def _resume_and_close():
            try:
                return await self.resume(log_path, context)
            finally:
                await self.close()

        self._initialized = False
        return asyncio.run(_resume_and_close())

    def run_sync(
        self,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> TaskResult:
        """
        同步运行研究任务

        Args:
            task: 任务描述
            context: 用户上下文 (可选)

        Returns:
            TaskResult: 研究结果
        """

        async def _run_and_close():
            try:
                return await self.run(task, context)
            finally:
                await self.close()

        self._initialized = False  # asyncio.run() 创建新循环，旧资源失效
        return asyncio.run(_run_and_close())

    async def run_batch(
        self,
        tasks: list[str],
        parallel: bool = False,
        max_concurrent: int = 3,
    ) -> list[TaskResult]:
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

        research_results = []
        for r in results:
            _perf = {}
            _turns = 0
            _tool_calls = 0
            _checkpoints = []
            if r.log_path and r.log_path.exists():
                try:
                    import json

                    _log_data = json.loads(r.log_path.read_text())
                    _perf = _log_data.get("perf_metrics", {})
                    _turns = _perf.get("main_loop_turns", {}).get("value", 0)
                    _tool_calls = _perf.get("main_loop_tool_calls", {}).get("value", 0)
                    _checkpoints = _log_data.get("checkpoints", [])
                except Exception as e:
                    logger.debug(f"[DeepResearch] Failed to parse log metrics: {e}")
            research_results.append(
                TaskResult(
                    task_id=r.task_id,
                    answer=r.final_answer,
                    boxed_answer=r.boxed_answer,
                    status=r.status,
                    duration_seconds=r.duration_seconds,
                    log_path=r.log_path,
                    error=r.error,
                    turns=_turns,
                    tool_calls=_tool_calls,
                    error_type=_classify_error(r.error) if r.error else None,
                    perf_metrics=_perf if _perf else None,
                    checkpoints=_checkpoints if _checkpoints else None,
                )
            )
        return research_results

    @staticmethod
    def supported_providers() -> list[str]:
        """Return list of supported LLM provider names."""
        return list(PROVIDER_REGISTRY.keys())

    async def list_tools(self) -> list[dict]:
        """List all available tools and their descriptions.

        Returns:
            List of dicts with 'server', 'name', 'description' keys.
        """
        await self._ensure_initialized()
        tools = []
        if self._factory and self._factory._tool_definitions:
            for server in self._factory._tool_definitions:
                server_name = server.get("name", "")
                for tool in server.get("tools", []):
                    if "error" in tool:
                        continue
                    tools.append(
                        {
                            "server": server_name,
                            "name": tool.get("name", ""),
                            "description": tool.get("description", ""),
                        }
                    )
        return tools

    async def validate(self) -> dict:
        """Validate configuration and connectivity.

        Returns:
            Dict with 'valid' (bool), 'errors' (list[str]), 'warnings' (list[str])
        """
        errors = []
        warnings = []

        # Validate config
        try:
            from mem_deep_research_core.config_schema import validate_agent_config

            config_dict = OmegaConf.to_container(self._cfg, resolve=False)
            validate_agent_config(config_dict)
        except Exception as e:
            warnings.append(f"Config validation: {str(e)[:200]}")

        # Check initialization
        try:
            await self._ensure_initialized()
        except Exception as e:
            errors.append(f"Initialization failed: {str(e)[:200]}")
            return {"valid": False, "errors": errors, "warnings": warnings}

        # Check tools
        tools = await self.list_tools()
        if not tools:
            warnings.append("No tools configured")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "tools_count": len(tools),
        }

    async def close(self) -> None:
        """Release all resources (MCP sessions, LLM clients)."""
        if self._factory:
            await self._factory.close()
            self._initialized = False

    def __del__(self):
        if hasattr(self, "_factory") and self._factory is not None and self._initialized:
            logger.debug("DeepResearch was garbage collected without calling close()")

    async def __aenter__(self):
        """Support async with DeepResearch(...) as dr: ..."""
        await self._ensure_initialized()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        return False

    @property
    def config(self) -> DictConfig:
        """获取当前配置"""
        return self._cfg
