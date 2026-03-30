"""
Agent 工厂模块

提供一体化的 Agent 配置加载、组件创建和任务执行。
简化 Agent 的启动流程，支持从配置文件一键启动。
"""

import asyncio
import logging
import pathlib
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from omegaconf import DictConfig

from mem_deep_research_core.utils.external_loader import load_env_file, load_yaml_config

logger = logging.getLogger("mem_deep_research")


@dataclass
class AgentConfig:
    """Agent 配置数据类"""

    config_path: pathlib.Path
    logs_dir: pathlib.Path
    prompts_dir: pathlib.Path | None = None
    env_file: pathlib.Path | None = None

    # 解析后的配置
    cfg: DictConfig | None = field(default=None, repr=False)

    def __post_init__(self):
        # 确保路径是 Path 对象
        if isinstance(self.config_path, str):
            self.config_path = pathlib.Path(self.config_path)
        if isinstance(self.logs_dir, str):
            self.logs_dir = pathlib.Path(self.logs_dir)
        if self.prompts_dir and isinstance(self.prompts_dir, str):
            self.prompts_dir = pathlib.Path(self.prompts_dir)
        if self.env_file and isinstance(self.env_file, str):
            self.env_file = pathlib.Path(self.env_file)


@dataclass
class TaskResult:
    """任务执行结果"""

    task_id: str
    final_answer: str
    boxed_answer: str
    log_path: pathlib.Path
    status: str  # "completed", "failed", "interrupted"
    duration_seconds: float
    error: str | None = None
    error_type: str | None = None  # v0.3: "llm_error", "tool_error", "config_error", "timeout"


class AgentFactory:
    """
    Agent 工厂类

    提供一体化的配置加载和 Agent 启动功能。

    Usage:
        # 方式 1: 从配置文件路径创建
        factory = AgentFactory.from_config_file("config/agent.yaml", logs_dir="logs")
        result = await factory.run("你的任务")

        # 方式 2: 从项目目录创建
        factory = AgentFactory.from_project_dir("/path/to/project")
        result = await factory.run("你的任务", config_name="my_agent")

        # 方式 3: 从已有配置创建
        cfg = OmegaConf.load("config.yaml")
        factory = AgentFactory.from_config(cfg, logs_dir="logs")
        result = await factory.run("你的任务")
    """

    def __init__(
        self,
        agent_config: AgentConfig,
        auto_load_env: bool = True,
    ):
        """
        初始化 Agent 工厂

        Args:
            agent_config: Agent 配置
            auto_load_env: 是否自动加载 .env 文件
        """
        self.agent_config = agent_config

        # 确保日志目录存在
        self.agent_config.logs_dir.mkdir(parents=True, exist_ok=True)

        # 加载环境变量
        if auto_load_env and self.agent_config.env_file:
            load_env_file(self.agent_config.env_file)

        # 组件缓存
        self._main_agent_tool_manager = None
        self._sub_agent_tool_managers = None
        self._output_formatter = None
        self._tool_definitions = None
        self._initialized = False

    @classmethod
    def from_config_file(
        cls,
        config_path: str | pathlib.Path,
        logs_dir: str | pathlib.Path = "logs",
        prompts_dir: str | pathlib.Path | None = None,
        env_file: str | pathlib.Path | None = None,
    ) -> "AgentFactory":
        """
        从配置文件路径创建 AgentFactory

        Args:
            config_path: 配置文件路径
            logs_dir: 日志目录
            prompts_dir: 提示词模板目录（可选）
            env_file: 环境变量文件路径（可选）
        """
        config_path = pathlib.Path(config_path)
        logs_dir = pathlib.Path(logs_dir)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # 加载配置
        cfg = load_yaml_config(config_path)

        agent_config = AgentConfig(
            config_path=config_path,
            logs_dir=logs_dir,
            prompts_dir=pathlib.Path(prompts_dir) if prompts_dir else None,
            env_file=pathlib.Path(env_file) if env_file else None,
            cfg=cfg,
        )

        return cls(agent_config)

    @classmethod
    def from_project_dir(
        cls,
        project_dir: str | pathlib.Path,
        config_name: str = "agent",
    ) -> "AgentFactory":
        """
        从项目目录创建 AgentFactory

        自动查找以下文件:
        - config/{config_name}.yaml
        - config/prompts/ (提示词目录)
        - logs/ (日志目录)
        - .env (环境变量文件)

        Args:
            project_dir: 项目目录路径
            config_name: 配置文件名（不含 .yaml 后缀）
        """
        project_dir = pathlib.Path(project_dir)

        config_path = project_dir / "config" / f"{config_name}.yaml"
        logs_dir = project_dir / "logs"
        prompts_dir = project_dir / "config" / "prompts"
        env_file = project_dir / ".env"

        return cls.from_config_file(
            config_path=config_path,
            logs_dir=logs_dir,
            prompts_dir=prompts_dir if prompts_dir.exists() else None,
            env_file=env_file if env_file.exists() else None,
        )

    @classmethod
    def from_config(
        cls,
        cfg: DictConfig,
        logs_dir: str | pathlib.Path = "logs",
        prompts_dir: str | pathlib.Path | None = None,
    ) -> "AgentFactory":
        """
        从已有的 OmegaConf 配置创建 AgentFactory

        Args:
            cfg: OmegaConf 配置对象
            logs_dir: 日志目录
            prompts_dir: 提示词模板目录（可选）
        """
        logs_dir = pathlib.Path(logs_dir)

        agent_config = AgentConfig(
            config_path=pathlib.Path("dynamic_config"),  # 标记为动态配置
            logs_dir=logs_dir,
            prompts_dir=pathlib.Path(prompts_dir) if prompts_dir else None,
            cfg=cfg,
        )

        return cls(agent_config, auto_load_env=False)

    async def initialize(self) -> None:
        """
        初始化 Agent 组件

        包括创建 ToolManager、加载工具定义等。
        """
        if self._initialized:
            return

        from mem_deep_research_core.core.pipeline import create_pipeline_components

        cfg = self.agent_config.cfg
        logs_dir = str(self.agent_config.logs_dir)

        # 创建 pipeline 组件
        (
            self._main_agent_tool_manager,
            self._sub_agent_tool_managers,
            self._output_formatter,
        ) = create_pipeline_components(cfg, logs_dir)

        # 获取工具定义
        logger.info("Loading tool definitions...")
        self._tool_definitions = await self._main_agent_tool_manager.get_all_tool_definitions()
        logger.info(f"Loaded {len(self._tool_definitions)} tool servers")

        self._initialized = True

    def _generate_task_id(self) -> str:
        """生成任务 ID"""
        return f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    async def run(
        self,
        task: str,
        task_id: str | None = None,
        task_file: str | None = None,
        context: dict[str, Any] | None = None,
        stream_queue: Any | None = None,
        history: list[dict[str, Any]] | None = None,
        on_progress: Callable[[str, Any], None] | None = None,
        resume_from: dict | None = None,
    ) -> TaskResult:
        """
        执行任务

        Args:
            task: 任务描述
            task_id: 任务 ID（可选，自动生成）
            task_file: 关联的文件路径（可选）
            context: 用户上下文（可选）
            stream_queue: 流式输出队列（可选）
            history: 对话历史（可选）
            on_progress: 进度回调函数（可选）

        Returns:
            TaskResult: 任务执行结果
        """
        from mem_deep_research_core.core.pipeline import execute_task_pipeline

        # 确保已初始化
        await self.initialize()

        # 生成任务 ID
        if task_id is None:
            task_id = self._generate_task_id()

        log_path = self.agent_config.logs_dir / f"{task_id}.json"

        start_time = datetime.now()

        if on_progress:
            on_progress("started", {"task_id": task_id, "task": task})

        try:
            final_answer, boxed_answer, log_file = await execute_task_pipeline(
                cfg=self.agent_config.cfg,
                task_name="agent_task",
                task_id=task_id,
                task_description=task,
                task_file_name=task_file,
                main_agent_tool_manager=self._main_agent_tool_manager,
                sub_agent_tool_managers=self._sub_agent_tool_managers,
                output_formatter=self._output_formatter,
                log_path=log_path,
                tool_definitions=self._tool_definitions,
                stream_queue=stream_queue,
                history=history,
                context=context,
                resume_from=resume_from,
            )

            duration = (datetime.now() - start_time).total_seconds()

            result = TaskResult(
                task_id=task_id,
                final_answer=final_answer,
                boxed_answer=boxed_answer,
                log_path=log_file,
                status="completed",
                duration_seconds=duration,
            )

            if on_progress:
                on_progress("completed", result)

            return result

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            error_str = str(e)

            # Classify error type
            error_type = "unknown"
            e_lower = error_str.lower()
            if "timeout" in e_lower:
                error_type = "timeout"
            elif "context limit" in e_lower or "context length" in e_lower:
                error_type = "context_limit"
            elif isinstance(e, ValueError) or "config" in e_lower:
                error_type = "config_error"
            else:
                error_type = "llm_error"

            result = TaskResult(
                task_id=task_id,
                final_answer=f"Error: {error_str}",
                boxed_answer="",
                log_path=log_path,
                status="failed",
                duration_seconds=duration,
                error=error_str,
                error_type=error_type,
            )

            if on_progress:
                on_progress("failed", result)

            return result

    async def close(self) -> None:
        """Clean up all resources — call on shutdown."""
        if self._main_agent_tool_manager:
            await self._main_agent_tool_manager.close_sessions()
        if self._sub_agent_tool_managers:
            for tm in self._sub_agent_tool_managers.values():
                await tm.close_sessions()
        self._initialized = False
        logger.info("AgentFactory resources closed")

    async def run_batch(
        self,
        tasks: list[str],
        parallel: bool = False,
        max_concurrent: int = 3,
        on_task_complete: Callable[[int, TaskResult], None] | None = None,
    ) -> list[TaskResult]:
        """
        批量执行任务

        Args:
            tasks: 任务列表
            parallel: 是否并行执行
            max_concurrent: 最大并发数（仅 parallel=True 时有效）
            on_task_complete: 单个任务完成回调

        Returns:
            任务结果列表
        """
        results = []

        if parallel:
            # 使用 semaphore 限制并发数
            semaphore = asyncio.Semaphore(max_concurrent)

            async def run_with_semaphore(idx: int, task: str):
                async with semaphore:
                    result = await self.run(task)
                    if on_task_complete:
                        on_task_complete(idx, result)
                    return result

            tasks_coro = [run_with_semaphore(i, t) for i, t in enumerate(tasks)]
            results = await asyncio.gather(*tasks_coro)
        else:
            for idx, task in enumerate(tasks):
                result = await self.run(task)
                results.append(result)
                if on_task_complete:
                    on_task_complete(idx, result)

        return results


# 便捷函数
async def run_agent(
    task: str, config_path: str | pathlib.Path, logs_dir: str | pathlib.Path = "logs", **kwargs
) -> TaskResult:
    """
    便捷函数：一行代码运行 Agent

    Args:
        task: 任务描述
        config_path: 配置文件路径
        logs_dir: 日志目录
        **kwargs: 传递给 AgentFactory.run() 的其他参数

    Returns:
        TaskResult: 任务执行结果

    Example:
        result = await run_agent(
            "What is the capital of France?",
            config_path="config/agent.yaml"
        )
        print(result.final_answer)
    """
    factory = AgentFactory.from_config_file(config_path, logs_dir)
    try:
        return await factory.run(task, **kwargs)
    finally:
        await factory.close()


async def run_agent_from_project(
    task: str, project_dir: str | pathlib.Path, config_name: str = "agent", **kwargs
) -> TaskResult:
    """
    便捷函数：从项目目录运行 Agent

    Args:
        task: 任务描述
        project_dir: 项目目录路径
        config_name: 配置文件名
        **kwargs: 传递给 AgentFactory.run() 的其他参数

    Returns:
        TaskResult: 任务执行结果
    """
    factory = AgentFactory.from_project_dir(project_dir, config_name)
    try:
        return await factory.run(task, **kwargs)
    finally:
        await factory.close()
