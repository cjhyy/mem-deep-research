"""
AgentRuntime — 运行时上下文容器

每个 DeepResearch 实例拥有独立的 AgentRuntime，持有 hooks、config_loader 等
原本的全局单例引用，保证多实例并发运行时互不污染。

设计原则:
1. AgentRuntime 是一个轻量容器，不包含业务逻辑
2. 所有原全局单例通过 AgentRuntime 注入
3. 向后兼容：不提供 runtime 时，自动使用全局默认实例
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("mem_deep_research")


class AgentRuntime:
    """
    运行时上下文容器

    持有原本的全局单例（hooks、config_loader），保证多实例隔离。

    Usage:
        # 显式创建（推荐，多实例安全）
        runtime = AgentRuntime()
        dr = DeepResearch(runtime=runtime)

        # 不传 runtime → 使用全局默认（单实例向后兼容）
        dr = DeepResearch()
    """

    def __init__(
        self,
        hooks: Any | None = None,
        config_loader: Any | None = None,
    ):
        """
        Args:
            hooks: HookRegistry 实例。None 时创建新的独立实例。
            config_loader: ConfigLoader 实例。None 时创建新的独立实例。
        """
        # 延迟导入避免循环依赖
        if hooks is None:
            from mem_deep_research_core.core.hooks import HookRegistry

            hooks = HookRegistry()
        if config_loader is None:
            from mem_deep_research_core.utils.external_loader import ConfigLoader

            config_loader = ConfigLoader()

        self._hooks = hooks
        self._config_loader = config_loader

    @property
    def hooks(self):
        """HookRegistry 实例"""
        return self._hooks

    @property
    def config_loader(self):
        """ConfigLoader 实例"""
        return self._config_loader

    def load_project_hooks(self, project_dir: str) -> None:
        """
        从项目目录加载钩子到本实例的 HookRegistry

        Args:
            project_dir: 项目目录路径
        """
        import importlib.util
        import sys
        from pathlib import Path

        # 清除之前注册的项目钩子
        self._hooks.clear()

        hooks_file = Path(project_dir) / "hooks.py"
        if not hooks_file.exists():
            logger.debug(f"[AgentRuntime] No hooks.py found in {project_dir}")
            return

        try:
            # 临时将本实例的 hooks 注入到项目 hooks 模块可见的位置
            # 项目 hooks.py 通过 `from mem_deep_research_core.core.hooks import hooks` 注册
            # 为了让项目代码注册到本实例，我们临时替换全局 hooks
            from mem_deep_research_core.core import hooks as hooks_module

            original_hooks = hooks_module.hooks
            hooks_module.hooks = self._hooks
            try:
                spec = importlib.util.spec_from_file_location("project_hooks", hooks_file)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Cannot load hooks from {hooks_file}")

                module = importlib.util.module_from_spec(spec)
                sys.modules["project_hooks"] = module
                spec.loader.exec_module(module)

                logger.info(f"[AgentRuntime] Loaded project hooks from {hooks_file}")
                logger.info(f"[AgentRuntime] Registered hooks: {self._hooks.list_hooks()}")
            finally:
                # 恢复全局 hooks
                hooks_module.hooks = original_hooks
        except Exception as e:
            logger.warning(f"[AgentRuntime] Failed to load project hooks: {e}")

    def set_project_dir(self, project_dir: str | None) -> None:
        """设置项目目录到 config_loader"""
        self._config_loader.set_project_dir(project_dir)

    def setup_hook_defaults(self) -> None:
        """注册框架内置的默认钩子实现

        在 Orchestrator 初始化前调用，确保所有 set_default 写入本实例而非全局。
        集中管理所有框架默认钩子——各模块不再在模块级注册到全局单例。
        """
        # orchestrator: agent/turn 生命周期
        from mem_deep_research_core.core.orchestrator import (
            _default_on_agent_end,
            _default_on_agent_start,
            _default_on_turn_end,
            _default_on_turn_start,
        )

        self._hooks.set_default("on_agent_start", _default_on_agent_start)
        self._hooks.set_default("on_agent_end", _default_on_agent_end)
        self._hooks.set_default("on_turn_start", _default_on_turn_start)
        self._hooks.set_default("on_turn_end", _default_on_turn_end)

        # tool_executor: tool start/end
        from mem_deep_research_core.core.tool_executor import (
            _default_on_tool_end,
            _default_on_tool_start,
        )

        self._hooks.set_default("on_tool_start", _default_on_tool_start)
        self._hooks.set_default("on_tool_end", _default_on_tool_end)

        # tool_result_formatter: thinking/result formatting
        from mem_deep_research_core.core.tool_result_formatter import (
            _default_thinking_generate,
            _default_tool_result_format,
        )

        self._hooks.set_default("on_thinking_generate", _default_thinking_generate)
        self._hooks.set_default("on_tool_result_format", _default_tool_result_format)

        # message_interceptor: message intercept
        from mem_deep_research_core.core.message_interceptor import (
            _default_on_message_intercept,
        )

        self._hooks.set_default("on_message_intercept", _default_on_message_intercept)

        # tool/manager: env injection
        from mem_deep_research_core.tool.manager import _default_env_inject

        self._hooks.set_default("on_env_inject", _default_env_inject)


def get_global_runtime() -> AgentRuntime:
    """获取使用全局单例的 AgentRuntime（向后兼容）

    返回一个包装了全局 hooks 和全局 config_loader 的 runtime。
    仅用于单实例场景和向后兼容。
    """
    from mem_deep_research_core.core.hooks import hooks as global_hooks
    from mem_deep_research_core.utils.external_loader import config_loader as global_config_loader

    return AgentRuntime(hooks=global_hooks, config_loader=global_config_loader)
