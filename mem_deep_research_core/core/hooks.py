"""
框架生命周期钩子系统

提供可扩展的钩子机制，允许项目在关键生命周期点注入自定义逻辑。

设计原则:
1. 框架定义钩子接口和默认行为
2. 项目通过注册函数覆盖或扩展行为
3. 原逻辑作为参数传入，用户决定是否调用

使用方式:
    from mem_deep_research_core.core.hooks import hooks

    # 方式1: 完全覆盖原逻辑
    @hooks.register("on_tool_result_format")
    def my_formatter(ctx, original_fn):
        # 不调用 original_fn，完全自定义
        return "my custom result"

    # 方式2: 扩展原逻辑
    @hooks.register("on_tool_result_format")
    def my_formatter(ctx, original_fn):
        result = original_fn(ctx)  # 先执行原逻辑
        return result + "\\n[enhanced]"  # 再增强

    # 方式3: 条件执行
    @hooks.register("on_tool_result_format")
    def my_formatter(ctx, original_fn):
        if ctx.tool_name == "my_tool":
            return "custom handling"
        return original_fn(ctx)  # 其他工具用原逻辑
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

logger = logging.getLogger("mem_deep_research")

T = TypeVar("T")


@dataclass
class HookContext:
    """钩子上下文 - 传递给钩子函数的参数"""

    # 通用字段
    hook_name: str

    # Agent 相关
    query: str | None = None
    result: Any | None = None
    context: dict[str, Any] | None = None

    # 工具相关
    tool_name: str | None = None
    server_name: str | None = None
    arguments: dict[str, Any] | None = None
    tool_result: dict[str, Any] | None = None
    duration_ms: int | None = None

    # 轮次相关
    turn_number: int | None = None
    tool_calls_count: int | None = None

    # 环境注入相关
    server_params: Any | None = None

    # 格式化相关
    formatted_result: str | None = None

    # 额外数据
    extra: dict[str, Any] = field(default_factory=dict)


# 钩子函数类型: (context, original_fn) -> result
HookFn = Callable[[HookContext, Callable[[HookContext], T]], T]


class HookRegistry:
    """
    钩子注册表

    管理所有生命周期钩子的注册和调用。
    """

    # 支持的钩子列表
    SUPPORTED_HOOKS = [
        # Agent 生命周期
        "on_agent_start",  # Agent 开始执行
        "on_agent_end",  # Agent 执行完成
        # 轮次生命周期
        "on_turn_start",  # 每轮对话开始
        "on_turn_end",  # 每轮对话结束
        # 工具生命周期
        "on_tool_start",  # 工具调用开始 (可修改参数)
        "on_tool_end",  # 工具调用完成 (可修改结果)
        # 格式化钩子
        "on_tool_result_format",  # 工具结果格式化
        "on_thinking_generate",  # thinking 描述生成
        # 环境注入钩子
        "on_env_inject",  # MCP 子进程环境变量注入
        # 消息处理钩子
        "on_message_intercept",  # 消息拦截处理
    ]

    def __init__(self):
        self._hooks: dict[str, list[HookFn]] = {name: [] for name in self.SUPPORTED_HOOKS}
        self._default_fns: dict[str, Callable] = {}

    def register(self, hook_name: str, priority: int = 0):
        """
        注册钩子的装饰器

        Args:
            hook_name: 钩子名称
            priority: 优先级 (数字越大越先执行)

        Example:
            @hooks.register("on_tool_end")
            def my_hook(ctx, original_fn):
                result = original_fn(ctx)
                return enhanced_result
        """
        if hook_name not in self.SUPPORTED_HOOKS:
            raise ValueError(f"Unknown hook: {hook_name}. Supported: {self.SUPPORTED_HOOKS}")

        def decorator(fn: HookFn) -> HookFn:
            self._hooks[hook_name].append((priority, fn))
            # 按优先级排序 (高优先级在前)
            self._hooks[hook_name].sort(key=lambda x: -x[0])
            logger.debug(f"[Hooks] Registered hook '{hook_name}' with priority {priority}")
            return fn

        return decorator

    def register_fn(self, hook_name: str, fn: HookFn, priority: int = 0) -> None:
        """
        直接注册钩子函数 (非装饰器方式)

        Args:
            hook_name: 钩子名称
            fn: 钩子函数
            priority: 优先级
        """
        if hook_name not in self.SUPPORTED_HOOKS:
            raise ValueError(f"Unknown hook: {hook_name}. Supported: {self.SUPPORTED_HOOKS}")

        self._hooks[hook_name].append((priority, fn))
        self._hooks[hook_name].sort(key=lambda x: -x[0])
        logger.debug(f"[Hooks] Registered hook '{hook_name}' with priority {priority}")

    def set_default(self, hook_name: str, default_fn: Callable[[HookContext], Any]) -> None:
        """
        设置钩子的默认实现

        框架内部使用，设置原逻辑函数。

        Args:
            hook_name: 钩子名称
            default_fn: 默认实现函数
        """
        self._default_fns[hook_name] = default_fn

    def call(self, hook_name: str, ctx: HookContext) -> Any:
        """
        调用钩子

        按优先级链式调用所有注册的钩子，每个钩子都可以选择是否调用下一个。

        Args:
            hook_name: 钩子名称
            ctx: 钩子上下文

        Returns:
            钩子执行结果
        """
        if hook_name not in self.SUPPORTED_HOOKS:
            raise ValueError(f"Unknown hook: {hook_name}")

        ctx.hook_name = hook_name
        registered = self._hooks.get(hook_name, [])
        default_fn = self._default_fns.get(hook_name, lambda c: None)

        if not registered:
            # 没有注册钩子，直接执行默认逻辑
            return default_fn(ctx)

        # 构建调用链
        def build_chain(remaining_hooks, final_fn):
            if not remaining_hooks:
                return final_fn

            _, current_hook = remaining_hooks[0]
            next_fn = build_chain(remaining_hooks[1:], final_fn)

            def chain_fn(c: HookContext):
                return current_hook(c, next_fn)

            return chain_fn

        chain = build_chain(registered, default_fn)
        return chain(ctx)

    def has_hooks(self, hook_name: str) -> bool:
        """检查是否有注册的钩子"""
        return bool(self._hooks.get(hook_name))

    def clear(self, hook_name: str | None = None) -> None:
        """
        清除钩子注册

        Args:
            hook_name: 指定钩子名称，None 表示清除所有
        """
        if hook_name:
            if hook_name in self._hooks:
                self._hooks[hook_name] = []
        else:
            for name in self._hooks:
                self._hooks[name] = []

    def list_hooks(self) -> dict[str, int]:
        """列出所有钩子及其注册数量"""
        return {name: len(hooks) for name, hooks in self._hooks.items()}


# 全局钩子注册表
hooks = HookRegistry()


# ============================================================
# 便捷装饰器
# ============================================================


def on_agent_start(priority: int = 0):
    """Agent 开始执行钩子"""
    return hooks.register("on_agent_start", priority)


def on_agent_end(priority: int = 0):
    """Agent 执行完成钩子"""
    return hooks.register("on_agent_end", priority)


def on_turn_start(priority: int = 0):
    """轮次开始钩子"""
    return hooks.register("on_turn_start", priority)


def on_turn_end(priority: int = 0):
    """轮次结束钩子"""
    return hooks.register("on_turn_end", priority)


def on_tool_start(priority: int = 0):
    """工具调用开始钩子 - 可修改参数"""
    return hooks.register("on_tool_start", priority)


def on_tool_end(priority: int = 0):
    """工具调用完成钩子 - 可修改结果"""
    return hooks.register("on_tool_end", priority)


def on_tool_result_format(priority: int = 0):
    """工具结果格式化钩子"""
    return hooks.register("on_tool_result_format", priority)


def on_thinking_generate(priority: int = 0):
    """thinking 描述生成钩子"""
    return hooks.register("on_thinking_generate", priority)


def on_env_inject(priority: int = 0):
    """环境变量注入钩子"""
    return hooks.register("on_env_inject", priority)


# ============================================================
# 项目钩子加载
# ============================================================


def load_project_hooks(project_dir: str) -> None:
    """
    从项目目录加载钩子定义

    查找 {project_dir}/hooks.py 文件并执行，
    该文件应该使用 @hooks.register() 装饰器注册钩子。

    Args:
        project_dir: 项目目录路径
    """
    import importlib.util
    import sys
    from pathlib import Path

    hooks_file = Path(project_dir) / "hooks.py"
    if not hooks_file.exists():
        logger.debug(f"[Hooks] No hooks.py found in {project_dir}")
        return

    try:
        spec = importlib.util.spec_from_file_location("project_hooks", hooks_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load hooks from {hooks_file}")

        module = importlib.util.module_from_spec(spec)
        sys.modules["project_hooks"] = module
        spec.loader.exec_module(module)

        logger.info(f"[Hooks] Loaded project hooks from {hooks_file}")
        logger.info(f"[Hooks] Registered hooks: {hooks.list_hooks()}")
    except Exception as e:
        logger.warning(f"[Hooks] Failed to load project hooks: {e}")
