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

import bisect
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from mem_deep_research_core.exceptions import GuardrailError

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

    # 工具批次 (on_tool_filter: 去重后待执行的工具调用列表)
    tool_calls_batch: list | None = None

    # Context 压缩 (on_context_compact: "masking" | "summarize" | "emergency")
    compact_action: str | None = None

    # 额外数据 (观测性数据如 assistant_text, message_count, total_tool_calls 等通过此字段传递)
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
        "on_tool_filter",  # 工具去重后、执行前 (可修改/重排/拦截工具调用列表)
        # Prompt 钩子
        "on_system_prompt_build",  # system prompt 生成后 (可修改 prompt)
        "on_summarize_prompt_build",  # summarize prompt 生成后 (可修改 prompt)
        "on_final_answer",  # 最终答案后处理（可修改 final answer 文本）
        # 格式化钩子
        "on_tool_result_format",  # 工具结果格式化
        "on_thinking_generate",  # thinking 描述生成
        # 环境注入钩子
        "on_env_inject",  # MCP 子进程环境变量注入
        # 消息处理钩子
        "on_message_intercept",  # 消息拦截处理
        # Guardrails (fail-fast validation)
        "on_before_llm_call",  # 验证 LLM 输入，可 raise GuardrailError 阻止调用
        "on_after_llm_call",  # 验证 LLM 输出，可 raise GuardrailError 拒绝结果
        # Context 管理
        "on_context_compact",  # context 压缩时 (可标记保护消息、观察压缩行为)
        # 反思
        "on_reflection_build",  # 反思 prompt 生成 (可修改反思内容)
        # 路由
        "on_route_classify",  # 任务复杂度分类 (可覆盖返回 "quick"/"standard"/"deep" 或 dict)
        "on_route_apply",  # 路由结果应用 (可修改 mode + reasoning_effort)
        # 存储
        "on_result_offload",  # 大结果卸载 (可覆盖存储后端: 文件/内存/S3/Redis)
        "on_result_restore",  # 卸载结果恢复 (可覆盖读取后端)
        "on_offload_evidence_prep",  # offload evidence sidecar prompt 构建 (可追加 tool-specific 指导)
        # 输入编译
        "on_query_compile",  # 用户 query 编译后 (可修改 query / 追加 attachments)
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
            bisect.insort(self._hooks[hook_name], (priority, fn), key=lambda x: -x[0])
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

        bisect.insort(self._hooks[hook_name], (priority, fn), key=lambda x: -x[0])
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

        # 构建调用链（每个用户钩子包裹 try-except，避免一个坏钩子终止整个运行）
        def build_chain(remaining_hooks, final_fn):
            if not remaining_hooks:
                return final_fn

            _, current_hook = remaining_hooks[0]
            next_fn = build_chain(remaining_hooks[1:], final_fn)

            def chain_fn(c: HookContext):
                try:
                    return current_hook(c, next_fn)
                except (GuardrailError, KeyboardInterrupt, SystemExit):
                    raise  # 护栏异常和系统异常不能被吞掉
                except Exception as e:
                    hook_label = getattr(current_hook, "__name__", repr(current_hook))
                    logger.error(
                        f"[Hooks] Hook '{hook_name}' ({hook_label}) raised {type(e).__name__}: {e}. "
                        f"Falling through to next handler.",
                        exc_info=True,
                    )
                    # Fall through to next hook / default
                    return next_fn(c)

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

    def clear_all(self) -> None:
        """Clear all hooks AND defaults — full reset for new project/task."""
        self.clear()
        self._default_fns.clear()

    def list_hooks(self) -> dict[str, int]:
        """列出所有钩子及其注册数量"""
        return {name: len(hook_list) for name, hook_list in self._hooks.items()}


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


def on_before_llm_call(priority: int = 0):
    """LLM 调用前验证钩子 — raise GuardrailError 可阻止调用"""
    return hooks.register("on_before_llm_call", priority)


def on_after_llm_call(priority: int = 0):
    """LLM 调用后验证钩子 — raise GuardrailError 可拒绝结果"""
    return hooks.register("on_after_llm_call", priority)


def on_final_answer(priority: int = 0):
    """最终答案后处理钩子 — 可修改 final answer 文本"""
    return hooks.register("on_final_answer", priority)


def on_tool_filter(priority: int = 0):
    """工具去重后、执行前钩子 — 可修改/重排/拦截工具调用列表"""
    return hooks.register("on_tool_filter", priority)


def on_context_compact(priority: int = 0):
    """Context 压缩钩子 — 观察或干预压缩行为"""
    return hooks.register("on_context_compact", priority)


def on_reflection_build(priority: int = 0):
    """反思 prompt 生成钩子 — 可修改反思内容"""
    return hooks.register("on_reflection_build", priority)


def on_system_prompt_build(priority: int = 0):
    """System prompt 生成后钩子 — 可修改 system prompt"""
    return hooks.register("on_system_prompt_build", priority)


def on_summarize_prompt_build(priority: int = 0):
    """Summarize prompt 生成后钩子 — 可修改 summarize prompt"""
    return hooks.register("on_summarize_prompt_build", priority)


def on_message_intercept(priority: int = 0):
    """消息拦截钩子"""
    return hooks.register("on_message_intercept", priority)


def on_route_classify(priority: int = 0):
    """任务复杂度分类钩子 — 可覆盖路由决策"""
    return hooks.register("on_route_classify", priority)


def on_route_apply(priority: int = 0):
    """路由结果应用钩子 — 可修改 mode 和 reasoning_effort"""
    return hooks.register("on_route_apply", priority)


def on_result_offload(priority: int = 0):
    """大结果卸载钩子 — 可覆盖存储后端"""
    return hooks.register("on_result_offload", priority)


def on_result_restore(priority: int = 0):
    """卸载结果恢复钩子 — 可覆盖读取后端"""
    return hooks.register("on_result_restore", priority)


def on_offload_evidence_prep(priority: int = 0):
    """Offload evidence sidecar prompt 构建钩子"""
    return hooks.register("on_offload_evidence_prep", priority)


def on_query_compile(priority: int = 0):
    """用户 query 编译后钩子 — 可修改 query 或追加 attachments"""
    return hooks.register("on_query_compile", priority)


# ============================================================
# 项目钩子加载
# ============================================================


import threading as _threading

_load_hooks_lock = _threading.Lock()


def load_project_hooks(project_dir: str, hook_registry: HookRegistry | None = None) -> None:
    """
    从项目目录加载钩子定义

    查找 {project_dir}/hooks.py 文件并执行，
    该文件应该使用 @hooks.register() 装饰器注册钩子。

    线程安全：并发加载时通过锁串行化，避免全局 hooks 交叉污染。

    Args:
        project_dir: 项目目录路径
        hook_registry: 目标 HookRegistry 实例。为 None 时使用全局 hooks。
    """
    import importlib.util
    import sys
    from pathlib import Path

    target = hook_registry if hook_registry is not None else hooks

    # Clear previously registered project hooks to avoid cross-project pollution
    target.clear()

    hooks_file = Path(project_dir) / "hooks.py"
    if not hooks_file.exists():
        logger.debug(f"[Hooks] No hooks.py found in {project_dir}")
        return

    # 使用唯一模块名避免 sys.modules 冲突
    module_name = f"_project_hooks_{id(target)}_{id(hooks_file)}"

    # 锁保护：临时替换全局 hooks 期间不允许其他线程同时加载
    with _load_hooks_lock:
        try:
            _swapped = False
            _original_hooks = None
            if target is not hooks:
                import mem_deep_research_core.core.hooks as _self_module

                _original_hooks = _self_module.hooks
                _self_module.hooks = target
                _swapped = True

            try:
                spec = importlib.util.spec_from_file_location(module_name, hooks_file)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Cannot load hooks from {hooks_file}")

                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

                logger.info(f"[Hooks] Loaded project hooks from {hooks_file}")
                logger.info(f"[Hooks] Registered hooks: {target.list_hooks()}")
            finally:
                if _swapped and _original_hooks is not None:
                    import mem_deep_research_core.core.hooks as _self_module

                    _self_module.hooks = _original_hooks
                # 清理 sys.modules 避免累积
                sys.modules.pop(module_name, None)
        except Exception as e:
            logger.warning(f"[Hooks] Failed to load project hooks: {e}")
