"""
LLM Router — 任务复杂度自动路由

Auto 模式下的统一路由入口，main_loop 只需调一次 router.route()。

路由优先级：
1. Hook on_route_classify — 用户完全自定义分类逻辑
2. 结构信号路由 — 根据 tool_count / task_engine / sub_agents 判断（零成本）
3. LLM 分类路由 — 用轻量模型判断（需配置 router_model）
4. 默认 standard

Auto 模式默认使用 adaptive 策略：跳过 LLM 分类，先用 standard 跑第一轮，
根据第一轮 LLM 的实际行为（有无工具调用、是否直接回答）自动定模式。
调用 LLMRouter.adaptive_classify() 即可。

路由结果通过 Hook on_route_apply 后处理，可修改 mode / reasoning_effort / thinking_params。

用法：
    router = LLMRouter(hooks=hooks, llm_client=client)
    result = await router.route(query, tool_count=5)
    # result.mode = "quick" | "standard" | "deep"
    # result.reasoning_effort = "low" | "medium" | "high"
    # result.thinking_params = {"thinking": {"type": "adaptive"}, ...}

Hook 扩展示例：
    @hooks.register("on_route_classify")
    def my_classifier(ctx, original_fn):
        return "deep"  # 覆盖分类

    @hooks.register("on_route_apply")
    def my_apply(ctx, original_fn):
        return {"mode": ctx.result, "reasoning_effort": "high"}  # 覆盖应用
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from mem_deep_research_core.core.constants import (
    EXECUTION_MODE_DEEP,
    EXECUTION_MODE_QUICK,
    EXECUTION_MODE_STANDARD,
)

logger = logging.getLogger("mem_deep_research")

# 默认路由 prompt
_ROUTING_SYSTEM_PROMPT = (
    "You are a task complexity classifier. "
    "Respond with EXACTLY one word: quick, standard, or deep."
)

_ROUTING_USER_TEMPLATE = (
    "Classify this task's complexity:\n"
    '- "quick" — simple question, greeting, calculation, translation, factual lookup (1-2 steps)\n'
    '- "standard" — moderate task needing tool calls, analysis, or multi-step work (3-10 steps)\n'
    '- "deep" — complex research, investigation, comparison, report generation (10+ steps)\n\n'
    "Task: {query}\n\n"
    "Context: {tool_count} tools available.\n\n"
    "Your answer (one word):"
)

# 执行模式 → 推荐 reasoning_effort 映射
_MODE_TO_EFFORT = {
    EXECUTION_MODE_QUICK: "low",
    EXECUTION_MODE_STANDARD: "medium",
    EXECUTION_MODE_DEEP: "high",
}

_VALID_MODES = {EXECUTION_MODE_QUICK, EXECUTION_MODE_STANDARD, EXECUTION_MODE_DEEP}


@dataclass
class RouteResult:
    """路由结果"""

    mode: str = EXECUTION_MODE_STANDARD
    reasoning_effort: str = "medium"
    thinking_params: dict = field(default_factory=dict)
    source: str = "default"  # "adaptive" | "router" | "structural" | "hook" | "default"
    metadata: dict = field(default_factory=dict)


class LLMRouter:
    """LLM 任务复杂度路由器

    统一路由入口，处理所有 auto 模式逻辑：
    - 结构信号路由（零成本）
    - LLM 分类路由（可选）
    - Provider thinking 参数注入
    - Hook 扩展

    所有 provider 差异通过 llm_client.get_thinking_params() 处理，
    router 不需要知道具体 provider 类型。
    """

    def __init__(
        self,
        hooks=None,
        llm_client=None,
        router_llm_client=None,
    ):
        """
        Args:
            hooks: HookRegistry 实例
            llm_client: 主 LLM 客户端（用于获取 thinking 参数）
            router_llm_client: 路由分类用的 LLM 客户端（轻量模型）
                              如果为 None，仅使用结构信号路由
        """
        self.hooks = hooks
        self.llm_client = llm_client
        self.router_llm_client = router_llm_client

    async def route(
        self,
        query: str,
        tool_count: int = 0,
        has_sub_agents: bool = False,
        task_engine_enabled: bool = False,
        context: dict[str, Any] | None = None,
    ) -> RouteResult:
        """执行路由判断。

        优先级：
        1. Hook on_route_classify
        2. 结构信号
        3. LLM 分类（如果 router_llm_client 可用）
        4. 默认 standard

        路由完成后自动注入 thinking_params（根据 llm_client 类型）。
        """
        from mem_deep_research_core.core.hooks import HookContext

        # 1. Hook: on_route_classify
        if self.hooks and self.hooks.has_hooks("on_route_classify"):
            hook_ctx = HookContext(
                hook_name="on_route_classify",
                query=query,
                context=context or {},
                extra={
                    "tool_count": tool_count,
                    "has_sub_agents": has_sub_agents,
                    "task_engine_enabled": task_engine_enabled,
                    "model_name": getattr(self.llm_client, "model_name", ""),
                    "supports_adaptive": (
                        self.llm_client.supports_adaptive_thinking()
                        if self.llm_client
                        else False
                    ),
                },
            )
            hook_result = await self.hooks.call("on_route_classify", hook_ctx)
            result = self._parse_hook_result(hook_result)
            if result is not None:
                logger.info(f"[LLMRouter] Hook classified as: {result.mode}")
                return self._finalize(result, query, context)

        # 2. 结构信号路由
        structural = self._structural_route(
            tool_count=tool_count,
            has_sub_agents=has_sub_agents,
        )
        if structural is not None:
            logger.info(f"[LLMRouter] Structural route: {structural.mode}")
            return self._finalize(structural, query, context)

        # 3. LLM 分类（优先用 router_llm_client，未配置时回退到主 llm_client）
        classify_client = self.router_llm_client or self.llm_client
        if classify_client is not None:
            llm_result = await self._llm_classify(query, tool_count, classify_client)
            if llm_result is not None:
                source = "router" if self.router_llm_client else "main_llm"
                llm_result.source = source
                logger.info(f"[LLMRouter] LLM classified as: {llm_result.mode} (via {source})")
                return self._finalize(llm_result, query, context)

        # 4. 默认 standard
        result = RouteResult(mode=EXECUTION_MODE_STANDARD, source="default")
        return self._finalize(result, query, context)

    def _structural_route(
        self,
        tool_count: int,
        has_sub_agents: bool,
    ) -> RouteResult | None:
        """结构信号路由（零成本）。

        确定性判断：
        - 有显式配置的子 agent → deep（用户意图明确）
        - 无工具 → quick
        - 其他 → 返回 None（交给下一层判断）
        """
        if has_sub_agents:
            return RouteResult(
                mode=EXECUTION_MODE_DEEP,
                reasoning_effort="high",
                source="structural",
                metadata={"reason": "sub_agents configured"},
            )

        if tool_count == 0:
            return RouteResult(
                mode=EXECUTION_MODE_QUICK,
                reasoning_effort="low",
                source="structural",
                metadata={"reason": "no tools available"},
            )

        return None

    async def _llm_classify(
        self, query: str, tool_count: int, client=None
    ) -> RouteResult | None:
        """用 LLM 判断任务复杂度。client 由调用方确定并传入。"""
        if client is None:
            return None

        user_prompt = _ROUTING_USER_TEMPLATE.format(
            query=query[:500],
            tool_count=tool_count,
        )

        try:
            response = await client.create_message(
                system_prompt=_ROUTING_SYSTEM_PROMPT,
                message_history=[
                    {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
                ],
                tool_definitions=[],
                keep_tool_result=-1,
            )

            if not response:
                return None

            # 兼容 OpenAI 风格 (choices[]) 和 Anthropic 风格 (content[])
            content = None
            if hasattr(response, "choices") and response.choices:
                # OpenAI / OpenRouter: response.choices[0].message.content
                content = getattr(response.choices[0].message, "content", None)
            elif hasattr(response, "content") and response.content:
                # Anthropic: response.content is list of content blocks
                blocks = response.content
                if blocks and hasattr(blocks[0], "text"):
                    content = blocks[0].text
            if not content:
                return None

            choice = content.strip().lower()

            # 精确匹配
            if choice in _VALID_MODES:
                return RouteResult(
                    mode=choice,
                    reasoning_effort=_MODE_TO_EFFORT.get(choice, "medium"),
                    source="router",
                )

            # 从回复中提取关键词
            for mode in (EXECUTION_MODE_DEEP, EXECUTION_MODE_STANDARD, EXECUTION_MODE_QUICK):
                if mode in choice:
                    return RouteResult(
                        mode=mode,
                        reasoning_effort=_MODE_TO_EFFORT.get(mode, "medium"),
                        source="router",
                    )

        except Exception as e:
            logger.warning(f"[LLMRouter] LLM classification failed: {e}")

        return None

    def _finalize(self, result: RouteResult, query: str, context: dict | None) -> RouteResult:
        """最终处理：注入 thinking_params + 调用 on_route_apply hook。"""
        # 注入 thinking_params（根据 llm_client 类型）
        if self.llm_client:
            # 将路由结果的 reasoning_effort 同步到 llm_client
            if hasattr(self.llm_client, "reasoning_effort"):
                self.llm_client.reasoning_effort = result.reasoning_effort
            result.thinking_params = self.llm_client.get_thinking_params()

        # Hook: on_route_apply
        if self.hooks and self.hooks.has_hooks("on_route_apply"):
            from mem_deep_research_core.core.hooks import HookContext

            hook_ctx = HookContext(
                hook_name="on_route_apply",
                query=query,
                context=context or {},
                result=result.mode,
                extra={
                    "reasoning_effort": result.reasoning_effort,
                    "thinking_params": result.thinking_params,
                    "source": result.source,
                    "metadata": result.metadata,
                    "model_name": getattr(self.llm_client, "model_name", ""),
                },
            )

            hook_result = self.hooks.call_sync("on_route_apply", hook_ctx)
            effort_changed = False
            if isinstance(hook_result, dict):
                result.mode = hook_result.get("mode", result.mode)
                new_effort = hook_result.get("reasoning_effort", result.reasoning_effort)
                effort_changed = new_effort != result.reasoning_effort
                result.reasoning_effort = new_effort
                if "thinking_params" in hook_result:
                    result.thinking_params = hook_result["thinking_params"]
                result.source = "hook"
            elif isinstance(hook_result, str) and hook_result in _VALID_MODES:
                result.mode = hook_result
                new_effort = _MODE_TO_EFFORT.get(hook_result, result.reasoning_effort)
                effort_changed = new_effort != result.reasoning_effort
                result.reasoning_effort = new_effort
                result.source = "hook"

            # Hook 修改了 effort 但没显式指定 thinking_params → 重新生成
            if effort_changed and "thinking_params" not in (hook_result if isinstance(hook_result, dict) else {}):
                if self.llm_client and hasattr(self.llm_client, "reasoning_effort"):
                    self.llm_client.reasoning_effort = result.reasoning_effort
                    result.thinking_params = self.llm_client.get_thinking_params()

            # 同步回 llm_client
            if self.llm_client and hasattr(self.llm_client, "reasoning_effort"):
                self.llm_client.reasoning_effort = result.reasoning_effort

        return result

    # ------------------------------------------------------------------
    # Adaptive classification — zero-cost, based on first-turn LLM behavior
    # ------------------------------------------------------------------

    # Threshold: if the first turn produces >= this many tool calls, upgrade to deep
    ADAPTIVE_DEEP_TOOL_THRESHOLD = 3

    @staticmethod
    def adaptive_classify(
        *,
        should_break: bool,
        tool_calls: list | None,
        has_spawn_agent: bool = False,
        allow_deep: bool = True,
    ) -> RouteResult:
        """Classify execution mode from first-turn LLM behavior (zero LLM cost).

        Called after the first main-LLM turn completes. Decides whether the task
        is quick (already answered), deep (many tools / spawn), or standard.

        Args:
            should_break: Whether the LLM signalled completion (no more turns needed).
            tool_calls: Raw tool_calls from the LLM response.
                        Format: [list_of_calls, list_of_raw_calls] or falsy.
            has_spawn_agent: Whether any tool call is spawn_agent.
            allow_deep: If False, clamp deep → standard (used by simple_auto mode).

        Returns:
            RouteResult with mode and reasoning_effort.
        """
        has_tools = bool(
            tool_calls
            and len(tool_calls) >= 2
            and (len(tool_calls[0]) > 0 or len(tool_calls[1]) > 0)
        )
        tool_count = len(tool_calls[0]) if has_tools else 0

        # Case 1: LLM answered directly without tools → quick
        if should_break and not has_tools:
            return RouteResult(
                mode=EXECUTION_MODE_QUICK,
                reasoning_effort="low",
                source="adaptive",
                metadata={"reason": "direct_answer_no_tools"},
            )

        # Case 2: Spawn agent requested or many tool calls → deep (if allowed)
        if has_spawn_agent or tool_count >= LLMRouter.ADAPTIVE_DEEP_TOOL_THRESHOLD:
            if allow_deep:
                return RouteResult(
                    mode=EXECUTION_MODE_DEEP,
                    reasoning_effort="high",
                    source="adaptive",
                    metadata={
                        "reason": "spawn_agent"
                        if has_spawn_agent
                        else f"tool_count={tool_count}",
                    },
                )
            # Clamped: would be deep but simple_auto caps at standard
            return RouteResult(
                mode=EXECUTION_MODE_STANDARD,
                reasoning_effort="medium",
                source="adaptive",
                metadata={
                    "reason": f"clamped_from_deep(tool_count={tool_count})",
                },
            )

        # Case 3: moderate tool usage → standard
        return RouteResult(
            mode=EXECUTION_MODE_STANDARD,
            reasoning_effort="medium",
            source="adaptive",
            metadata={"reason": f"tool_count={tool_count}"},
        )

    def _parse_hook_result(self, hook_result) -> RouteResult | None:
        """解析 on_route_classify hook 的返回值"""
        if isinstance(hook_result, str) and hook_result in _VALID_MODES:
            return RouteResult(
                mode=hook_result,
                reasoning_effort=_MODE_TO_EFFORT.get(hook_result, "medium"),
                source="hook",
            )
        elif isinstance(hook_result, dict) and hook_result.get("mode") in _VALID_MODES:
            return RouteResult(
                mode=hook_result["mode"],
                reasoning_effort=hook_result.get("reasoning_effort", "medium"),
                source="hook",
                metadata=hook_result.get("metadata", {}),
            )
        return None
