"""Hook 系统单元测试"""

import pytest

from mem_deep_research_core.core.hooks import HookContext, HookRegistry


@pytest.fixture
def registry():
    """每个测试使用独立的 HookRegistry"""
    return HookRegistry()


class TestHookRegistration:
    """钩子注册测试"""

    def test_register_and_call(self, registry):
        """注册钩子后 call 能触发"""
        called = []

        def my_hook(ctx, original_fn):
            called.append(ctx.hook_name)
            return "hooked"

        registry.register_fn("on_agent_start", my_hook)
        result = registry.call("on_agent_start", HookContext(hook_name="on_agent_start"))
        assert result == "hooked"
        assert called == ["on_agent_start"]

    def test_decorator_register(self, registry):
        """装饰器注册方式"""

        @registry.register("on_tool_end")
        def my_hook(ctx, original_fn):
            return "decorated"

        result = registry.call("on_tool_end", HookContext(hook_name="on_tool_end"))
        assert result == "decorated"

    def test_unknown_hook_raises(self, registry):
        """未知钩子名抛 ValueError"""
        with pytest.raises(ValueError, match="Unknown hook"):
            registry.register("on_nonexistent")

        with pytest.raises(ValueError, match="Unknown hook"):
            registry.register_fn("on_nonexistent", lambda c, o: None)

        with pytest.raises(ValueError, match="Unknown hook"):
            registry.call("on_nonexistent", HookContext(hook_name="on_nonexistent"))


class TestHookChaining:
    """钩子链式调用测试"""

    def test_priority_order(self, registry):
        """优先级链式调用顺序正确（高优先级先执行）"""
        order = []

        def hook_low(ctx, original_fn):
            order.append("low")
            return original_fn(ctx)

        def hook_high(ctx, original_fn):
            order.append("high")
            return original_fn(ctx)

        registry.register_fn("on_turn_start", hook_low, priority=1)
        registry.register_fn("on_turn_start", hook_high, priority=10)

        registry.call("on_turn_start", HookContext(hook_name="on_turn_start"))
        assert order == ["high", "low"]

    def test_chain_with_default(self, registry):
        """钩子可以调用 original_fn 执行默认逻辑"""
        registry.set_default("on_agent_start", lambda ctx: "default_result")

        def my_hook(ctx, original_fn):
            base = original_fn(ctx)
            return base + "_enhanced"

        registry.register_fn("on_agent_start", my_hook)
        result = registry.call("on_agent_start", HookContext(hook_name="on_agent_start"))
        assert result == "default_result_enhanced"

    def test_hook_overrides_default(self, registry):
        """钩子可完全覆盖默认实现（不调用 original_fn）"""
        registry.set_default("on_agent_end", lambda ctx: "default")

        def override_hook(ctx, original_fn):
            return "completely_overridden"

        registry.register_fn("on_agent_end", override_hook)
        result = registry.call("on_agent_end", HookContext(hook_name="on_agent_end"))
        assert result == "completely_overridden"


class TestHookDefaults:
    """默认实现测试"""

    def test_no_hooks_uses_default(self, registry):
        """无注册钩子时走默认实现"""
        registry.set_default("on_turn_end", lambda ctx: "default_turn_end")
        result = registry.call("on_turn_end", HookContext(hook_name="on_turn_end"))
        assert result == "default_turn_end"

    def test_no_hooks_no_default_returns_none(self, registry):
        """无注册钩子且无默认实现返回 None"""
        result = registry.call("on_agent_start", HookContext(hook_name="on_agent_start"))
        assert result is None


class TestHookClear:
    """清除钩子测试"""

    def test_clear_specific_hook(self, registry):
        """清除指定钩子"""
        registry.register_fn("on_agent_start", lambda c, o: "a")
        registry.register_fn("on_agent_end", lambda c, o: "b")
        registry.clear("on_agent_start")

        assert not registry.has_hooks("on_agent_start")
        assert registry.has_hooks("on_agent_end")

    def test_clear_all_hooks(self, registry):
        """清除所有钩子"""
        registry.register_fn("on_agent_start", lambda c, o: "a")
        registry.register_fn("on_turn_start", lambda c, o: "b")
        registry.clear()

        assert not registry.has_hooks("on_agent_start")
        assert not registry.has_hooks("on_turn_start")

    def test_list_hooks(self, registry):
        """列出所有钩子及注册数量"""
        registry.register_fn("on_agent_start", lambda c, o: "a")
        registry.register_fn("on_agent_start", lambda c, o: "b")

        hooks_info = registry.list_hooks()
        assert hooks_info["on_agent_start"] == 2
        assert hooks_info["on_agent_end"] == 0


class TestHookContext:
    """HookContext 数据结构测试"""

    def test_context_fields(self):
        """验证 HookContext 字段正确传递"""
        ctx = HookContext(
            hook_name="on_tool_start",
            tool_name="search",
            server_name="google",
            arguments={"query": "test"},
            turn_number=3,
            extra={"custom": "data"},
        )
        assert ctx.hook_name == "on_tool_start"
        assert ctx.tool_name == "search"
        assert ctx.arguments == {"query": "test"}
        assert ctx.turn_number == 3
        assert ctx.extra["custom"] == "data"

    def test_context_defaults(self):
        """验证 HookContext 默认值"""
        ctx = HookContext(hook_name="test")
        assert ctx.query is None
        assert ctx.tool_name is None
        assert ctx.turn_number is None
        assert ctx.extra == {}

    def test_new_context_fields(self):
        """验证新增 HookContext 字段"""
        ctx = HookContext(
            hook_name="on_tool_filter",
            tool_calls_batch=[{"tool_name": "search"}],
            compact_action="masking",
        )
        assert ctx.tool_calls_batch == [{"tool_name": "search"}]
        assert ctx.compact_action == "masking"

    def test_new_context_fields_default_none(self):
        """新增字段默认为 None"""
        ctx = HookContext(hook_name="test")
        assert ctx.tool_calls_batch is None
        assert ctx.compact_action is None


class TestNewHooks:
    """新增 hook 测试"""

    def test_on_tool_filter(self, registry):
        """on_tool_filter 可修改工具调用列表"""

        def filter_hook(ctx, original_fn):
            # 过滤掉 tool_name == "blocked"
            return [c for c in ctx.tool_calls_batch if c.get("tool_name") != "blocked"]

        registry.register_fn("on_tool_filter", filter_hook)

        batch = [
            {"tool_name": "search", "arguments": {}},
            {"tool_name": "blocked", "arguments": {}},
            {"tool_name": "read", "arguments": {}},
        ]
        result = registry.call(
            "on_tool_filter",
            HookContext(hook_name="on_tool_filter", tool_calls_batch=batch),
        )
        assert len(result) == 2
        assert result[0]["tool_name"] == "search"
        assert result[1]["tool_name"] == "read"

    def test_on_context_compact(self, registry):
        """on_context_compact 通知压缩行为"""
        events = []

        def compact_hook(ctx, original_fn):
            events.append(ctx.compact_action)
            return original_fn(ctx)

        registry.register_fn("on_context_compact", compact_hook)
        registry.call(
            "on_context_compact",
            HookContext(hook_name="on_context_compact", compact_action="masking"),
        )
        registry.call(
            "on_context_compact",
            HookContext(hook_name="on_context_compact", compact_action="emergency"),
        )
        assert events == ["masking", "emergency"]

    def test_on_reflection_build(self, registry):
        """on_reflection_build 可修改反思 prompt"""

        def reflection_hook(ctx, original_fn):
            return ctx.result + "\n\nFocus on data quality."

        registry.register_fn("on_reflection_build", reflection_hook)
        result = registry.call(
            "on_reflection_build",
            HookContext(hook_name="on_reflection_build", result="Original reflection"),
        )
        assert result == "Original reflection\n\nFocus on data quality."

    def test_on_tool_filter_not_registered_returns_default(self, registry):
        """on_tool_filter 未注册时返回 None (默认行为)"""
        result = registry.call(
            "on_tool_filter",
            HookContext(hook_name="on_tool_filter", tool_calls_batch=[{"tool_name": "a"}]),
        )
        assert result is None

    def test_on_final_answer(self, registry):
        """on_final_answer 可改写最终答案文本"""

        def final_answer_hook(ctx, original_fn):
            return ctx.result + " [reviewed]"

        registry.register_fn("on_final_answer", final_answer_hook)
        result = registry.call(
            "on_final_answer",
            HookContext(hook_name="on_final_answer", result="Base answer"),
        )
        assert result == "Base answer [reviewed]"
