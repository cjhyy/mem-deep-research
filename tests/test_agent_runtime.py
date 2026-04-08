"""
Tests for AgentRuntime isolation.

Covers:
- Two runtimes have independent hooks and config_loader
- load_project_hooks registers to the correct instance
- Global fallback (get_global_runtime) backward compat
- setup_hook_defaults writes to instance, not global
"""

import pytest

from mem_deep_research_core.core.agent_runtime import AgentRuntime, get_global_runtime
from mem_deep_research_core.core.hooks import HookContext, HookRegistry
from mem_deep_research_core.core.hooks import hooks as global_hooks


@pytest.fixture(autouse=True)
def clean_global_hooks():
    """Reset global hooks before/after each test."""
    global_hooks.clear_all()
    yield
    global_hooks.clear_all()


# ============================================================
# Isolation: two runtimes don't share hooks
# ============================================================


class TestRuntimeIsolation:
    def test_separate_hook_registries(self):
        """Two AgentRuntime instances have independent HookRegistries."""
        rt1 = AgentRuntime()
        rt2 = AgentRuntime()

        assert rt1.hooks is not rt2.hooks

        # Register a hook on rt1 only
        rt1.hooks.register_fn("on_agent_start", lambda ctx, fn: "rt1", priority=0)

        assert rt1.hooks.has_hooks("on_agent_start")
        assert not rt2.hooks.has_hooks("on_agent_start")

    def test_separate_config_loaders(self):
        """Two AgentRuntime instances have independent ConfigLoaders."""
        rt1 = AgentRuntime()
        rt2 = AgentRuntime()

        assert rt1.config_loader is not rt2.config_loader

    def test_hook_call_isolation(self):
        """Hooks registered on rt1 don't fire when rt2.hooks.call() is invoked."""
        rt1 = AgentRuntime()
        rt2 = AgentRuntime()

        calls = []

        rt1.hooks.set_default("on_agent_start", lambda ctx: None)
        rt2.hooks.set_default("on_agent_start", lambda ctx: None)

        rt1.hooks.register_fn(
            "on_agent_start",
            lambda ctx, fn: calls.append("rt1") or fn(ctx),
            priority=0,
        )

        ctx = HookContext(hook_name="on_agent_start", query="test")

        rt2.hooks.call("on_agent_start", ctx)
        assert calls == [], "rt1's hook should NOT fire on rt2.hooks.call()"

        rt1.hooks.call("on_agent_start", ctx)
        assert calls == ["rt1"]

    def test_set_default_isolation(self):
        """set_default on rt1 doesn't affect rt2."""
        rt1 = AgentRuntime()
        rt2 = AgentRuntime()

        rt1.hooks.set_default("on_agent_start", lambda ctx: "default_rt1")

        ctx = HookContext(hook_name="on_agent_start", query="test")
        assert rt1.hooks.call("on_agent_start", ctx) == "default_rt1"
        assert rt2.hooks.call("on_agent_start", ctx) is None  # no default set


# ============================================================
# Global fallback
# ============================================================


class TestGlobalFallback:
    def test_get_global_runtime_uses_global_hooks(self):
        """get_global_runtime wraps the module-level global hooks singleton."""
        rt = get_global_runtime()
        assert rt.hooks is global_hooks

    def test_explicit_hooks_override_global(self):
        """Passing explicit hooks to AgentRuntime bypasses global."""
        custom = HookRegistry()
        rt = AgentRuntime(hooks=custom)
        assert rt.hooks is custom
        assert rt.hooks is not global_hooks


# ============================================================
# setup_hook_defaults
# ============================================================


class TestSetupHookDefaults:
    def test_defaults_registered_on_instance(self):
        """setup_hook_defaults writes defaults to instance hooks, not global."""
        rt = AgentRuntime()
        rt.setup_hook_defaults()

        # Instance should have defaults
        ctx = HookContext(hook_name="on_agent_start", query="test")
        # Should not raise — default is registered
        rt.hooks.call("on_agent_start", ctx)

        # Global should NOT have these defaults (we cleared in fixture)
        assert "on_agent_start" not in global_hooks._default_fns or \
               global_hooks._default_fns.get("on_agent_start") is None


# ============================================================
# load_project_hooks
# ============================================================


class TestLoadProjectHooks:
    def test_load_from_nonexistent_dir(self):
        """Loading hooks from a dir without hooks.py should not raise."""
        rt = AgentRuntime()
        rt.load_project_hooks("/tmp/nonexistent_project_dir_12345")
        # No hooks registered
        assert not rt.hooks.has_hooks("on_agent_start")

    def test_load_restores_global_hooks(self, tmp_path):
        """After load_project_hooks, the global hooks singleton is restored."""
        # Write a hooks.py that registers on the global hooks import
        hooks_file = tmp_path / "hooks.py"
        hooks_file.write_text(
            "from mem_deep_research_core.core.hooks import hooks\n"
            "@hooks.register('on_agent_start', priority=5)\n"
            "def my_hook(ctx, original_fn):\n"
            "    return 'project_hook'\n"
        )

        original_global = global_hooks

        rt = AgentRuntime()
        rt.load_project_hooks(str(tmp_path))

        # Global should be restored to original object
        from mem_deep_research_core.core import hooks as hooks_module
        assert hooks_module.hooks is original_global

        # But the runtime's hooks should have the registered hook
        assert rt.hooks.has_hooks("on_agent_start")
