"""Phase 2c 测试：DeepResearchProfile / StandardProfile 的 lifecycle 钩子决策

验证 profile 钩子返回值正确，对应原 main_loop.py 中 29 处 is_deep_mode /
is_quick_mode 分支的语义等价。
"""

import pytest
from unittest.mock import MagicMock

from mem_deep_research_core.core.memory import SessionMemory
from mem_deep_research_core.core.profiles import (
    DeepResearchProfile,
    StandardProfile,
)


def _ctx(*, mode="standard", tool_calls=0, sm=None):
    """构造 mock ProfileContext。"""
    c = MagicMock()
    c.mode = mode
    c.tool_calls_executed = tool_calls
    c.session_memory = sm if sm is not None else SessionMemory()
    return c


# =========================================================
# StandardProfile lifecycle 钩子
# =========================================================


class TestStandardProfileLifecycle:
    @pytest.mark.asyncio
    async def test_no_reflection(self):
        p = StandardProfile()
        assert await p.should_inject_reflection(_ctx()) is False

    @pytest.mark.asyncio
    async def test_no_verify(self):
        p = StandardProfile()
        assert await p.should_run_verify(_ctx(mode="deep", tool_calls=5)) is False

    @pytest.mark.asyncio
    async def test_no_task_plan(self):
        # Base Profile 默认无 create_task_plan / should_create_task_plan，
        # StandardProfile 继承默认，期望 create_task_plan 返回 None
        p = StandardProfile()
        assert await p.create_task_plan(_ctx()) is None

    @pytest.mark.asyncio
    async def test_no_inline_skills(self):
        p = StandardProfile()
        assert await p.should_process_inline_skills(_ctx()) is False

    @pytest.mark.asyncio
    async def test_no_final_summary_by_default(self):
        p = StandardProfile()
        assert await p.needs_final_summary(0, "", _ctx()) is False
        assert await p.needs_final_summary(5, "text", _ctx(mode="deep", tool_calls=5)) is False

    @pytest.mark.asyncio
    async def test_final_summary_user_opt_in(self):
        p = StandardProfile(config={"generate_summary": True})
        assert await p.needs_final_summary(0, "text", _ctx()) is True


# =========================================================
# DeepResearchProfile lifecycle 钩子
# =========================================================


class TestDeepResearchProfileLifecycle:
    @pytest.mark.asyncio
    async def test_reflection_enabled_in_deep_standard(self):
        p = DeepResearchProfile()
        assert await p.should_inject_reflection(_ctx(mode="deep")) is True
        assert await p.should_inject_reflection(_ctx(mode="standard")) is True

    @pytest.mark.asyncio
    async def test_reflection_disabled_in_quick(self):
        p = DeepResearchProfile()
        assert await p.should_inject_reflection(_ctx(mode="quick")) is False

    @pytest.mark.asyncio
    async def test_reflection_can_be_disabled_by_config(self):
        p = DeepResearchProfile(config={"reflection_enabled": False})
        assert await p.should_inject_reflection(_ctx(mode="deep")) is False

    @pytest.mark.asyncio
    async def test_verify_requires_all_conditions(self):
        p = DeepResearchProfile()
        sm = SessionMemory()
        sm.add_finding("some fact")  # 让 session_memory 非空

        # 全部条件满足
        assert await p.should_run_verify(_ctx(mode="deep", tool_calls=3, sm=sm)) is True

        # mode 不是 deep
        assert await p.should_run_verify(_ctx(mode="standard", tool_calls=3, sm=sm)) is False
        assert await p.should_run_verify(_ctx(mode="quick", tool_calls=3, sm=sm)) is False

        # 无工具调用
        assert await p.should_run_verify(_ctx(mode="deep", tool_calls=0, sm=sm)) is False

        # session_memory 为空
        assert await p.should_run_verify(
            _ctx(mode="deep", tool_calls=3, sm=SessionMemory())
        ) is False

    @pytest.mark.asyncio
    async def test_verify_can_be_disabled_by_config(self):
        p = DeepResearchProfile(config={"enable_verify": False})
        sm = SessionMemory()
        sm.add_finding("fact")
        assert await p.should_run_verify(_ctx(mode="deep", tool_calls=3, sm=sm)) is False

    @pytest.mark.asyncio
    async def test_task_plan_enabled_except_quick(self):
        p = DeepResearchProfile()
        assert await p.should_create_task_plan(_ctx(mode="deep")) is True
        assert await p.should_create_task_plan(_ctx(mode="standard")) is True
        assert await p.should_create_task_plan(_ctx(mode="quick")) is False

    @pytest.mark.asyncio
    async def test_task_plan_user_disable(self):
        p = DeepResearchProfile(config={"auto_task_plan": False})
        assert await p.should_create_task_plan(_ctx(mode="deep")) is False

    @pytest.mark.asyncio
    async def test_inline_skills_except_quick(self):
        p = DeepResearchProfile()
        assert await p.should_process_inline_skills(_ctx(mode="deep")) is True
        assert await p.should_process_inline_skills(_ctx(mode="standard")) is True
        assert await p.should_process_inline_skills(_ctx(mode="quick")) is False

    @pytest.mark.asyncio
    async def test_summary_forced_in_deep_with_tools(self):
        p = DeepResearchProfile()
        assert await p.needs_final_summary(3, "text", _ctx(mode="deep", tool_calls=3)) is True

    @pytest.mark.asyncio
    async def test_summary_not_forced_without_tools(self):
        """mode=deep + tool_calls=0 时走默认 generate_summary_default=True。"""
        p = DeepResearchProfile()
        assert await p.needs_final_summary(0, "text", _ctx(mode="deep", tool_calls=0)) is True

    @pytest.mark.asyncio
    async def test_summary_respects_disabled_default(self):
        """用户显式设 generate_summary=False 后，没 tool_calls 的场景不生成 summary。"""
        p = DeepResearchProfile(config={"generate_summary": False})
        # deep + 有工具调用仍强制
        assert await p.needs_final_summary(3, "text", _ctx(mode="deep", tool_calls=3)) is True
        # 无工具 → 遵循 default=False
        assert await p.needs_final_summary(0, "text", _ctx(mode="standard", tool_calls=0)) is False
