"""测试 offload 默认关闭 + sidecar 注入跟随 OffloadEvidenceStrategy"""

from unittest.mock import MagicMock

import pytest

from mem_deep_research_core.core.constants import DEFAULT_RESULT_OFFLOAD_THRESHOLD
from mem_deep_research_core.core.context_manager import ContextManager, ContextManagerConfig
from mem_deep_research_core.core.profiles import DeepResearchProfile, StandardProfile
from mem_deep_research_core.memory_extraction import (
    OffloadEvidenceStrategy,
    SummaryEvidenceStrategy,
)


class TestOffloadDefaults:
    def test_default_threshold_is_zero(self):
        """Offload 默认关闭：DEFAULT_RESULT_OFFLOAD_THRESHOLD = 0"""
        assert DEFAULT_RESULT_OFFLOAD_THRESHOLD == 0

    def test_context_manager_config_default_disables_offload(self):
        """ContextManagerConfig 默认不开启 offload。"""
        cfg = ContextManagerConfig()
        assert cfg.result_offload_threshold == 0

    def test_backup_large_result_noop_when_threshold_zero(self):
        """threshold=0 时 backup_large_result 直接返回 None，不写文件。"""
        cm = ContextManager(config=ContextManagerConfig(result_offload_threshold=0))
        ref = cm.backup_large_result("x" * 10000, tool_name="search", turn=1)
        assert ref is None

    def test_backup_large_result_works_when_enabled(self, tmp_path):
        """用户显式开启 offload 时照常工作。"""
        cfg = ContextManagerConfig(
            result_offload_threshold=100,
            result_offload_dir=str(tmp_path),
        )
        cm = ContextManager(config=cfg)
        cm.set_offload_dir(str(tmp_path))
        ref = cm.backup_large_result("x" * 500, tool_name="search", turn=1)
        assert ref is not None
        assert ref.startswith("toolmsg_")


class TestOffloadEvidenceStrategyPresence:
    """验证 profile 的默认 strategies 含 OffloadEvidenceStrategy
    （sidecar 注入的 gate 依赖此 strategy 是否在 profile.extraction_strategies 中）。"""

    def test_standard_profile_includes_offload_evidence(self):
        p = StandardProfile()
        names = [getattr(s, "name", "") for s in p.extraction_strategies]
        assert "offload_evidence" in names

    def test_deep_research_profile_includes_offload_evidence(self):
        p = DeepResearchProfile()
        names = [getattr(s, "name", "") for s in p.extraction_strategies]
        assert "offload_evidence" in names

    def test_user_can_remove_offload_evidence(self):
        """用户显式覆盖 extraction_strategies 时，sidecar gate 会检测到并跳过注入。
        此处只验证 profile 允许完全覆盖。"""
        p = StandardProfile(config={"extraction_strategies": []})
        assert p.extraction_strategies == []

        # 用户保留 summary_evidence 但移除 offload_evidence
        p2 = StandardProfile(
            config={"extraction_strategies": [SummaryEvidenceStrategy()]}
        )
        names = [getattr(s, "name", "") for s in p2.extraction_strategies]
        assert names == ["summary_evidence"]
        assert "offload_evidence" not in names
