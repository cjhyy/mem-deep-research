"""
Regression tests for framework iteration fixes (doc #19).

Covers:
- P0-1: ToolManager MCP session context isolation (fingerprint + recreation)
- P0-2: SubAgentRunner ContextManager inherits main config
- P0-3: Config validation fail-fast on critical fields
- P1-4: InputCompiler @file security boundary
- P1-5: Answer cleaning — evidence/next_skills tag stripping
- P2-8: Version metadata consistency
"""

import os
import tempfile
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from mem_deep_research_core.core.hooks import HookRegistry


# ============================================================
# P0-1: ToolManager context fingerprint
# ============================================================


class TestToolManagerContextFingerprint:
    def test_fingerprint_empty_context(self):
        from mem_deep_research_core.tool.manager import ToolManager

        assert ToolManager._compute_context_fingerprint(None) == ""
        assert ToolManager._compute_context_fingerprint({}) == ""

    def test_fingerprint_ignores_internal_keys(self):
        from mem_deep_research_core.tool.manager import ToolManager

        ctx_a = {"user_id": "123", "_secure": {"token": "abc"}}
        ctx_b = {"user_id": "123", "_secure": {"token": "xyz"}}
        # _secure is skipped, so fingerprints should match
        assert ToolManager._compute_context_fingerprint(ctx_a) == (
            ToolManager._compute_context_fingerprint(ctx_b)
        )

    def test_fingerprint_changes_with_context(self):
        from mem_deep_research_core.tool.manager import ToolManager

        ctx_a = {"user_id": "user-A", "org_id": "org-1"}
        ctx_b = {"user_id": "user-B", "org_id": "org-1"}
        fp_a = ToolManager._compute_context_fingerprint(ctx_a)
        fp_b = ToolManager._compute_context_fingerprint(ctx_b)
        assert fp_a != fp_b

    def test_fingerprint_stable(self):
        from mem_deep_research_core.tool.manager import ToolManager

        ctx = {"user_id": "u1", "org_id": "o1", "timezone": "UTC"}
        assert ToolManager._compute_context_fingerprint(ctx) == (
            ToolManager._compute_context_fingerprint(ctx)
        )

    def test_fingerprint_skips_non_string_values(self):
        from mem_deep_research_core.tool.manager import ToolManager

        ctx = {"user_id": "u1", "count": 42, "active": True}
        # Only user_id is a string, so fingerprint should be non-empty
        fp = ToolManager._compute_context_fingerprint(ctx)
        assert fp != ""


# ============================================================
# P0-2: SubAgentRunner._create_context_manager
# ============================================================


class TestSubAgentContextManagerInheritance:
    def test_create_context_manager_inherits_config(self):
        from mem_deep_research_core.core.sub_agent_runner import SubAgentRunner
        from mem_deep_research_core.utils.external_loader import ConfigLoader

        cfg = OmegaConf.create({
            "main_agent": {
                "max_turns": 3,
                "context_manager": {
                    "compact_at_ratio": 0.5,
                    "summarize_at_ratio": 0.7,
                    "result_offload_threshold": 8000,
                    "enable_dedup": False,
                },
            },
            "output_dir": "test_logs/",
        })
        runner = SubAgentRunner(
            sub_agent_tool_managers={},
            sub_agent_llm_client=MagicMock(),
            output_formatter=MagicMock(),
            cfg=cfg,
            task_log=MagicMock(),
            hooks=HookRegistry(),
            config_loader=ConfigLoader(),
        )

        cm = runner._create_context_manager(MagicMock())

        assert cm.config.compact_at_ratio == 0.5
        assert cm.config.summarize_at_ratio == 0.7
        assert cm.config.result_offload_threshold == 8000
        assert cm.config.enable_dedup is False

    def test_create_context_manager_sets_offload_dir(self):
        from mem_deep_research_core.core.sub_agent_runner import SubAgentRunner
        from mem_deep_research_core.utils.external_loader import ConfigLoader

        cfg = OmegaConf.create({
            "main_agent": {"max_turns": 3},
            "output_dir": "/tmp/test_output",
        })
        runner = SubAgentRunner(
            sub_agent_tool_managers={},
            sub_agent_llm_client=MagicMock(),
            output_formatter=MagicMock(),
            cfg=cfg,
            task_log=MagicMock(),
            hooks=HookRegistry(),
            config_loader=ConfigLoader(),
        )

        cm = runner._create_context_manager(MagicMock())

        assert cm.config.result_offload_dir or cm._offload_dir
        # Should have set offload dir to output_dir/offloaded_results
        expected_suffix = os.path.join("/tmp/test_output", "offloaded_results")
        assert cm._offload_dir == expected_suffix

    def test_create_context_manager_passes_hooks(self):
        from mem_deep_research_core.core.sub_agent_runner import SubAgentRunner
        from mem_deep_research_core.utils.external_loader import ConfigLoader

        hooks = HookRegistry()
        cfg = OmegaConf.create({"main_agent": {"max_turns": 3}})
        runner = SubAgentRunner(
            sub_agent_tool_managers={},
            sub_agent_llm_client=MagicMock(),
            output_formatter=MagicMock(),
            cfg=cfg,
            task_log=MagicMock(),
            hooks=hooks,
            config_loader=ConfigLoader(),
        )

        cm = runner._create_context_manager(MagicMock())

        assert cm._hooks is hooks

    def test_create_context_manager_inherits_parent_offload_dir(self):
        """When parent_context_manager is provided, offload_dir should come from
        the parent's resolved path, not re-derived from config."""
        from mem_deep_research_core.core.context_manager import ContextManager
        from mem_deep_research_core.core.sub_agent_runner import SubAgentRunner
        from mem_deep_research_core.utils.external_loader import ConfigLoader

        cfg = OmegaConf.create({
            "main_agent": {"max_turns": 3},
            "output_dir": "/default/path",
        })
        runner = SubAgentRunner(
            sub_agent_tool_managers={},
            sub_agent_llm_client=MagicMock(),
            output_formatter=MagicMock(),
            cfg=cfg,
            task_log=MagicMock(),
            hooks=HookRegistry(),
            config_loader=ConfigLoader(),
        )

        # Simulate parent with a custom offload dir
        parent_cm = ContextManager()
        parent_cm.set_offload_dir("/custom/offload/path")

        cm = runner._create_context_manager(MagicMock(), parent_context_manager=parent_cm)

        assert cm._offload_dir == "/custom/offload/path"

    def test_create_context_manager_falls_back_without_parent(self):
        """Without parent_context_manager, offload_dir should come from config."""
        from mem_deep_research_core.core.sub_agent_runner import SubAgentRunner
        from mem_deep_research_core.utils.external_loader import ConfigLoader

        cfg = OmegaConf.create({
            "main_agent": {"max_turns": 3},
            "output_dir": "/config/path",
        })
        runner = SubAgentRunner(
            sub_agent_tool_managers={},
            sub_agent_llm_client=MagicMock(),
            output_formatter=MagicMock(),
            cfg=cfg,
            task_log=MagicMock(),
            hooks=HookRegistry(),
            config_loader=ConfigLoader(),
        )

        cm = runner._create_context_manager(MagicMock())

        assert cm._offload_dir == os.path.join("/config/path", "offloaded_results")


# ============================================================
# Offload registry merge
# ============================================================


class TestOffloadRegistryMerge:
    def test_merge_adds_new_entries(self):
        from mem_deep_research_core.core.context_manager import ContextManager, OffloadRecord

        parent = ContextManager()
        child = ContextManager()
        child._offload_registry["ref_a"] = OffloadRecord(ref="ref_a", turn=1, char_count=100)

        parent.merge_offload_registry(child)

        assert "ref_a" in parent._offload_registry

    def test_merge_overwrites_on_collision(self):
        """Collision should overwrite with sub-agent record (not silently discard)."""
        from mem_deep_research_core.core.context_manager import ContextManager, OffloadRecord

        parent = ContextManager()
        parent._offload_registry["ref_x"] = OffloadRecord(
            ref="ref_x", turn=1, char_count=50, state="backed_up"
        )
        child = ContextManager()
        child._offload_registry["ref_x"] = OffloadRecord(
            ref="ref_x", turn=2, char_count=200, state="backed_up"
        )

        parent.merge_offload_registry(child)

        # Sub-agent's record should win
        assert parent._offload_registry["ref_x"].char_count == 200
        assert parent._offload_registry["ref_x"].turn == 2


# ============================================================
# P0-3: Config validation fail-fast
# ============================================================


class TestConfigValidationFailFast:
    def test_valid_config_does_not_raise(self):
        """Normal config should pass validation silently."""
        from mem_deep_research_core.deep_research import DeepResearch

        dr = DeepResearch.__new__(DeepResearch)
        dr._cfg = OmegaConf.create({
            "main_agent": {
                "llm": {
                    "provider_class": "ClaudeAnthropicClient",
                    "model_name": "claude-sonnet-4-20250514",
                },
                "prompt": {"agent_type": "main"},
            }
        })
        # Should not raise
        dr._validate_config()

    def test_missing_critical_field_raises(self):
        """Missing provider_class should raise ConfigValidationError."""
        from mem_deep_research_core.deep_research import DeepResearch
        from mem_deep_research_core.exceptions import ConfigValidationError

        dr = DeepResearch.__new__(DeepResearch)
        dr._cfg = OmegaConf.create({
            "main_agent": {
                "llm": {
                    # provider_class missing — critical
                    "model_name": "some-model",
                },
                "prompt": {"agent_type": "main"},
            }
        })
        with pytest.raises(ConfigValidationError, match="critical"):
            dr._validate_config()


# ============================================================
# P1-4: InputCompiler @file security
# ============================================================


class TestInputCompilerSecurity:
    def test_blocks_env_file(self):
        from mem_deep_research_core.core.input_compiler import InputCompiler

        compiler = InputCompiler(hooks=HookRegistry())

        with tempfile.NamedTemporaryFile(
            prefix=".env", suffix="", delete=False, mode="w"
        ) as f:
            f.write("SECRET=abc123")
            env_path = f.name

        try:
            # Rename to exact .env
            env_dir = os.path.dirname(env_path)
            dot_env_path = os.path.join(env_dir, ".env")
            os.rename(env_path, dot_env_path)
            result = compiler._read_file(dot_env_path)
            assert result is None
        finally:
            if os.path.exists(dot_env_path):
                os.unlink(dot_env_path)

    def test_blocks_private_key(self):
        from mem_deep_research_core.core.input_compiler import InputCompiler

        compiler = InputCompiler(hooks=HookRegistry())

        with tempfile.NamedTemporaryFile(
            prefix="id_rsa", suffix="", delete=False, mode="w", dir=tempfile.gettempdir()
        ) as f:
            f.write("PRIVATE KEY CONTENT")
            key_path = f.name

        try:
            real_key = os.path.join(os.path.dirname(key_path), "id_rsa")
            os.rename(key_path, real_key)
            result = compiler._read_file(real_key)
            assert result is None
        finally:
            if os.path.exists(real_key):
                os.unlink(real_key)

    def test_allowlist_blocks_outside_dirs(self):
        from mem_deep_research_core.core.input_compiler import InputCompiler

        with tempfile.TemporaryDirectory() as allowed_dir:
            with tempfile.TemporaryDirectory() as outside_dir:
                # Create a file outside allowed dir
                outside_file = os.path.join(outside_dir, "data.txt")
                with open(outside_file, "w") as f:
                    f.write("some data")

                compiler = InputCompiler(
                    hooks=HookRegistry(),
                    file_ref_allowed_dirs=[allowed_dir],
                )

                result = compiler._read_file(outside_file)
                assert result is None

    def test_allowlist_allows_inside_dirs(self):
        from mem_deep_research_core.core.input_compiler import InputCompiler

        with tempfile.TemporaryDirectory() as allowed_dir:
            inside_file = os.path.join(allowed_dir, "data.txt")
            with open(inside_file, "w") as f:
                f.write("allowed content")

            compiler = InputCompiler(
                hooks=HookRegistry(),
                file_ref_allowed_dirs=[allowed_dir],
            )

            result = compiler._read_file(inside_file)
            assert result == "allowed content"

    def test_no_allowlist_allows_normal_files(self):
        from mem_deep_research_core.core.input_compiler import InputCompiler

        compiler = InputCompiler(hooks=HookRegistry())

        with tempfile.NamedTemporaryFile(
            suffix=".txt", delete=False, mode="w"
        ) as f:
            f.write("hello world")
            path = f.name

        try:
            result = compiler._read_file(path)
            assert result == "hello world"
        finally:
            os.unlink(path)


# ============================================================
# P1-5: Tag stripping
# ============================================================


class TestTagStripping:
    def test_strip_response_language_tag(self):
        from mem_deep_research_core.core.main_loop import _strip_response_language_tag

        text = "Hello <response_language>English</response_language> world"
        cleaned = _strip_response_language_tag(text)
        assert "<response_language>" not in cleaned
        assert "Hello" in cleaned
        assert "world" in cleaned

    def test_strip_next_skills_tag(self):
        from mem_deep_research_core.skills.inline_selector import InlineSkillSelector

        text = "Analysis done.\n<next_skills>skill-a, skill-b</next_skills>"
        cleaned = InlineSkillSelector.strip_next_skills_tag(text)
        assert "<next_skills>" not in cleaned
        assert "Analysis done." in cleaned

    def test_strip_evidence_tags(self):
        from mem_deep_research_core.core.main_loop import _extract_evidence_tags
        from mem_deep_research_core.core.memory import SessionMemory

        mem = SessionMemory()
        text = "Found result. <evidence>important finding</evidence> End."
        cleaned = _extract_evidence_tags(text, turn=1, session_memory=mem)
        assert "<evidence>" not in cleaned
        assert "Found result." in cleaned


# ============================================================
# P2-8: Version consistency
# ============================================================


class TestVersionConsistency:
    def test_core_version_matches_wrapper(self):
        import mem_deep_research
        import mem_deep_research_core

        assert mem_deep_research.__version__ == mem_deep_research_core.__version__

    def test_version_is_not_empty(self):
        import mem_deep_research_core

        assert mem_deep_research_core.__version__
        assert len(mem_deep_research_core.__version__) > 0
