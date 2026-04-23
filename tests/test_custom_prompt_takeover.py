"""Tests for custom_system_template takeover semantics + evidence_extraction silent-inject fix.

Covers:
- custom_takes_over=False (default): framework still appends presets / chinese / language_tag.
- custom_takes_over=True: framework appends nothing; custom template fully controls output.
- Placeholders ({{presets}}, {{chinese_context}}, {{language_tag}}, {{mcp_tools_section}})
  are exposed to the custom template so it can render them at chosen positions.
- PromptBuilder skips auto-injecting `evidence_extraction` when a custom template is set.
"""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from mem_deep_research_core.prompts.agent_prompt import AgentPrompt


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def custom_templates_dir(tmp_path: Path) -> Path:
    """Create a temp templates dir with a minimal custom system template."""
    tpl_dir = tmp_path / "templates"
    tpl_dir.mkdir()
    # Minimal custom template that deliberately does NOT reference placeholders,
    # so we can assert exactly what the framework appended (or didn't).
    (tpl_dir / "my_custom.md").write_text("CUSTOM BODY ONLY", encoding="utf-8")

    # A second template that DOES reference placeholders for the takeover path.
    (tpl_dir / "my_custom_with_placeholders.md").write_text(
        "CUSTOM BODY\n\n{{presets}}\n\n{{language_tag}}\n\nEND",
        encoding="utf-8",
    )
    return tpl_dir


# ============================================================
# custom_takes_over=False (default, backward-compatible)
# ============================================================


class TestCustomTakesOverFalse:
    def test_framework_still_appends_language_tag(self, custom_templates_dir):
        prompt = AgentPrompt(
            agent_type="main",
            custom_system_template="my_custom",
            templates_dir=custom_templates_dir,
            custom_takes_over=False,
        )
        out = prompt.generate_system_prompt_with_mcp_tools(
            mcp_servers=[],
            response_language="auto",
        )
        assert "CUSTOM BODY ONLY" in out
        # language_tag is appended despite custom template not referencing it
        assert "<response_language>" in out

    def test_framework_still_appends_presets(self, custom_templates_dir):
        prompt = AgentPrompt(
            agent_type="main",
            presets=["evidence_extraction"],
            custom_system_template="my_custom",
            templates_dir=custom_templates_dir,
            custom_takes_over=False,
        )
        out = prompt.generate_system_prompt_with_mcp_tools(
            mcp_servers=[],
            response_language="English",  # disable language_tag
        )
        assert "CUSTOM BODY ONLY" in out
        # evidence_extraction preset content is appended
        assert "Evidence Extraction Protocol" in out

    def test_framework_still_appends_chinese_context(self, custom_templates_dir):
        prompt = AgentPrompt(
            agent_type="main",
            custom_system_template="my_custom",
            templates_dir=custom_templates_dir,
            custom_takes_over=False,
        )
        out = prompt.generate_system_prompt_with_mcp_tools(
            mcp_servers=[],
            chinese_context=True,
            response_language="English",
        )
        assert "CUSTOM BODY ONLY" in out
        # chinese_context template content is appended
        assert "中文语境" in out


# ============================================================
# custom_takes_over=True (new behavior)
# ============================================================


class TestCustomTakesOverTrue:
    def test_no_appends_when_takeover_enabled(self, custom_templates_dir):
        prompt = AgentPrompt(
            agent_type="main",
            presets=["evidence_extraction"],
            custom_system_template="my_custom",
            templates_dir=custom_templates_dir,
            custom_takes_over=True,
        )
        out = prompt.generate_system_prompt_with_mcp_tools(
            mcp_servers=[],
            chinese_context=True,  # would normally append chinese_context
            response_language="auto",  # would normally append language_tag
        )
        assert out.strip() == "CUSTOM BODY ONLY"
        # Nothing appended
        assert "Evidence Extraction Protocol" not in out
        assert "<response_language>" not in out
        assert "中文语境" not in out

    def test_placeholders_exposed_to_custom_template(self, custom_templates_dir):
        """Custom template can render presets/language_tag via placeholders."""
        prompt = AgentPrompt(
            agent_type="main",
            presets=["evidence_extraction"],
            custom_system_template="my_custom_with_placeholders",
            templates_dir=custom_templates_dir,
            custom_takes_over=True,
        )
        out = prompt.generate_system_prompt_with_mcp_tools(
            mcp_servers=[],
            response_language="auto",
        )
        assert "CUSTOM BODY" in out
        assert "END" in out
        # Rendered at placeholder positions, not appended
        assert "Evidence Extraction Protocol" in out
        assert "## Language" in out
        # Framework did NOT append duplicates at the tail — each appears exactly once
        assert out.count("Evidence Extraction Protocol") == 1
        assert out.count("## Language") == 1

    def test_takeover_falls_back_when_template_missing(self, custom_templates_dir):
        """Missing custom template → fallback to default build; takeover ignored."""
        prompt = AgentPrompt(
            agent_type="main",
            custom_system_template="does_not_exist",
            templates_dir=custom_templates_dir,
            custom_takes_over=True,
        )
        out = prompt.generate_system_prompt_with_mcp_tools(
            mcp_servers=[],
            response_language="English",
        )
        # Fell back to default objective template
        assert "General Objective" in out


# ============================================================
# Default path (no custom template) — unchanged behavior
# ============================================================


class TestDefaultPathUnchanged:
    def test_default_appends_language_tag_and_presets(self):
        prompt = AgentPrompt(
            agent_type="main",
            presets=["evidence_extraction"],
        )
        out = prompt.generate_system_prompt_with_mcp_tools(
            mcp_servers=[],
            response_language="auto",
        )
        assert "General Objective" in out
        assert "Evidence Extraction Protocol" in out
        assert "<response_language>" in out


# ============================================================
# PromptBuilder: evidence_extraction silent-inject only when no custom template
# ============================================================


def _make_prompt_builder(custom_system_template: str | None, templates_dir: Path | None):
    """Build a minimal PromptBuilder with mocked config for testing injection logic."""
    from mem_deep_research_core.core.hooks import HookRegistry
    from mem_deep_research_core.core.prompt_builder import PromptBuilder
    from mem_deep_research_core.utils.external_loader import ConfigLoader

    cfg_dict = {
        "main_agent": {
            "prompt": {
                "agent_type": "main",
                "tool_format": "native",
                "presets": [],
            },
            "context_manager": {"enable_evidence_extraction": True},
            "skill_selection": {"enabled": False},
            "task_engine": {"enabled": False},
        }
    }
    if custom_system_template:
        cfg_dict["main_agent"]["prompt"]["custom_system_template"] = custom_system_template
        cfg_dict["main_agent"]["prompt"]["custom_takes_over"] = True
    if templates_dir:
        cfg_dict["main_agent"]["prompt"]["templates_dir"] = str(templates_dir)

    cfg = OmegaConf.create(cfg_dict)
    hooks = HookRegistry()
    config_loader = ConfigLoader()

    builder = PromptBuilder(
        cfg=cfg,
        context={},
        chinese_context=False,
        hooks=hooks,
        config_loader=config_loader,
    )
    return builder


class TestEvidenceExtractionSilentInject:
    def test_auto_injects_when_no_custom_template(self):
        builder = _make_prompt_builder(custom_system_template=None, templates_dir=None)
        system_prompt, _, _ = builder.build_system_prompt(
            tool_definitions=[], initial_user_content="hi"
        )
        assert "Evidence Extraction Protocol" in system_prompt

    def test_does_not_inject_when_custom_template_set(self, custom_templates_dir):
        builder = _make_prompt_builder(
            custom_system_template="my_custom",
            templates_dir=custom_templates_dir,
        )
        system_prompt, _, _ = builder.build_system_prompt(
            tool_definitions=[], initial_user_content="hi"
        )
        # custom_takes_over=True → only the custom body survives
        assert "CUSTOM BODY ONLY" in system_prompt
        assert "Evidence Extraction Protocol" not in system_prompt


# ============================================================
# chinese_context ⇄ response_language normalization
# ============================================================


class TestChineseContextNormalization:
    """chinese_context=True 与 response_language="Chinese" 应产生完全相同的 prompt。"""

    def _render(self, *, chinese_context: bool, response_language: str) -> str:
        prompt = AgentPrompt(agent_type="main")
        return prompt.generate_system_prompt_with_mcp_tools(
            mcp_servers=[],
            chinese_context=chinese_context,
            response_language=response_language,
        )

    def test_chinese_context_true_suppresses_language_tag(self):
        out = self._render(chinese_context=True, response_language="auto")
        # 中文模板已追加
        assert "中文语境" in out
        # 不再追加 language detection 段（auto 被归一化为 Chinese）
        assert "## Language" not in out

    def test_response_language_chinese_triggers_chinese_template(self):
        out = self._render(chinese_context=False, response_language="Chinese")
        # 即使 chinese_context=False，中文模板也应被追加
        assert "中文语境" in out
        assert "## Language" not in out

    def test_two_forms_produce_identical_output(self):
        a = self._render(chinese_context=True, response_language="auto")
        b = self._render(chinese_context=False, response_language="Chinese")
        # 允许日期/时间戳微小差异：比较结构性段落
        assert ("中文语境" in a) == ("中文语境" in b)
        assert ("## Language" in a) == ("## Language" in b)
        assert "General Objective" in a and "General Objective" in b


# ============================================================
# xml + native-capable provider warning
# ============================================================


class TestXmlNativeProviderWarning:
    def test_warns_when_xml_with_claude_anthropic(self, caplog):
        from omegaconf import OmegaConf

        from mem_deep_research_core.core.hooks import HookRegistry
        from mem_deep_research_core.core.prompt_builder import PromptBuilder
        from mem_deep_research_core.utils.external_loader import ConfigLoader

        cfg = OmegaConf.create({
            "main_agent": {
                "prompt": {"agent_type": "main", "tool_format": "xml", "presets": []},
                "llm": {"provider_class": "ClaudeAnthropicClient"},
                "context_manager": {"enable_evidence_extraction": False},
                "skill_selection": {"enabled": False},
                "task_engine": {"enabled": False},
            }
        })
        builder = PromptBuilder(
            cfg=cfg,
            context={},
            chinese_context=False,
            hooks=HookRegistry(),
            config_loader=ConfigLoader(),
        )
        fake_tool_defs = [{"name": "svr", "tools": [{"name": "t", "description": "d", "schema": {}}]}]
        with caplog.at_level("WARNING", logger="mem_deep_research"):
            builder.build_system_prompt(
                tool_definitions=fake_tool_defs, initial_user_content="hi"
            )
        assert any("tool_format=xml" in rec.message for rec in caplog.records)

    def test_no_warn_when_tool_format_native(self, caplog):
        from omegaconf import OmegaConf

        from mem_deep_research_core.core.hooks import HookRegistry
        from mem_deep_research_core.core.prompt_builder import PromptBuilder
        from mem_deep_research_core.utils.external_loader import ConfigLoader

        cfg = OmegaConf.create({
            "main_agent": {
                "prompt": {"agent_type": "main", "tool_format": "native", "presets": []},
                "llm": {"provider_class": "ClaudeAnthropicClient"},
                "context_manager": {"enable_evidence_extraction": False},
                "skill_selection": {"enabled": False},
                "task_engine": {"enabled": False},
            }
        })
        builder = PromptBuilder(
            cfg=cfg,
            context={},
            chinese_context=False,
            hooks=HookRegistry(),
            config_loader=ConfigLoader(),
        )
        fake_tool_defs = [{"name": "svr", "tools": [{"name": "t", "description": "d", "schema": {}}]}]
        with caplog.at_level("WARNING", logger="mem_deep_research"):
            builder.build_system_prompt(
                tool_definitions=fake_tool_defs, initial_user_content="hi"
            )
        assert not any("tool_format=xml" in rec.message for rec in caplog.records)
