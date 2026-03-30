"""
InlineSkillSelector 测试

覆盖：
- parse_next_skills: 从 LLM 回复中解析 <next_skills> 标签
- strip_next_skills_tag: 移除标签
- build_skill_catalog_prompt: 构建 skill catalog 段落
- update_pending_skills: 更新待注入 skill
- consume/inject: 消费和注入逻辑
- StructuredTagExtractor 集成: next_skills 作为 reasoning tag 被提取
"""

from unittest.mock import MagicMock

import pytest

from mem_deep_research_core.skills.inline_selector import InlineSkillSelector
from mem_deep_research_core.skills.matcher import SkillMatcher
from mem_deep_research_core.utils.stream_parsing_utils import StructuredTagExtractor

# ========== Fixtures ==========


@pytest.fixture
def mock_matcher():
    """创建一个带 mock skills 的 SkillMatcher"""
    matcher = MagicMock(spec=SkillMatcher)
    matcher.skills = {
        "search_strategy": {
            "name": "search_strategy",
            "type": "knowledge",
            "description": "搜索策略指南",
            "content": "# 搜索策略\n搜索时请遵循...",
            "metadata": {"priority": 10},
        },
        "report_writing": {
            "name": "report_writing",
            "type": "knowledge",
            "description": "报告撰写指南",
            "content": "# 报告撰写\n写报告时请...",
            "metadata": {"priority": 5},
        },
        "tanka_tool_usage": {
            "name": "tanka_tool_usage",
            "type": "tool_guide",
            "description": "Tanka 工具使用指南",
            "content": "# Tanka 使用\n使用 Tanka 时...",
            "metadata": {"priority": 8},
        },
    }
    matcher.get_skill_summaries.return_value = [
        MagicMock(name="search_strategy", description="搜索策略指南", skill_type="knowledge"),
        MagicMock(name="report_writing", description="报告撰写指南", skill_type="knowledge"),
        MagicMock(
            name="tanka_tool_usage", description="Tanka 工具使用指南", skill_type="tool_guide"
        ),
    ]
    return matcher


@pytest.fixture
def selector(mock_matcher):
    return InlineSkillSelector(mock_matcher, chinese=True)


@pytest.fixture
def selector_en(mock_matcher):
    return InlineSkillSelector(mock_matcher, chinese=False)


# ========== parse_next_skills 测试 ==========


class TestParseNextSkills:
    def test_basic_parse(self):
        text = "搜索完成了。\n<next_skills>report_writing</next_skills>"
        assert InlineSkillSelector.parse_next_skills(text) == ["report_writing"]

    def test_multiple_skills_comma(self):
        text = "<next_skills>search_strategy, report_writing</next_skills>"
        assert InlineSkillSelector.parse_next_skills(text) == ["search_strategy", "report_writing"]

    def test_multiple_skills_chinese_comma(self):
        text = "<next_skills>search_strategy，report_writing</next_skills>"
        assert InlineSkillSelector.parse_next_skills(text) == ["search_strategy", "report_writing"]

    def test_multiple_skills_space(self):
        text = "<next_skills>search_strategy report_writing</next_skills>"
        assert InlineSkillSelector.parse_next_skills(text) == ["search_strategy", "report_writing"]

    def test_with_whitespace(self):
        text = "<next_skills>  search_strategy ,  report_writing  </next_skills>"
        assert InlineSkillSelector.parse_next_skills(text) == ["search_strategy", "report_writing"]

    def test_no_tag(self):
        text = "普通的回复内容，没有 skill 声明"
        assert InlineSkillSelector.parse_next_skills(text) == []

    def test_empty_tag(self):
        text = "<next_skills></next_skills>"
        assert InlineSkillSelector.parse_next_skills(text) == []

    def test_empty_text(self):
        assert InlineSkillSelector.parse_next_skills("") == []
        assert InlineSkillSelector.parse_next_skills(None) == []

    def test_tag_in_middle_of_text(self):
        text = "我先搜索一下。\n<next_skills>report_writing</next_skills>\n好的继续。"
        assert InlineSkillSelector.parse_next_skills(text) == ["report_writing"]

    def test_multiline_tag(self):
        text = "<next_skills>\nsearch_strategy,\nreport_writing\n</next_skills>"
        assert InlineSkillSelector.parse_next_skills(text) == ["search_strategy", "report_writing"]


# ========== strip_next_skills_tag 测试 ==========


class TestStripNextSkillsTag:
    def test_strip_tag(self):
        text = "回复内容。\n<next_skills>report_writing</next_skills>"
        result = InlineSkillSelector.strip_next_skills_tag(text)
        assert result == "回复内容。"
        assert "<next_skills>" not in result

    def test_strip_no_tag(self):
        text = "普通回复"
        assert InlineSkillSelector.strip_next_skills_tag(text) == "普通回复"

    def test_strip_empty(self):
        assert InlineSkillSelector.strip_next_skills_tag("") == ""
        assert InlineSkillSelector.strip_next_skills_tag(None) is None

    def test_strip_tag_in_middle(self):
        text = "前面。<next_skills>skill1</next_skills>后面。"
        result = InlineSkillSelector.strip_next_skills_tag(text)
        assert result == "前面。后面。"


# ========== build_skill_catalog_prompt 测试 ==========


class TestBuildSkillCatalogPrompt:
    def test_chinese_catalog(self, selector):
        result = selector.build_skill_catalog_prompt()
        assert "## Skill 声明" in result
        assert "可用 skills:" in result
        assert "<next_skills>" in result
        assert "search_strategy" in result
        assert "report_writing" in result

    def test_english_catalog(self, selector_en):
        result = selector_en.build_skill_catalog_prompt()
        assert "## Skill Declaration" in result
        assert "Available skills:" in result
        assert "<next_skills>" in result

    def test_empty_skills(self, mock_matcher):
        mock_matcher.get_skill_summaries.return_value = []
        selector = InlineSkillSelector(mock_matcher, chinese=True)
        assert selector.build_skill_catalog_prompt() == ""


# ========== update_pending_skills 测试 ==========


class TestUpdatePendingSkills:
    def test_valid_skills(self, selector):
        text = "<next_skills>search_strategy, report_writing</next_skills>"
        result = selector.update_pending_skills(text)
        assert result == ["search_strategy", "report_writing"]
        assert selector.has_pending_skills()

    def test_invalid_skill_filtered(self, selector):
        text = "<next_skills>search_strategy, nonexistent_skill</next_skills>"
        result = selector.update_pending_skills(text)
        assert result == ["search_strategy"]

    def test_all_invalid(self, selector):
        text = "<next_skills>nonexistent_1, nonexistent_2</next_skills>"
        result = selector.update_pending_skills(text)
        assert result == []
        assert not selector.has_pending_skills()

    def test_no_tag(self, selector):
        result = selector.update_pending_skills("普通回复")
        assert result == []
        assert not selector.has_pending_skills()


# ========== consume_pending_skills 测试 ==========


class TestConsumePendingSkills:
    def test_consume_clears(self, selector):
        selector.update_pending_skills("<next_skills>search_strategy</next_skills>")
        consumed = selector.consume_pending_skills()
        assert consumed == ["search_strategy"]
        assert not selector.has_pending_skills()
        assert selector.consume_pending_skills() == []

    def test_consume_empty(self, selector):
        assert selector.consume_pending_skills() == []


# ========== inject_pending_skills 测试 ==========


class TestInjectPendingSkills:
    def test_inject_calls_injector(self, selector):
        selector.update_pending_skills("<next_skills>search_strategy</next_skills>")
        base = "System prompt content"
        selector.inject_pending_skills(base)
        # inject 之后 pending 被清空
        assert not selector.has_pending_skills()

    def test_inject_no_pending(self, selector):
        base = "System prompt content"
        result = selector.inject_pending_skills(base)
        assert result == base  # 没有 pending，原样返回


# ========== reset 测试 ==========


class TestReset:
    def test_reset_clears_pending(self, selector):
        selector.update_pending_skills("<next_skills>search_strategy</next_skills>")
        assert selector.has_pending_skills()
        selector.reset()
        assert not selector.has_pending_skills()


# ========== StructuredTagExtractor 集成测试 ==========


class TestStructuredTagExtractorIntegration:
    """测试 <next_skills> 作为 reasoning tag 被 StructuredTagExtractor 正确提取"""

    def test_next_skills_extracted_as_reasoning_block(self):
        extractor = StructuredTagExtractor(reasoning_tags=["thinking", "next_skills"])
        text = "搜索完成。<next_skills>report_writing</next_skills>"
        output, blocks = extractor.process(text, is_last=True)

        # <next_skills> 内容被提取为 reasoning block
        assert len(blocks) == 1
        assert blocks[0].tag_name == "next_skills"
        assert blocks[0].content == "report_writing"

        # 输出中不包含 <next_skills> 标签
        assert "<next_skills>" not in output
        assert "搜索完成。" in output

    def test_next_skills_with_other_tags(self):
        extractor = StructuredTagExtractor(reasoning_tags=["thinking", "next_skills"])
        text = "<thinking>分析中...</thinking>回复内容<next_skills>search_strategy</next_skills>"
        output, blocks = extractor.process(text, is_last=True)

        assert len(blocks) == 2
        tag_names = [b.tag_name for b in blocks]
        assert "thinking" in tag_names
        assert "next_skills" in tag_names
        assert "回复内容" in output

    def test_next_skills_streaming(self):
        """模拟流式输入，标签跨 chunk"""
        extractor = StructuredTagExtractor(reasoning_tags=["next_skills"])

        # chunk 1: 部分标签
        out1, blocks1 = extractor.process("搜索完成。<next_sk", is_last=False)
        assert len(blocks1) == 0

        # chunk 2: 标签完成
        out2, blocks2 = extractor.process("ills>report_writing</next_skills>", is_last=True)
        assert len(blocks2) == 1
        assert blocks2[0].content == "report_writing"
