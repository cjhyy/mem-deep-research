"""
Inline Skill 选择器

让 LLM 在每轮回复中顺便输出下一轮需要的 skill，零额外开销。

流程：
1. 第一轮：用规则匹配兜底
2. 之后每轮：从 LLM 回复中解析 <next_skills> 标签
3. 下一轮自动注入对应 skill 的 knowledge 到 system prompt

用法：
    selector = InlineSkillSelector(matcher)

    # 构建 skill catalog（追加到 system prompt）
    catalog_section = selector.build_skill_catalog_prompt()

    # 从 LLM 回复中解析 <next_skills>
    skill_names = selector.parse_next_skills(response_text)

    # 注入选中的 skill
    system_prompt = selector.inject_skills(system_prompt, skill_names)
"""

import logging
import re

from mem_deep_research_core.skills.matcher import SkillInjector, SkillMatcher

logger = logging.getLogger(__name__)

# 匹配 <next_skills>skill1, skill2</next_skills>
NEXT_SKILLS_PATTERN = re.compile(
    r"<next_skills>\s*(.*?)\s*</next_skills>",
    re.DOTALL,
)


class InlineSkillSelector:
    """
    Inline Skill 选择器

    让 LLM 在每轮回复中通过 <next_skills> 标签声明下一轮需要的 skill，
    零额外 LLM 调用开销。
    """

    # <next_skills> 标签名，用于 StructuredTagExtractor 提取
    TAG_NAME = "next_skills"

    def __init__(self, matcher: SkillMatcher, chinese: bool = False):
        """
        Args:
            matcher: SkillMatcher 实例
            chinese: 是否使用中文提示
        """
        self.matcher = matcher
        self.injector = SkillInjector(matcher)
        self.chinese = chinese
        # 当前轮 LLM 声明的下一轮 skill 名称
        self._pending_skills: list[str] = []

    def build_skill_catalog_prompt(self) -> str:
        """
        构建 skill catalog 段落，追加到 system prompt 中。

        告诉 LLM 有哪些可用 skill，并指导它用 <next_skills> 声明。
        """
        summaries = self.matcher.get_skill_summaries()
        if not summaries:
            return ""

        lines = []
        for s in summaries:
            lines.append(f"- **{s.name}**: {s.description}")
        catalog = "\n".join(lines)

        if self.chinese:
            return (
                "\n\n## Skill 声明\n\n"
                "可用 skills:\n"
                f"{catalog}\n\n"
                "如果你判断**下一步**需要某个 skill 的指导，"
                "请在回复末尾输出：\n"
                "<next_skills>skill_name_1, skill_name_2</next_skills>\n\n"
                "如果不需要任何 skill，不用输出此标签。\n"
            )
        else:
            return (
                "\n\n## Skill Declaration\n\n"
                "Available skills:\n"
                f"{catalog}\n\n"
                "If you determine that the **next step** needs guidance from a skill, "
                "output at the end of your response:\n"
                "<next_skills>skill_name_1, skill_name_2</next_skills>\n\n"
                "If no skill is needed, do not output this tag.\n"
            )

    @staticmethod
    def parse_next_skills(text: str) -> list[str]:
        """
        从 LLM 回复中解析 <next_skills> 标签。

        Args:
            text: LLM 完整回复文本

        Returns:
            解析出的 skill 名称列表
        """
        if not text:
            return []

        match = NEXT_SKILLS_PATTERN.search(text)
        if not match:
            return []

        raw = match.group(1).strip()
        if not raw:
            return []

        # 支持逗号分隔和空格分隔
        names = [n.strip() for n in re.split(r"[,，\s]+", raw) if n.strip()]
        return names

    @staticmethod
    def strip_next_skills_tag(text: str) -> str:
        """
        从文本中移除 <next_skills> 标签及内容。

        用于清理 LLM 回复，避免标签内容显示给用户。
        """
        if not text:
            return text
        return NEXT_SKILLS_PATTERN.sub("", text).rstrip()

    def update_pending_skills(self, response_text: str) -> list[str]:
        """
        从 LLM 回复中解析并更新待注入的 skill 列表。

        Args:
            response_text: LLM 完整回复

        Returns:
            解析出的 skill 名称列表（仅存在于 matcher 中的）
        """
        raw_names = self.parse_next_skills(response_text)
        valid_names = [n for n in raw_names if n in self.matcher.skills]

        if raw_names and not valid_names:
            logger.warning(
                f"[InlineSkill] LLM requested skills {raw_names} but none are valid. "
                f"Available: {list(self.matcher.skills.keys())}"
            )
        elif valid_names:
            logger.info(f"[InlineSkill] Next turn skills: {valid_names}")

        self._pending_skills = valid_names
        return valid_names

    def get_pending_skills(self) -> list[str]:
        """获取待注入的 skill 名称列表"""
        return self._pending_skills

    def has_pending_skills(self) -> bool:
        """是否有待注入的 skill"""
        return bool(self._pending_skills)

    def consume_pending_skills(self) -> list[str]:
        """
        获取并清空待注入的 skill 列表。

        Returns:
            skill 名称列表
        """
        skills = self._pending_skills
        self._pending_skills = []
        return skills

    def inject_pending_skills(self, base_prompt: str) -> str:
        """
        将待注入的 skill 内容注入到 system prompt 中。

        注入后自动清空 pending 列表。

        Args:
            base_prompt: 基础 system prompt

        Returns:
            注入 skill 后的 system prompt
        """
        skill_names = self.consume_pending_skills()
        if not skill_names:
            return base_prompt

        return self.injector.inject_selected_skills(
            base_prompt=base_prompt,
            skill_names=skill_names,
        )

    def reset(self):
        """重置状态"""
        self._pending_skills = []
