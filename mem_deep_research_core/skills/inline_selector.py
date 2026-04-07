"""
Inline Skill 选择器

让 LLM 在每轮回复中顺便输出下一轮需要的 skill，零额外开销。

流程（progressive=True，默认）：
1. 第一轮：仅注入 skill catalog（名称+描述，~10 tokens/skill）
2. LLM 声明 <next_skills>skill1, skill2</next_skills>
3. 下一轮：注入声明的 skill 的完整内容
4. 如果 LLM 未声明任何 skill，不注入任何内容

流程（progressive=False，传统模式）：
1. 第一轮：用规则匹配兜底，注入完整 skill 内容
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
from typing import TYPE_CHECKING

from mem_deep_research_core.skills.matcher import SkillInjector, SkillMatcher

if TYPE_CHECKING:
    from mem_deep_research_core.skills.skill_command import SkillCommand

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

    def __init__(
        self,
        matcher: SkillMatcher,
        chinese: bool = False,
        progressive: bool = True,
        skill_commands: "dict[str, SkillCommand] | None" = None,
        catalog_budget_chars: int = 8000,
        description_max_chars: int = 250,
    ):
        """
        Args:
            matcher: SkillMatcher 实例
            chinese: 是否使用中文提示
            progressive: 是否启用渐进加载（默认 True）。
            skill_commands: 统一格式的 SkillCommand 字典（优先于 matcher）
            catalog_budget_chars: catalog 总字符预算（默认 8000，约 context 1%）
            description_max_chars: 单个 skill 描述最大字符数
        """
        self.matcher = matcher
        self.injector = SkillInjector(matcher)
        self.chinese = chinese
        self.progressive = progressive
        self._skill_commands = skill_commands or {}
        self._catalog_budget_chars = catalog_budget_chars
        self._description_max_chars = description_max_chars
        # 当前轮 LLM 声明的下一轮 skill 名称
        self._pending_skills: list[str] = []
        # 跟踪是否为第一轮（尚未收到任何 LLM 回复）
        self._first_turn = True
        # 条件激活: 已触碰的文件列表
        self._touched_files: list[str] = []

    def set_touched_files(self, files: list[str]) -> None:
        """更新已触碰文件列表，用于条件激活 paths 匹配"""
        self._touched_files = files

    def _get_visible_skills(self) -> list[tuple[str, str]]:
        """获取当前可见的 skill 列表: [(name, description), ...]

        优先使用 SkillCommand（支持 paths 条件激活和 disable_model_invocation），
        回退到 matcher summaries。
        """
        if self._skill_commands:
            entries = []
            for name, sc in self._skill_commands.items():
                if sc.disable_model_invocation:
                    continue
                if not sc.matches_paths(self._touched_files):
                    continue
                desc = sc.get_catalog_entry(self._description_max_chars)
                entries.append((name, desc))
            return entries

        # Fallback: use matcher summaries
        summaries = self.matcher.get_skill_summaries()
        return [(s.name, s.description[: self._description_max_chars]) for s in summaries]

    def build_skill_catalog_prompt(self) -> str:
        """
        构建 skill catalog 段落，追加到 system prompt 中。

        Budget-aware: 总字符不超过 catalog_budget_chars，单个描述不超过 description_max_chars。
        """
        entries = self._get_visible_skills()
        if not entries:
            return ""

        # Budget-aware: 如果总描述超过预算，截断描述
        lines = []
        total_chars = 0
        for name, desc in entries:
            line = f"- **{name}**: {desc}"
            if total_chars + len(line) > self._catalog_budget_chars:
                # 超出预算，只显示名称
                line = f"- **{name}**"
                if total_chars + len(line) > self._catalog_budget_chars:
                    break  # 连名称都放不下了
            total_chars += len(line) + 1  # +1 for newline
            lines.append(line)

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

    def get_first_turn_injection(self, query: str, context: dict, tools: list[str]) -> list[str]:
        """
        第一轮 skill 注入策略。

        progressive=True: 返回空列表（catalog 已在 system prompt 中，无需注入完整 skill）。
        progressive=False: 用规则匹配兜底，返回匹配到的 skill 名称列表。

        Args:
            query: 用户查询
            context: 上下文信息
            tools: 计划使用的工具列表

        Returns:
            需要注入完整内容的 skill 名称列表
        """
        if self.progressive:
            # 渐进模式：catalog 已在 system prompt 中，第一轮不注入完整 skill
            return []
        else:
            # 传统模式：用规则匹配，返回匹配到的 skill 名称
            matched = self.matcher.match(query=query, context=context, tools_to_use=tools)
            return [s.name for s in matched]

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
        all_known = set(self.matcher.skills.keys()) | set(self._skill_commands.keys())
        valid_names = [n for n in raw_names if n in all_known]

        if raw_names and not valid_names:
            logger.warning(
                f"[InlineSkill] LLM requested skills {raw_names} but none are valid. "
                f"Available: {sorted(all_known)}"
            )
        elif valid_names:
            logger.info(f"[InlineSkill] Next turn skills: {valid_names}")

        self._pending_skills = valid_names
        self._first_turn = False  # 收到 LLM 回复后不再是第一轮
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
        self._first_turn = True
