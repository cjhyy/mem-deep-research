"""
Skill 情境匹配器

负责：
1. 加载 Markdown 格式的 Skill 定义
2. 根据用户查询匹配相关 Skills
3. 将 Skills 注入到系统提示词

Skill 文件格式 (Markdown + YAML Front Matter):
```markdown
---
name: skill_name
version: "1.0.0"
triggers:
  keywords: ["关键词1", "关键词2"]
  intents: ["intent1", "intent2"]
  tools_mentioned: ["tool1", "tool2"]
metadata:
  priority: 10
  tags: ["tag1", "tag2"]
---

# Skill 内容

Markdown 正文作为 Skill 内容...
```
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class SkillSummary:
    """Skill 摘要，用于 LLM 选择"""

    name: str
    skill_type: str
    description: str
    when_to_use: str


@dataclass
class MatchedSkill:
    """匹配到的 Skill"""

    name: str
    content: str
    examples: list[dict]
    priority: int
    match_score: float
    match_reasons: list[str]
    skill_type: str = "knowledge"


class SkillMatcher:
    """Skill 情境匹配器"""

    def __init__(self, skills_dir: Path):
        """
        初始化 Skill 匹配器。

        Args:
            skills_dir: Skills 目录路径，应包含 definitions/ 子目录
        """
        self.skills_dir = Path(skills_dir)
        self.skills: dict[str, dict] = {}
        self._load_skills()

    def _load_skills(self) -> None:
        """加载所有 Skill 定义 (Markdown 格式)"""
        definitions_dir = self.skills_dir / "definitions"
        if not definitions_dir.exists():
            logger.warning(f"Skills definitions dir not found: {definitions_dir}")
            return

        for md_file in definitions_dir.glob("*.md"):
            try:
                skill = self._parse_skill_markdown(md_file)
                if skill:
                    skill_name = skill.get("name", md_file.stem)
                    self.skills[skill_name] = skill
                    logger.debug(f"Loaded skill: {skill_name}")
            except Exception as e:
                logger.warning(f"Failed to load skill {md_file}: {e}")

        logger.info(f"Loaded {len(self.skills)} skills from {definitions_dir}")

    def _parse_skill_markdown(self, md_file: Path) -> dict | None:
        """
        解析 Markdown 格式的 Skill 文件。

        文件格式:
        ---
        name: skill_name
        triggers:
          keywords: [...]
        metadata:
          priority: 10
        ---

        # Skill Content

        Markdown content here...

        Args:
            md_file: Markdown 文件路径

        Returns:
            解析后的 Skill 配置字典
        """
        with open(md_file, encoding="utf-8") as f:
            content = f.read()

        # 解析 YAML Front Matter
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                try:
                    front_matter = yaml.safe_load(parts[1])
                    if front_matter is None:
                        front_matter = {}
                    markdown_content = parts[2].strip()
                    front_matter["content"] = markdown_content

                    # 兼容处理：type 默认 knowledge，when_to_use 默认取 description
                    if "type" not in front_matter:
                        front_matter["type"] = "knowledge"
                    if "when_to_use" not in front_matter:
                        front_matter["when_to_use"] = front_matter.get("description", "")

                    return front_matter
                except yaml.YAMLError as e:
                    logger.warning(f"Failed to parse YAML front matter in {md_file}: {e}")
                    return None

        # 如果没有 front matter，使用文件名作为 name，整个内容作为 content
        return {
            "name": md_file.stem,
            "content": content,
            "triggers": {},
            "metadata": {"priority": 0},
            "type": "knowledge",
            "when_to_use": "",
        }

    def match(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        tools_to_use: list[str] | None = None,
        max_skills: int = 3,
        min_score: float = 0.5,
    ) -> list[MatchedSkill]:
        """
        根据情境匹配相关 Skills。

        Args:
            query: 用户查询
            context: 上下文信息 (task_type, complexity, intent, etc.)
            tools_to_use: 计划使用的工具列表
            max_skills: 最大返回数量
            min_score: 最小匹配分数阈值

        Returns:
            按 (priority, score) 排序的匹配 Skills
        """
        matched = []
        context = context or {}
        tools_to_use = tools_to_use or []

        for skill_name, skill in self.skills.items():
            score, reasons = self._calculate_match_score(skill, query, context, tools_to_use)

            if score >= min_score:
                matched.append(
                    MatchedSkill(
                        name=skill_name,
                        content=skill.get("content", ""),
                        examples=skill.get("examples", []),
                        priority=skill.get("metadata", {}).get("priority", 0),
                        match_score=score,
                        match_reasons=reasons,
                        skill_type=skill.get("type", "knowledge"),
                    )
                )

        # 按 (priority, score) 降序排序
        matched.sort(key=lambda s: (s.priority, s.match_score), reverse=True)

        return matched[:max_skills]

    def _calculate_match_score(
        self,
        skill: dict,
        query: str,
        context: dict[str, Any],
        tools_to_use: list[str],
    ) -> tuple[float, list[str]]:
        """
        计算 Skill 匹配分数。

        评分规则:
        - 关键词匹配: +1.0 per keyword
        - 意图匹配: +2.0
        - 工具匹配: +1.5 per tool
        - 上下文条件匹配: +1.0 per condition

        Returns:
            (score, reasons)
        """
        triggers = skill.get("triggers", {})
        score = 0.0
        reasons = []

        # 处理 query 可能是列表的情况 (来自 orchestrator 的 initial_user_content)
        if isinstance(query, list):
            # 从列表中提取文本内容
            query_text = " ".join(
                item.get("text", "") if isinstance(item, dict) else str(item) for item in query
            )
        else:
            query_text = str(query) if query else ""

        query_lower = query_text.lower()

        # 1. 关键词匹配
        keywords = triggers.get("keywords", [])
        for keyword in keywords:
            if keyword.lower() in query_lower:
                score += 1.0
                reasons.append(f"keyword:{keyword}")

        # 2. 意图匹配
        intents = triggers.get("intents", [])
        user_intent = context.get("intent", "")
        if user_intent and user_intent in intents:
            score += 2.0
            reasons.append(f"intent:{user_intent}")

        # 3. 工具匹配
        tools_mentioned = triggers.get("tools_mentioned", [])
        for tool in tools_to_use:
            if tool in tools_mentioned:
                score += 1.5
                reasons.append(f"tool:{tool}")

        # 4. 上下文条件匹配
        conditions = triggers.get("context_conditions", [])
        for cond in conditions:
            field = cond.get("field")
            operator = cond.get("operator")
            expected = cond.get("value")

            ctx_value = context.get(field)
            if self._evaluate_condition(ctx_value, operator, expected):
                score += 1.0
                reasons.append(f"context:{field}={expected}")

        return score, reasons

    def _evaluate_condition(self, ctx_value: Any, operator: str, expected: Any) -> bool:
        """评估上下文条件"""
        if ctx_value is None:
            return False

        if operator == "equals":
            return ctx_value == expected
        elif operator == "in":
            return ctx_value in expected
        elif operator == "contains":
            return expected in str(ctx_value)
        elif operator == "regex":
            try:
                return bool(re.search(expected, str(ctx_value)))
            except re.error:
                logger.warning(f"[SkillMatcher] Invalid regex pattern: {expected}")
                return False
        elif operator == "gte":
            return ctx_value >= expected
        elif operator == "lte":
            return ctx_value <= expected
        elif operator == "gt":
            return ctx_value > expected
        elif operator == "lt":
            return ctx_value < expected

        return False

    def reload(self) -> None:
        """重新加载 Skills"""
        self.skills.clear()
        self._load_skills()

    def list_skills(self) -> list[str]:
        """列出所有加载的 Skill 名称"""
        return list(self.skills.keys())

    def get_skill_summaries(self) -> list[SkillSummary]:
        """返回所有 Skill 的摘要信息，供 LLM 选择器使用"""
        summaries = []
        for skill_name, skill in self.skills.items():
            summaries.append(
                SkillSummary(
                    name=skill_name,
                    skill_type=skill.get("type", "knowledge"),
                    description=skill.get("description", ""),
                    when_to_use=skill.get("when_to_use", skill.get("description", "")),
                )
            )
        return summaries


class SkillInjector:
    """Skill 内容注入器"""

    def __init__(self, matcher: SkillMatcher):
        """
        初始化 Skill 注入器。

        Args:
            matcher: SkillMatcher 实例
        """
        self.matcher = matcher

    def inject_skills(
        self,
        base_prompt: str,
        query: str,
        context: dict[str, Any] | None = None,
        tools_to_use: list[str] | None = None,
        max_skills: int = 3,
        include_examples: bool = True,
        max_examples_per_skill: int = 2,
    ) -> str:
        """
        将匹配的 Skills 注入到系统提示词中。

        Args:
            base_prompt: 基础系统提示词
            query: 用户查询
            context: 上下文
            tools_to_use: 计划使用的工具
            max_skills: 最大 Skill 数量
            include_examples: 是否包含示例
            max_examples_per_skill: 每个 Skill 最大示例数

        Returns:
            注入 Skills 后的提示词
        """
        # 匹配 Skills
        matched_skills = self.matcher.match(
            query=query,
            context=context,
            tools_to_use=tools_to_use,
            max_skills=max_skills,
        )

        if not matched_skills:
            logger.debug("No skills matched for query")
            return base_prompt

        logger.info(f"Injecting {len(matched_skills)} skills: {[s.name for s in matched_skills]}")

        # 构建 Skills 部分
        skills_section = self._build_skills_section(
            matched_skills,
            include_examples,
            max_examples_per_skill,
        )

        # 注入到提示词
        return self._inject_section(base_prompt, skills_section)

    def inject_selected_skills(
        self,
        base_prompt: str,
        skill_names: list[str],
        include_examples: bool = True,
        max_examples_per_skill: int = 2,
    ) -> str:
        """
        按 skill 名称列表直接注入 skill 内容（跳过匹配流程）。

        Args:
            base_prompt: 基础系统提示词
            skill_names: 已选择的 skill 名称列表
            include_examples: 是否包含示例
            max_examples_per_skill: 每个 Skill 最大示例数

        Returns:
            注入 Skills 后的提示词
        """
        if not skill_names:
            return base_prompt

        # 按名称查找 skill 并构建 MatchedSkill 列表
        selected_skills = []
        for name in skill_names:
            skill = self.matcher.skills.get(name)
            if skill:
                selected_skills.append(
                    MatchedSkill(
                        name=name,
                        content=skill.get("content", ""),
                        examples=skill.get("examples", []),
                        priority=skill.get("metadata", {}).get("priority", 0),
                        match_score=1.0,
                        match_reasons=["llm_selected"],
                        skill_type=skill.get("type", "knowledge"),
                    )
                )
            else:
                logger.warning(f"Selected skill '{name}' not found in loaded skills")

        if not selected_skills:
            return base_prompt

        logger.info(
            f"Injecting {len(selected_skills)} selected skills: {[s.name for s in selected_skills]}"
        )

        skills_section = self._build_skills_section(
            selected_skills, include_examples, max_examples_per_skill
        )
        return self._inject_section(base_prompt, skills_section)

    def _build_skills_section(
        self,
        skills: list[MatchedSkill],
        include_examples: bool,
        max_examples: int,
    ) -> str:
        """构建 Skills 部分内容"""
        sections = [
            "",
            "---",
            "",
            "## 相关经验指南",
            "",
            "以下是与当前任务相关的经验和最佳实践：",
            "",
        ]

        for skill in skills:
            # 添加 Skill 内容
            sections.append(skill.content)
            sections.append("")

        sections.append("---")
        return "\n".join(sections)

    def _inject_section(self, base_prompt: str, skills_section: str) -> str:
        """将 Skills 部分注入到提示词中"""
        # 策略 1: 在 "## 可用工具" 之前插入
        if "## 可用工具" in base_prompt:
            return base_prompt.replace("## 可用工具", f"{skills_section}\n\n## 可用工具")

        # 策略 2: 在 "## Available Tools" 之前插入
        if "## Available Tools" in base_prompt:
            return base_prompt.replace(
                "## Available Tools", f"{skills_section}\n\n## Available Tools"
            )

        # 策略 3: 在 "# Tools" 之前插入
        if "# Tools" in base_prompt:
            return base_prompt.replace("# Tools", f"{skills_section}\n\n# Tools")

        # 策略 4: 追加到末尾
        return f"{base_prompt}\n\n{skills_section}"


# ============ 便捷函数 ============

_skill_matcher: SkillMatcher | None = None
_skill_injector: SkillInjector | None = None


def init_skill_system(skills_dir: Path) -> tuple[SkillMatcher, SkillInjector]:
    """初始化 Skill 系统"""
    global _skill_matcher, _skill_injector
    _skill_matcher = SkillMatcher(skills_dir)
    _skill_injector = SkillInjector(_skill_matcher)
    return _skill_matcher, _skill_injector


def get_skill_matcher() -> SkillMatcher | None:
    """获取 Skill 匹配器"""
    return _skill_matcher


def get_skill_injector() -> SkillInjector | None:
    """获取 Skill 注入器"""
    return _skill_injector


def match_skills(
    query: str,
    context: dict[str, Any] | None = None,
    tools_to_use: list[str] | None = None,
    max_skills: int = 3,
) -> list[MatchedSkill]:
    """匹配 Skills 的便捷函数"""
    if _skill_matcher is None:
        return []
    return _skill_matcher.match(query, context, tools_to_use, max_skills)


def inject_skills(
    base_prompt: str,
    query: str,
    context: dict[str, Any] | None = None,
    tools_to_use: list[str] | None = None,
) -> str:
    """注入 Skills 的便捷函数"""
    if _skill_injector is None:
        return base_prompt
    return _skill_injector.inject_skills(base_prompt, query, context, tools_to_use)
