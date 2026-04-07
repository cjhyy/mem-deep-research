"""
Skill 技能系统

提供经验复用和自动注入能力:
- SkillCommand: 统一 Skill 数据模型（兼容 Claude Code SKILL.md + 遗留格式）
- SkillLoader: 多源 Skill 扫描与加载器
- SkillMatcher: 根据情境匹配相关 Skills
- SkillInjector: 将 Skills 注入到系统提示词或 meta message
- LLMSkillSelector: 基于 LLM 的 Skill 选择器
- InlineSkillSelector: LLM 在回复中声明下一轮 skill（零额外开销）
- SkillSummary: Skill 摘要数据结构
"""

from .inline_selector import InlineSkillSelector
from .llm_selector import LLMSkillSelector
from .matcher import MatchedSkill, SkillInjector, SkillMatcher, SkillSummary
from .skill_command import SkillCommand
from .skill_loader import SkillLoader

__all__ = [
    "SkillCommand",
    "SkillLoader",
    "SkillMatcher",
    "SkillInjector",
    "MatchedSkill",
    "SkillSummary",
    "LLMSkillSelector",
    "InlineSkillSelector",
]
