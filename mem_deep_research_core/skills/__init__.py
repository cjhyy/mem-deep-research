"""
Skill 技能系统

提供经验复用和自动注入能力:
- SkillMatcher: 根据情境匹配相关 Skills
- SkillInjector: 将 Skills 注入到系统提示词
- LLMSkillSelector: 基于 LLM 的 Skill 选择器
- InlineSkillSelector: LLM 在回复中声明下一轮 skill（零额外开销）
- SkillSummary: Skill 摘要数据结构
"""

from .inline_selector import InlineSkillSelector
from .llm_selector import LLMSkillSelector
from .matcher import MatchedSkill, SkillInjector, SkillMatcher, SkillSummary

__all__ = [
    "SkillMatcher",
    "SkillInjector",
    "MatchedSkill",
    "SkillSummary",
    "LLMSkillSelector",
    "InlineSkillSelector",
]
