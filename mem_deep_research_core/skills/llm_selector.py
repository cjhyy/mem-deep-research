"""
LLM Skill 选择器

使用轻量 LLM 根据用户查询选择相关 Skills，
失败时 fallback 到规则匹配。
"""

import json
import logging
import os
import re

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from mem_deep_research_core.prompts.template_loader import PromptTemplateLoader
from mem_deep_research_core.skills.matcher import SkillMatcher, SkillSummary

logger = logging.getLogger(__name__)

# 模块级模板加载器
_loader = PromptTemplateLoader()


class LLMSkillSelector:
    """基于 LLM 的 Skill 选择器"""

    def __init__(
        self,
        matcher: SkillMatcher,
        api_key: str,
        base_url: str | None = None,
        model: str = "gpt-4o-mini",
        max_skills: int = 3,
        fallback_to_rules: bool = True,
    ):
        """
        初始化 LLM Skill 选择器。

        Args:
            matcher: SkillMatcher 实例（用于获取 skill 摘要和 fallback）
            api_key: OpenAI API key
            base_url: API base URL
            model: 用于选择的模型
            max_skills: 默认最大选择数量
            fallback_to_rules: LLM 失败时是否降级到规则匹配
        """
        self.matcher = matcher
        self.api_key = api_key
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = model
        self.max_skills = max_skills
        self.fallback_to_rules = fallback_to_rules

    def _build_catalog_text(self, summaries: list[SkillSummary]) -> str:
        """格式化 skill 列表供 LLM 阅读"""
        lines = []
        for s in summaries:
            lines.append(
                f"- **{s.name}** (type: {s.skill_type})\n"
                f"  描述: {s.description}\n"
                f"  使用场景: {s.when_to_use}"
            )
        return "\n".join(lines)

    def _parse_response(self, text: str) -> list[str]:
        """解析 LLM 返回的 JSON 响应，提取 skill 名称列表"""
        # 尝试直接解析 JSON
        try:
            data = json.loads(text)
            selected = data.get("selected", [])
            if isinstance(selected, list):
                return [s for s in selected if isinstance(s, str)]
        except (json.JSONDecodeError, AttributeError):
            pass

        # 尝试从文本中提取 JSON
        json_match = re.search(r'\{[^}]*"selected"\s*:\s*\[([^\]]*)\][^}]*\}', text)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                selected = data.get("selected", [])
                if isinstance(selected, list):
                    return [s for s in selected if isinstance(s, str)]
            except (json.JSONDecodeError, AttributeError):
                pass

        # 最后尝试从引号中提取 skill 名称
        names = re.findall(r'"([a-zA-Z_][a-zA-Z0-9_]*)"', text)
        # 过滤掉已知的 JSON key
        known_keys = {"selected", "type", "name"}
        return [n for n in names if n not in known_keys]

    async def select(
        self,
        query: str,
        tool_names: list[str] | None = None,
        max_skills: int | None = None,
        context: dict | None = None,
    ) -> list[str]:
        """
        使用 LLM 选择相关 Skills。

        Args:
            query: 用户查询
            tool_names: 当前可用的工具名称列表
            max_skills: 最大选择数量
            context: 上下文信息（fallback 时使用）

        Returns:
            选中的 skill 名称列表
        """
        if not self.matcher.skills:
            return []

        max_skills = max_skills or self.max_skills
        summaries = self.matcher.get_skill_summaries()

        if not summaries:
            return []

        # 处理 query 可能是列表的情况
        if isinstance(query, list):
            query_text = " ".join(
                item.get("text", "") if isinstance(item, dict) else str(item) for item in query
            )
        else:
            query_text = str(query) if query else ""

        try:
            selected = await self._llm_select(query_text, summaries, tool_names or [], max_skills)
            # 过滤掉不存在的 skill 名称
            valid_names = set(self.matcher.skills.keys())
            selected = [s for s in selected if s in valid_names]
            logger.info(f"LLM skill selection result: {selected}")
            return selected[:max_skills]
        except Exception as e:
            logger.warning(f"LLM skill selection failed: {e}")
            if self.fallback_to_rules:
                logger.info("Falling back to rule-based skill matching")
                matched = self.matcher.match(
                    query=query,
                    context=context,
                    tools_to_use=tool_names,
                    max_skills=max_skills,
                )
                return [s.name for s in matched]
            return []

    @retry(
        wait=wait_exponential(multiplier=2, max=10),
        stop=stop_after_attempt(2),
    )
    async def _llm_select(
        self,
        query_text: str,
        summaries: list[SkillSummary],
        tool_names: list[str],
        max_skills: int,
    ) -> list[str]:
        """执行 LLM 调用进行 skill 选择"""
        client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=30,
        )

        catalog_text = self._build_catalog_text(summaries)
        tool_names_text = ", ".join(tool_names) if tool_names else "无"

        prompt = _loader.load_and_render(
            "skills/select_skills",
            skill_catalog=catalog_text,
            query=query_text[:1000],  # 截断过长查询
            tool_names=tool_names_text,
            max_skills=str(max_skills),
        )

        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )

        if not response.choices:
            logger.warning("LLM skill selection returned empty choices")
            return []
        result = response.choices[0].message.content
        if not result or not result.strip():
            logger.warning("LLM skill selection returned empty result")
            return []

        return self._parse_response(result)
