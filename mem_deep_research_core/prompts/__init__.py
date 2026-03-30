"""
Prompt 模块

统一的 AgentPrompt 类，通过配置决定行为。

使用方式:
    from mem_deep_research_core.prompts import AgentPrompt

    # 基本 Agent
    prompt = AgentPrompt(agent_type="main", tool_format="xml")

    # 研究型 Agent
    prompt = AgentPrompt(
        agent_type="main",
        tool_format="xml",
        presets=["research", "time_sensitive"],
    )

    # 子 Agent
    worker = AgentPrompt(agent_type="worker", tool_format="xml")
"""

from mem_deep_research_core.prompts.agent_prompt import AgentPrompt
from mem_deep_research_core.prompts.template_loader import (
    BUILTIN_TEMPLATES_DIR,
    PromptTemplateLoader,
)

__all__ = [
    "AgentPrompt",
    "PromptTemplateLoader",
    "BUILTIN_TEMPLATES_DIR",
]
