"""
Prompt Builder 模块

从 Orchestrator 提取的 prompt 构建逻辑，负责：
- 系统提示词构建
- 任务指导生成
- Skill 选择与注入
- Hint 生成
"""

import logging
from typing import Any

from mem_deep_research_core.core.constants import parse_bool_config
from mem_deep_research_core.core.hooks import HookContext, hooks
from mem_deep_research_core.prompts.template_loader import PromptTemplateLoader
from mem_deep_research_core.utils.external_loader import external_loader
from mem_deep_research_core.utils.tool_utils import _load_agent_prompt

logger = logging.getLogger("mem_deep_research")


class PromptBuilder:
    """Prompt 构建器 — 负责 system prompt、skill 注入、hint 生成

    支持 section 级缓存：将 system prompt 拆为静态段（agent 角色、工具描述等）
    和动态段（skill 注入、hook 修改等）。静态段在 session 内缓存不变，
    动态段每次调用重算。这对 Anthropic API prompt caching 友好——
    保持静态前缀不变可大幅提升 cache hit rate。
    """

    # Section 缓存边界标记（不出现在 prompt 中，仅内部使用）
    _DYNAMIC_BOUNDARY = "__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__"

    def __init__(
        self,
        cfg,
        context: dict[str, Any],
        chinese_context: bool,
        inline_skill_selector=None,
    ):
        self.cfg = cfg
        self.context = context
        self.chinese_context = chinese_context
        self.inline_skill_selector = inline_skill_selector
        self._template_loader = PromptTemplateLoader()

        # Section cache: 静态段缓存
        self._section_cache: dict[str, str] = {}
        # 缓存的完整 system prompt（静态部分）
        self._cached_static_prompt: str | None = None
        # 上次使用的 prompt 实例和 task_engine_cfg
        self._cached_prompt_instance = None
        self._cached_task_engine_cfg = None

    def build_task_guidance(self) -> str:
        """构建任务指导"""
        if self.chinese_context:
            return self._template_loader.load_template("guidance/task_guidance_chinese")
        return ""

    async def generate_hints(self, task_description: str) -> str:
        """生成任务提示"""
        from mem_deep_research_core.utils.summary_utils import extract_hints

        if not self.cfg.main_agent.input_process.hint_generation:
            return ""

        add_message_id = parse_bool_config(self.cfg.main_agent.get("add_message_id", False))

        try:
            hint_content = await extract_hints(
                task_description,
                self.cfg.main_agent.openai_api_key,
                self.chinese_context,
                add_message_id,
                self.cfg.main_agent.input_process.get(
                    "hint_llm_base_url", "https://api.openai.com/v1"
                ),
            )
            return self._template_loader.load_and_render(
                "hints/hint_prefix",
                hint_content=hint_content,
            )
        except Exception as e:
            logger.error(f"Hint generation failed: {e}", exc_info=True)
            return ""

    async def select_skills(
        self,
        query,
        tool_definitions: list,
    ) -> list[str] | None:
        """使用 LLM 选择相关 Skills

        inline 模式下：
        - progressive=True（默认）：第一轮仅注入 catalog，不注入完整 skill
        - progressive=False：第一轮用规则匹配 fallback 注入完整 skill

        Args:
            query: 用户查询（str 或 list）
            tool_definitions: 工具定义列表

        Returns:
            选中的 skill 名称列表，如果 LLM selector 不可用则返回 None
        """
        # inline 模式：不做额外 LLM 调用
        if self.inline_skill_selector:
            first_turn = getattr(self.inline_skill_selector, "_first_turn", True)
            if first_turn and not self.inline_skill_selector.progressive:
                # 传统模式第一轮：用规则匹配返回 skill 名称
                tool_names = [t.get("name", "") for t in tool_definitions if isinstance(t, dict)]
                query_text = query if isinstance(query, str) else str(query)
                return self.inline_skill_selector.get_first_turn_injection(
                    query=query_text, context=self.context, tools=tool_names
                )
            # progressive 模式或非第一轮：catalog 已在 prompt 中，等 LLM 声明
            return None

        try:
            selector = external_loader.get_llm_skill_selector(self.cfg)
            if not selector:
                return None

            tool_names = [t.get("name", "") for t in tool_definitions if isinstance(t, dict)]
            selected = await selector.select(
                query=query,
                tool_names=tool_names,
                context=self.context,
            )
            logger.info(f"LLM selected skills: {selected}")
            return selected
        except Exception as e:
            logger.warning(f"Skill selection failed: {e}")
            return None

    def build_system_prompt(
        self, tool_definitions, initial_user_content, selected_skill_names=None
    ):
        """构建系统提示词

        使用 section 级缓存：
        - 静态段（agent 角色 + 工具描述）：session 内缓存
        - 动态段（skill 注入、hook 修改）：每次重算
        这样 Anthropic API 的 prompt caching 可以命中静态前缀。
        """
        # 获取 prompt 配置
        prompt_cfg = {}
        if hasattr(self.cfg.main_agent, "prompt") and self.cfg.main_agent.prompt:
            prompt_cfg = dict(self.cfg.main_agent.prompt)

        # task_engine 配置转换为 presets
        task_engine_cfg = getattr(self.cfg.main_agent, "task_engine", None)
        if task_engine_cfg is not None:
            task_engine_cfg = dict(task_engine_cfg)

        main_agent_prompt_instance = _load_agent_prompt(prompt_cfg)

        # === 静态段：agent 角色 + 工具描述（缓存） ===
        if self._cached_static_prompt is None:
            extra_context = ""
            static_prompt = main_agent_prompt_instance.generate_system_prompt_with_mcp_tools(
                mcp_servers=tool_definitions,
                chinese_context=self.chinese_context,
                extra_context=extra_context,
                task_engine_cfg=task_engine_cfg,
            )
            self._cached_static_prompt = static_prompt
            self._cached_prompt_instance = main_agent_prompt_instance
            self._cached_task_engine_cfg = task_engine_cfg
            self._section_cache["base"] = static_prompt
            logger.debug("[PromptBuilder] Static section cached")
        else:
            static_prompt = self._cached_static_prompt
            main_agent_prompt_instance = self._cached_prompt_instance
            task_engine_cfg = self._cached_task_engine_cfg

        # Insert dynamic boundary between static and dynamic sections
        system_prompt = static_prompt + f"\n{self._DYNAMIC_BOUNDARY}\n"

        # === 动态段：skill 注入（每次重算） ===
        skill_cfg = self.cfg.main_agent.get("skill_selection", {})
        skill_enabled = skill_cfg.get("enabled", True) if skill_cfg is not None else True
        if not skill_enabled:
            pass  # skill_selection.enabled=false — 跳过所有 skill 注入
        elif self.inline_skill_selector:
            # Inline 模式：追加 skill catalog 到 system prompt（告诉 LLM 可用 skill 列表）
            catalog_prompt = self.inline_skill_selector.build_skill_catalog_prompt()
            if catalog_prompt:
                system_prompt += catalog_prompt
                logger.debug("Inline skill catalog appended to system prompt")
            # 非渐进模式第一轮：同时注入规则匹配到的完整 skill 内容
            if selected_skill_names:
                system_prompt = self.inline_skill_selector.injector.inject_selected_skills(
                    base_prompt=system_prompt,
                    skill_names=selected_skill_names,
                )
                logger.debug(f"Inline legacy first-turn skills injected: {selected_skill_names}")
        else:
            skill_injector = external_loader.get_skill_injector()
            if skill_injector and initial_user_content:
                if selected_skill_names is not None:
                    # LLM 选择路径：按名称直接注入
                    system_prompt = skill_injector.inject_selected_skills(
                        base_prompt=system_prompt,
                        skill_names=selected_skill_names,
                    )
                    logger.debug(f"LLM-selected skills injected: {selected_skill_names}")
                else:
                    # Fallback 路径：规则匹配注入
                    tools_to_use = [
                        t.get("name", "") for t in tool_definitions if isinstance(t, dict)
                    ]
                    system_prompt = skill_injector.inject_skills(
                        base_prompt=system_prompt,
                        query=initial_user_content,
                        context=self.context,
                        tools_to_use=tools_to_use,
                    )
                    logger.debug("Rule-matched skills injected into system prompt")

        # Hook: on_system_prompt_build — post-process system prompt (dynamic)
        hook_result = hooks.call(
            "on_system_prompt_build",
            HookContext(
                hook_name="on_system_prompt_build",
                context=self.context,
                result=system_prompt,
            ),
        )
        if isinstance(hook_result, str):
            system_prompt = hook_result

        return system_prompt, main_agent_prompt_instance, task_engine_cfg

    def build_skill_meta_message(
        self, skill_names: list[str], skill_commands: dict | None = None
    ) -> dict | None:
        """构建 isMeta user message 用于 skill 内容注入（Claude Code 模式）。

        Args:
            skill_names: 要注入的 skill 名称列表
            skill_commands: SkillCommand 字典

        Returns:
            isMeta user message dict, 或 None
        """
        injector = getattr(self.inline_skill_selector, "injector", None)
        if injector:
            return injector.build_meta_message(skill_names, skill_commands)
        return None

    def invalidate_cache(self):
        """手动使缓存失效（工具定义变更等场景使用）"""
        self._cached_static_prompt = None
        self._section_cache.clear()
        logger.debug("[PromptBuilder] Section cache invalidated")
