"""
统一的 Agent Prompt 类

通过模板配置实现不同类型的 prompt，无需多个类。
所有行为通过配置决定：
- 模板文件决定 prompt 内容
- tool_format 决定工具调用格式 (xml/native)
- agent_type 决定是主 Agent 还是子 Agent
- presets 决定附加的协议模块

Usage:
    # 通过配置文件
    agent:
      prompt:
        agent_type: main           # main 或 worker
        tool_format: xml           # xml 或 native
        presets: [research, time_sensitive]  # 可选的预设模块

    # 通过代码
    prompt = AgentPrompt(
        agent_type="main",
        tool_format="xml",
        presets=["research"],
    )
"""

import contextlib
import datetime
import logging
from pathlib import Path
from typing import Any

from mem_deep_research_core.prompts.template_loader import PromptTemplateLoader

logger = logging.getLogger(__name__)


class AgentPrompt:
    """
    统一的 Agent Prompt 类

    通过配置而非继承来实现不同的 prompt 行为。
    """

    def __init__(
        self,
        agent_type: str = "main",  # "main" 或 "worker"
        tool_format: str = "xml",  # "xml" 或 "native"
        presets: list[str] | None = None,  # ["research", "time_sensitive", ...]
        templates_dir: Path | None = None,
        custom_system_template: str | None = None,
        custom_summarize_template: str | None = None,
        minimal: bool = False,
        custom_takes_over: bool = False,
    ):
        """
        初始化 Agent Prompt

        Args:
            agent_type: Agent 类型 ("main" 主 Agent, "worker" 子 Agent)
            tool_format: 工具调用格式 ("xml" XML 标签格式, "native" 原生格式)
            presets: 预设模块列表，如 ["research", "time_sensitive"]
            templates_dir: 自定义模板目录（会优先于内置模板）
            custom_system_template: 自定义系统 prompt 模板名
            custom_summarize_template: 自定义总结 prompt 模板名
            minimal: 最小化 system prompt — 跳过 intro/objective，只保留日期+工具定义(如有)
            custom_takes_over: 仅在 custom_system_template 生效时有意义。
                - False（默认，向后兼容）：custom 模板替换主体后，框架仍会追加
                  presets / chinese_context / language detection 段落。
                - True：custom 模板完全接管，框架不再追加任何段落。所有默认段
                  通过占位符暴露给模板（{{system_intro}}/{{tool_format}}/
                  {{mcp_tools}}/{{objective}}/{{presets}}/{{chinese_context}}/
                  {{language_tag}}），由模板按需引用。
        """
        self.agent_type = agent_type
        self.tool_format = tool_format
        self.presets = presets or []
        self.is_main_agent = agent_type == "main"
        self.minimal = minimal
        self.custom_takes_over = custom_takes_over

        # 自定义模板名
        self.custom_system_template = custom_system_template
        self.custom_summarize_template = custom_summarize_template

        # 初始化模板加载器
        self.loader = PromptTemplateLoader(templates_dir)

    def _format_mcp_tools(self, mcp_servers: list[Any]) -> str:
        """格式化 MCP 工具列表"""
        if not mcp_servers:
            return ""

        tools_section = ""
        for server in mcp_servers:
            tools_section += f"## Server name: {server['name']}\n"

            if "tools" in server and server["tools"]:
                for tool in server["tools"]:
                    # 跳过加载失败的工具
                    if "error" in tool and "name" not in tool:
                        continue
                    tools_section += f"### Tool name: {tool['name']}\n"
                    tools_section += f"Description: {tool['description']}\n"
                    tools_section += f"Input JSON schema: {tool['schema']}\n"

        return tools_section

    def _load_base_template(self, name: str) -> str:
        """加载基础模板"""
        return self.loader.load_template(f"base/{name}")

    def _load_preset_template(self, name: str) -> str:
        """加载预设模板"""
        return self.loader.load_template(f"presets/{name}")

    def generate_system_prompt_with_mcp_tools(
        self,
        mcp_servers: list[Any],
        chinese_context: bool = False,
        task_engine_cfg: dict | None = None,
        extra_context: str = "",
        response_language: str = "auto",
        **kwargs,
    ) -> str:
        """
        生成系统 prompt

        Args:
            mcp_servers: MCP 服务器配置列表
            chinese_context: 是否使用中文语境
            task_engine_cfg: 深度研究配置
            extra_context: 额外的上下文内容
            **kwargs: 其他参数

        Returns:
            完整的系统 prompt
        """
        formatted_date = datetime.datetime.today().strftime("%Y-%m-%d")
        formatted_time = datetime.datetime.utcnow().strftime("%H:%M:%S (UTC+0)")
        mcp_tools = self._format_mcp_tools(mcp_servers)

        # 预加载默认模块内容，供 custom 模板通过占位符引用
        # {{system_intro}} - 基础介绍+日期时间
        # {{tool_format}}  - 工具调用格式说明 (xml/native)
        # {{mcp_tools}}    - 工具列表 JSON schema
        # {{objective}}    - 目标说明
        default_intro = self.loader.load_and_render(
            "base/system_intro",
            date=formatted_date,
            time=formatted_time,
        )

        default_tool_format = ""
        with contextlib.suppress(FileNotFoundError):
            default_tool_format = self._load_base_template(f"tool_format_{self.tool_format}")

        objective_template = "objective_worker" if self.agent_type == "worker" else "objective_main"
        try:
            default_objective = self._load_base_template(objective_template)
        except FileNotFoundError:
            default_objective = "# General Objective\n\nYou accomplish a given task iteratively, breaking it down into clear steps and working through them methodically."

        # mcp_tools_section：xml 格式下工具 JSON schema 的完整成品（含包装语），
        # 暴露给 custom 模板避免用户手抄样板；native 格式下为空（工具走 API tools 字段）。
        if mcp_tools and self.tool_format != "native":
            default_mcp_tools_section = (
                f"Here are the functions available in JSONSchema format:\n\n{mcp_tools}"
            )
        else:
            default_mcp_tools_section = ""

        # 归一化 chinese_context 与 response_language：
        # - chinese_context=True 等同 response_language="Chinese"（CLAUDE.md 约定）
        # - response_language="Chinese" 反向触发 chinese_context 模板
        # 两种写法在 prompt 输出上完全一致，避免用户选错字段。
        is_chinese = chinese_context or response_language == "Chinese"
        effective_response_language = "Chinese" if chinese_context else response_language

        # 预渲染 presets / chinese_context / language_tag 三段，既用于默认拼装也作为
        # custom_takes_over=True 时的占位符暴露给 custom 模板。
        presets_section = self._render_presets_section(task_engine_cfg)
        chinese_section = self._render_chinese_context_section() if is_chinese else ""
        language_section = self._render_language_tag_section(effective_response_language)

        # 构建系统 prompt
        parts = []

        # 0. minimal 模式：只保留工具定义（如有），不加 intro/objective/presets
        if self.minimal:
            if extra_context:
                parts.append(extra_context.strip())
            if mcp_tools:
                if default_tool_format:
                    parts.append(default_tool_format)
                if default_mcp_tools_section:
                    parts.append(default_mcp_tools_section)
            return "\n\n".join(parts)

        # 1. 自定义模板覆盖整个主体，通过占位符按需引用默认模块
        if self.custom_system_template:
            try:
                custom_content = self.loader.load_and_render(
                    self.custom_system_template,
                    date=formatted_date,
                    time=formatted_time,
                    system_intro=default_intro,
                    tool_format=default_tool_format,
                    mcp_tools=mcp_tools,
                    mcp_tools_section=default_mcp_tools_section,
                    objective=default_objective,
                    presets=presets_section,
                    chinese_context=chinese_section,
                    language_tag=language_section,
                )
                if extra_context:
                    custom_content = extra_context.strip() + "\n\n" + custom_content
                parts.append(custom_content)
            except FileNotFoundError:
                logger.warning(
                    f"[AgentPrompt] Custom system template '{self.custom_system_template}' not found "
                    f"in search paths: {[str(p) for p in self.loader._search_paths]}. "
                    f"Falling back to default prompt. "
                    f"If templates_dir is a relative path, ensure project_dir is set correctly."
                )

        # custom_takes_over=True：custom 模板完全接管，框架不再追加任何段落。
        # 注意：custom 模板加载失败时 parts 为空，此时会继续走默认构建（fallback）。
        if self.custom_system_template and self.custom_takes_over and parts:
            return "\n\n".join(parts)

        # 2. 默认构建（无 custom 模板或加载失败时）
        if not parts:
            if mcp_tools:
                # 有工具：完整 intro + 工具格式 + 工具列表
                intro = default_intro
                if extra_context:
                    intro = extra_context.strip() + "\n\n" + intro
                parts.append(intro)

                if default_tool_format:
                    parts.append(default_tool_format)
                # xml 模式下额外注入 JSON schema；native 模式由 API tools 字段承载
                if default_mcp_tools_section:
                    parts.append(default_mcp_tools_section)
            else:
                # 无工具：只保留日期时间，不注入工具相关描述
                date_line = f"Today is: {formatted_date}. Current time: {formatted_time}."
                if extra_context:
                    date_line = extra_context.strip() + "\n\n" + date_line
                parts.append(date_line)

            parts.append(default_objective)

        # 3. 追加 presets / chinese_context / language_tag
        # custom_takes_over=True 的早返回已在上方处理；此处覆盖两种路径：
        #   - 无 custom 模板（走默认构建）
        #   - 有 custom 模板但 custom_takes_over=False（向后兼容）
        if presets_section:
            parts.append(presets_section)
        if chinese_section:
            parts.append(chinese_section)
        if language_section:
            parts.append(language_section)

        return "\n\n".join(parts)

    def _render_presets_section(self, task_engine_cfg: dict | None) -> str:
        """渲染 presets 段（含 task_engine_cfg 自动追加的预设）。"""
        effective_presets = list(self.presets)

        if task_engine_cfg and task_engine_cfg.get("enabled", False):
            if "task_completion" not in effective_presets:
                effective_presets.append("task_completion")
            if task_engine_cfg.get("require_explicit_planning", False):
                if "task_planning" not in effective_presets:
                    effective_presets.append("task_planning")

        rendered = []
        for preset in effective_presets:
            try:
                rendered.append(self._load_preset_template(preset))
            except FileNotFoundError:
                pass
        return "\n\n".join(rendered)

    def _render_chinese_context_section(self) -> str:
        """渲染中文语境段（按 agent_type 选择模板）。"""
        chinese_template = (
            "chinese_worker" if self.agent_type == "worker" else "chinese_context"
        )
        try:
            return self._load_base_template(chinese_template)
        except FileNotFoundError:
            return ""

    def _render_language_tag_section(self, response_language: str) -> str:
        """渲染 language detection 段（仅当 response_language=auto 时生效）。"""
        if response_language != "auto":
            return ""
        return (
            "## Language\n\n"
            "On your **first reply only**, emit a `<response_language>` tag declaring "
            "the language you will use for this session, based on the user's query language:\n\n"
            "```\n<response_language>Chinese</response_language>\n```\n\n"
            "Supported values: `Chinese`, `English`, `Japanese`, `Korean`, or any other "
            "language matching the user's input. After the first reply, do not emit this tag again."
        )

    def generate_summarize_prompt(
        self,
        task_description: str,
        task_failed: bool = False,
        chinese_context: bool = False,
        target_language: str = "English",
        **kwargs,
    ) -> str:
        """
        生成总结 prompt

        Args:
            task_description: 原始任务描述
            task_failed: 任务是否失败
            chinese_context: 是否使用中文语境
            target_language: 目标语言
            **kwargs: 其他参数

        Returns:
            完整的总结 prompt
        """
        # Build the task_failed conditional block in Python instead of
        # Handlebars syntax (which render_template does not support).
        task_failed_message = ""
        if task_failed:
            task_failed_message = (
                "**Important: You have either exhausted the context token limit "
                "or reached the maximum number of interaction turns without arriving "
                "at a conclusive answer. Therefore, you failed to complete the task. "
                "You Must explicitly state that you failed to complete the task in "
                "your response.**\n"
            )

        template_vars = dict(
            task_description=task_description,
            task_failed=task_failed,
            task_failed_message=task_failed_message,
            chinese_context=chinese_context,
            target_language=target_language,
        )

        # 尝试加载自定义模板
        if self.custom_summarize_template:
            try:
                return self.loader.load_and_render(
                    self.custom_summarize_template,
                    **template_vars,
                )
            except FileNotFoundError:
                logger.warning(
                    f"[AgentPrompt] Custom summarize template '{self.custom_summarize_template}' "
                    f"not found in search paths: {[str(p) for p in self.loader._search_paths]}. "
                    f"Falling back to default summarize prompt."
                )

        # 使用基础模板
        return self.loader.load_and_render(
            "base/summarize",
            **template_vars,
        )

    def expose_agent_as_tool(self, subagent_name: str, **kwargs) -> dict:
        """
        将 Agent 暴露为工具（仅用于子 Agent）

        Args:
            subagent_name: 子 Agent 名称

        Returns:
            工具定义字典
        """
        if self.is_main_agent:
            return {}

        tool_description = self.loader.load_template("base/sub_agent_tool_description")

        return {
            "name": subagent_name,
            "tools": [
                {
                    "name": "execute_subtask",
                    "description": tool_description,
                    "schema": {
                        "type": "object",
                        "properties": {"subtask": {"title": "Subtask", "type": "string"}},
                        "required": ["subtask"],
                        "title": "execute_subtaskArguments",
                    },
                }
            ],
        }
