"""
Orchestrator 模块

协调 Agent 执行流程的核心模块，负责：
- 主 Agent 执行循环
- 工具调用协调
- 子 Agent 调度
- 流式输出管理
"""

import asyncio
import logging
import os
import re
import time
import uuid
from collections.abc import Callable
from typing import Any

from omegaconf import DictConfig

from mem_deep_research_core.core.answer_handler import post_process_final_answer
from mem_deep_research_core.core.context_manager import ContextManager, ContextManagerConfig
from mem_deep_research_core.core.hooks import HookContext, hooks
from mem_deep_research_core.core.interceptor_config import InterceptorConfig, InterceptorPresets
from mem_deep_research_core.core.llm_call_handler import (
    LLMCallHandler,
    SummaryHandler,
)
from mem_deep_research_core.core.main_loop import MainLoopRunner
from mem_deep_research_core.core.prompt_builder import PromptBuilder
from mem_deep_research_core.core.message_interceptor import MessageInterceptorHandler
from mem_deep_research_core.core.monitoring import (
    ExecutionMonitor,
    MonitoringConfig,
)

# 导入拆分后的模块
from mem_deep_research_core.core.stream_handler import StreamHandler
from mem_deep_research_core.core.sub_agent_runner import SubAgentRunner
from mem_deep_research_core.core.task_planner import TaskPlanner
from mem_deep_research_core.core.tool_executor import ToolExecutor
from mem_deep_research_core.core.tool_result_formatter import ToolResultFormatter
from mem_deep_research_core.llm.provider_client_base import LLMProviderClientBase
from mem_deep_research_core.mem_deep_research_logging.task_tracer import TaskTracer
from mem_deep_research_core.tool.manager import ToolManager
from mem_deep_research_core.utils.external_loader import external_loader
from mem_deep_research_core.utils.io_utils import OutputFormatter, process_input
from mem_deep_research_core.utils.stream_parsing_utils import TextInterceptor
from mem_deep_research_core.utils.tool_utils import expose_sub_agents_as_tools

from mem_deep_research_core.core.constants import (
    DEFAULT_SCRAPE_MAX_LENGTH,
    FALLBACK_LOOP_TERMINATED,
    RECENT_TOOL_LOOKBACK,
    SYSTEM_MESSAGE_KEYWORDS,
    ensure_dict,
    generate_message_id,
    parse_bool_config,
)


# ========== 默认钩子实现 ==========


def _default_on_agent_start(ctx: HookContext):
    """Agent 开始执行 - 默认实现"""
    logger.debug(
        f"[Hook] on_agent_start: query={ctx.query}, agent_type={ctx.extra.get('agent_type')}"
    )
    return None


def _default_on_agent_end(ctx: HookContext):
    """Agent 执行完成 - 默认实现"""
    logger.debug(f"[Hook] on_agent_end: query={ctx.query}, turns_used={ctx.turn_number}")
    return None


def _default_on_turn_start(ctx: HookContext):
    """轮次开始 - 默认实现"""
    logger.debug(f"[Hook] on_turn_start: turn={ctx.turn_number}")
    return None


def _default_on_turn_end(ctx: HookContext):
    """轮次结束 - 默认实现"""
    logger.debug(
        f"[Hook] on_turn_end: turn={ctx.turn_number}, tool_calls_count={ctx.tool_calls_count}"
    )
    return None


hooks.set_default("on_agent_start", _default_on_agent_start)
hooks.set_default("on_agent_end", _default_on_agent_end)
hooks.set_default("on_turn_start", _default_on_turn_start)
hooks.set_default("on_turn_end", _default_on_turn_end)

logger = logging.getLogger("mem_deep_research")


def _list_tools(sub_agent_tool_managers: dict[str, ToolManager]):
    """创建带缓存的工具列表获取函数"""
    cache = None

    async def wrapped():
        nonlocal cache
        if cache is None:
            if not sub_agent_tool_managers:
                result = {}
            else:
                result = {
                    name: await tool_manager.get_all_tool_definitions()
                    for name, tool_manager in sub_agent_tool_managers.items()
                }
            cache = result
        return cache

    return wrapped


class Orchestrator:
    """Agent 执行协调器"""

    def __init__(
        self,
        main_agent_tool_manager: ToolManager,
        sub_agent_tool_managers: dict[str, ToolManager],
        llm_client: LLMProviderClientBase,
        output_formatter: OutputFormatter,
        cfg: DictConfig,
        task_log: TaskTracer,
        sub_agent_llm_client: LLMProviderClientBase | None = None,
        stream_queue: Any | None = None,
        tool_definitions: list[dict[str, Any]] | None = None,
        sub_agent_tool_definitions: dict[str, list[dict[str, Any]]] | None = None,
        context: dict[str, Any] | None = None,
    ):
        # 基础组件
        self.main_agent_tool_manager = main_agent_tool_manager
        self.sub_agent_tool_managers = sub_agent_tool_managers
        self.llm_client = llm_client
        self.sub_agent_llm_client = sub_agent_llm_client or llm_client
        self.output_formatter = output_formatter
        self.cfg = cfg
        self.task_log = task_log
        self.stream_queue = stream_queue
        self.tool_definitions = tool_definitions
        self.sub_agent_tool_definitions = sub_agent_tool_definitions
        self.context = context or {}

        # 缓存函数
        self._list_sub_agent_tools = _list_tools(sub_agent_tool_managers)

        # 语言配置：response_language 优先，chinese_context 向后兼容
        self.response_language = self.cfg.main_agent.get("response_language", "auto")
        chinese_context_val = parse_bool_config(self.cfg.main_agent.get("chinese_context", False))
        if chinese_context_val and self.response_language == "auto":
            # 旧配置 chinese_context=true → 等同 response_language="Chinese"
            self.response_language = "Chinese"
        # chinese_context 内部标志：当 response_language 明确为 Chinese 时为 True
        self.chinese_context = self.response_language == "Chinese"

        self.add_message_id = parse_bool_config(self.cfg.main_agent.get("add_message_id", False))
        logger.info(f"add_message_id: {self.add_message_id}")

        # 传递 task_log 给 LLM 客户端
        if self.llm_client and task_log:
            self.llm_client.task_log = task_log
        if self.sub_agent_llm_client and task_log and self.sub_agent_llm_client != self.llm_client:
            self.sub_agent_llm_client.task_log = task_log

        # 设置上下文到工具管理器
        if self.context and self.context.get("user_id"):
            self.main_agent_tool_manager.set_context(self.context)
            for sub_manager in self.sub_agent_tool_managers.values():
                sub_manager.set_context(self.context)

        # ========== 初始化组合模块 ==========
        self._init_modules()

    def _init_modules(self):
        """初始化各个组合模块"""
        self._init_stream_and_interceptor()
        self._init_monitoring_and_tools()
        self._init_llm_and_summary()
        self._init_skills_and_prompt()
        self._init_context_manager()
        self.current_agent_id: str | None = None

    def _init_stream_and_interceptor(self):
        """初始化流式处理器和消息拦截器"""
        # 流式处理器
        self.stream_handler = StreamHandler(self.stream_queue)

        # 工具结果格式化器
        self.tool_formatter = ToolResultFormatter(self.context)

        # 消息拦截处理器
        interceptor_config = self._load_interceptor_config()
        self.key_message_interceptor = TextInterceptor(
            interceptor_config.get_all_filter_keywords(),
            reasoning_tags=interceptor_config.reasoning_tags,
        )
        self.message_interceptor = MessageInterceptorHandler(
            config=interceptor_config,
            stream_reasoning_callback=self.stream_handler.stream_reasoning,
            stream_tool_call_callback=self.stream_handler.stream_tool_call,
            stream_message_callback=self.stream_handler.stream_message,
            context=self.context,
        )

        # 最终消息拦截器
        self._final_message_interceptor = TextInterceptor(["<use_mcp_tool>"])

    def _init_monitoring_and_tools(self):
        """初始化执行监控器、工具执行器和子 Agent 运行器"""
        # 执行监控器（从配置读取）
        monitoring_cfg_dict = ensure_dict(self.cfg.main_agent.get("monitoring", {}))
        if monitoring_cfg_dict:
            try:
                from mem_deep_research_core.config_schema import MonitoringConfigSchema
                monitoring_schema = MonitoringConfigSchema(**monitoring_cfg_dict)
                monitoring_config = MonitoringConfig.from_schema(monitoring_schema)
            except Exception as e:
                logger.warning(f"[Orchestrator] Invalid monitoring config, using defaults: {e}")
                monitoring_config = MonitoringConfig()
        else:
            monitoring_config = MonitoringConfig()
        self.monitor = ExecutionMonitor(
            config=monitoring_config,
            stream_reasoning_callback=self._stream_tool_reasoning,  # 保留：有自定义逻辑
        )

        # 工具执行器
        self.tool_executor = ToolExecutor(
            tool_manager=self.main_agent_tool_manager,
            output_formatter=self.output_formatter,
            tool_result_formatter=self.tool_formatter,
            context=self.context,
            stream_tool_call=self.stream_handler.stream_tool_call,
            stream_tool_reasoning=self._stream_tool_reasoning,  # 保留：有自定义逻辑
            stream_usage_info=self.stream_handler.stream_usage_info,
        )
        # scrape_max_length: 优先从配置读，fallback 环境变量，最后默认 20000
        scrape_max_length = monitoring_cfg_dict.get("scrape_max_length", None)
        if scrape_max_length is None:
            try:
                scrape_max_length = int(os.getenv("SCRAPE_MAX_LENGTH", str(DEFAULT_SCRAPE_MAX_LENGTH)))
            except (ValueError, TypeError):
                scrape_max_length = DEFAULT_SCRAPE_MAX_LENGTH
        self.tool_executor.set_scrape_max_length(scrape_max_length)

        # 子 Agent 运行器
        self.sub_agent_runner = SubAgentRunner(
            sub_agent_tool_managers=self.sub_agent_tool_managers,
            sub_agent_llm_client=self.sub_agent_llm_client,
            output_formatter=self.output_formatter,
            cfg=self.cfg,
            task_log=self.task_log,
            context=self.context,
            chinese_context=self.chinese_context,
            response_language=self.response_language,
            stream_handler=self.stream_handler,
            stream_tool_reasoning=self._stream_tool_reasoning,
            handle_llm_call=self._handle_llm_call_with_logging,
            handle_summary=self._handle_summary_with_context_limit_retry,
            intercept_key_message=self._intercept_key_message,
            streaming_final_message=self._streaming_final_message,
        )
        if self.sub_agent_tool_definitions:
            self.sub_agent_runner.set_cached_tool_definitions(self.sub_agent_tool_definitions)

    def _init_llm_and_summary(self):
        """初始化 LLM 调用处理器和摘要处理器"""
        # LLM 调用处理器
        self.llm_handler = LLMCallHandler(
            main_llm_client=self.llm_client,
            sub_agent_llm_client=self.sub_agent_llm_client,
            task_log=self.task_log,
            add_message_id=self.add_message_id,
            keep_tool_result=self.cfg.main_agent.keep_tool_result,
            stream_error_callback=self.stream_handler.stream_tool_call,
        )

        # 摘要处理器
        self.summary_handler = SummaryHandler(
            llm_call_handler=self.llm_handler,
            chinese_context=self.chinese_context,
            response_language=self.response_language,
        )
        self.summary_handler.context = self.context

    def _init_skills_and_prompt(self):
        """初始化 Task Planner、Inline Skill Selector 和 Prompt Builder"""
        # Task Planner — 仅在 deep_research.enabled AND auto_planning 时启用
        dr_cfg = self.cfg.main_agent.get("deep_research", {})
        planning_enabled = (
            dr_cfg and dr_cfg.get("enabled", False) and dr_cfg.get("auto_planning", False)
        )
        self.task_planner = TaskPlanner(enabled=planning_enabled)

        # Inline Skill Selector
        self.inline_skill_selector = external_loader.get_inline_skill_selector(
            self.cfg, chinese=self.chinese_context
        )
        # 将 <next_skills> 注册为 reasoning tag，使其被 TextInterceptor 自动提取并从输出中剥离
        if self.inline_skill_selector:
            from mem_deep_research_core.skills.inline_selector import InlineSkillSelector

            current_tags = list(self.key_message_interceptor.tag_extractor.reasoning_tags)
            if InlineSkillSelector.TAG_NAME not in current_tags:
                current_tags.append(InlineSkillSelector.TAG_NAME)
                self.key_message_interceptor.set_reasoning_tags(current_tags)

        # Prompt Builder
        self.prompt_builder = PromptBuilder(
            cfg=self.cfg,
            context=self.context,
            chinese_context=self.chinese_context,
            inline_skill_selector=self.inline_skill_selector,
        )

    def _init_context_manager(self):
        """初始化 Context Manager（三级 context 管理 + dedup + source registry）"""
        cm_cfg_dict = ensure_dict(self.cfg.main_agent.get("context_manager", {}))
        cm_config = ContextManagerConfig(**cm_cfg_dict) if cm_cfg_dict else ContextManagerConfig()
        self.context_manager = ContextManager(config=cm_config)
        # 注入 token 估算函数
        if hasattr(self.llm_client, "_estimate_tokens"):
            self.context_manager.set_token_estimator(self.llm_client._estimate_tokens)

    @staticmethod
    def _extract_recent_tool_names(message_history: list, lookback: int = RECENT_TOOL_LOOKBACK) -> list:
        """从最近消息中提取 tool_use 的 name 列表"""
        names = []
        for msg in message_history[-lookback:]:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        name = block.get("name")
                        if name and name not in names:
                            names.append(name)
        return names

    @staticmethod
    def _deduplicate_trailing_messages(message_history: list) -> int:
        """移除消息历史末尾重复的 assistant 响应，保留第一次出现。

        当循环检测终止时，message_history 可能包含多轮相同的 assistant 响应，
        这会导致摘要 LLM 困惑或生成空内容。此方法从末尾向前扫描，
        移除连续重复的 assistant 消息（基于文本内容 hash），
        并在末尾追加一条说明，引导摘要 LLM 基于已有信息作答。

        Returns:
            int: 移除的消息数量
        """
        if len(message_history) < 4:
            return 0

        def _text_hash(msg: dict) -> int:
            content = msg.get("content", "")
            if isinstance(content, list):
                texts = [
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                ]
                text = "".join(texts)
            elif isinstance(content, str):
                text = content
            else:
                text = str(content)
            return hash(text[:500]) if text else 0

        # 从末尾收集连续 assistant 消息的 hash
        i = len(message_history) - 1
        tail_hashes = []
        while i >= 0 and message_history[i].get("role") == "assistant":
            tail_hashes.append((i, _text_hash(message_history[i])))
            i -= 1
            # 跳过中间的 user 消息（如 INJECT_HINT）
            if i >= 0 and message_history[i].get("role") == "user":
                content_str = str(message_history[i].get("content", ""))
                if any(kw in content_str for kw in SYSTEM_MESSAGE_KEYWORDS):
                    i -= 1

        if len(tail_hashes) < 2:
            return 0

        # 找出重复 hash 的索引，保留最早的一个
        seen_hashes = {}
        indices_to_remove = []
        for idx, h in reversed(tail_hashes):  # 从前往后遍历
            if h in seen_hashes:
                indices_to_remove.append(idx)
            else:
                seen_hashes[h] = idx

        if not indices_to_remove:
            return 0

        # 同时移除紧跟在被删 assistant 消息后面的 INJECT_HINT user 消息
        all_remove = set(indices_to_remove)
        for idx in indices_to_remove:
            # 检查 idx+1 和 idx-1 是否为注入的指令消息
            for neighbor in (idx + 1, idx - 1):
                if 0 <= neighbor < len(message_history) and neighbor not in all_remove:
                    msg = message_history[neighbor]
                    if msg.get("role") == "user":
                        content_str = str(msg.get("content", ""))
                        if any(kw in content_str for kw in SYSTEM_MESSAGE_KEYWORDS):
                            all_remove.add(neighbor)

        # 按索引从大到小移除
        for idx in sorted(all_remove, reverse=True):
            if idx < len(message_history):
                message_history.pop(idx)

        removed = len(all_remove)
        if removed > 0:
            # 追加引导消息，帮助摘要 LLM 生成有效输出
            message_history.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": FALLBACK_LOOP_TERMINATED,
                        }
                    ],
                }
            )
            logger.info(
                f"[DEDUP] Removed {removed} duplicate/injected messages from history tail, "
                f"history now {len(message_history)} messages"
            )

        return removed

    def _load_interceptor_config(self) -> InterceptorConfig:
        """从配置文件加载拦截器配置"""
        interceptor_cfg = self.cfg.main_agent.get("interceptor", {})

        if not interceptor_cfg:
            return InterceptorConfig()

        preset = interceptor_cfg.get("preset")
        if preset:
            config = InterceptorPresets.from_name(preset)
            logger.info(f"[INTERCEPTOR] Using preset: {preset}")
        else:
            config = InterceptorConfig()

        # 应用自定义配置
        if "filter_tags" in interceptor_cfg:
            config.filter_tags = list(interceptor_cfg.filter_tags)
        if "reasoning_tags" in interceptor_cfg:
            config.reasoning_tags = list(interceptor_cfg.reasoning_tags)
        if "show_reasoning" in interceptor_cfg:
            config.show_reasoning = interceptor_cfg.show_reasoning
        if "show_tool_calls" in interceptor_cfg:
            config.show_tool_calls = interceptor_cfg.show_tool_calls
        if "show_text_output" in interceptor_cfg:
            config.show_text_output = interceptor_cfg.show_text_output
        if "strip_reasoning_from_output" in interceptor_cfg:
            config.strip_reasoning_from_output = interceptor_cfg.strip_reasoning_from_output
        if "custom_tag_handlers" in interceptor_cfg:
            config.custom_tag_handlers = dict(interceptor_cfg.custom_tag_handlers)

        logger.info(
            f"[INTERCEPTOR] Config: filter_tags={config.filter_tags}, reasoning_tags={config.reasoning_tags}"
        )
        return config

    # ========== 流式输出方法 ==========

    async def _stream_tool_reasoning(
        self, tool_name: str, action: str, details: str, parent_uid: str = None
    ):
        """输出工具调用的推理过程"""
        reasoning_id = f"tool_reason_{uuid.uuid4().hex[:8]}"

        # START / RESULT_SUMMARY 的 details 由 hooks 生成，已是完整格式化内容，直接使用。
        # 其他 action（ERROR、监控事件等）保留框架前缀。
        if action in ("START", "RESULT_SUMMARY", "QUERY"):
            content = details
        elif action == "ERROR":
            content = f"❌ **工具错误**: {tool_name}\n{details}"
        else:
            content = f"📝 **{action}**: {details}"

        await self.stream_handler.stream_reasoning(
            reasoning_id=reasoning_id,
            content=content,
            parent_uid=parent_uid or self.current_agent_id,
            status="SUCCESS",
        )

    # ========== 消息拦截方法 ==========

    async def _intercept_key_message(self, message_id: str, message: str, is_last: bool):
        """拦截并处理流式消息"""
        try:
            result, reasoning_blocks = self.key_message_interceptor.process(message, is_last)

            if reasoning_blocks:
                logger.info(f"[REASONING] Extracted {len(reasoning_blocks)} blocks")

            for block in reasoning_blocks:
                try:
                    await self.stream_handler.stream_reasoning(
                        reasoning_id=block.uid,
                        content=block.content,
                        parent_uid=self.current_agent_id,
                        status="SUCCESS",
                    )
                except Exception as e:
                    logger.warning(f"Failed to stream reasoning block: {e}")

            if result is not None:
                if not result.strip():
                    return True
                if self.key_message_interceptor.is_unbreakable_string(result):
                    return False
                else:
                    await self.stream_handler.stream_tool_call(
                        "show_text", {"text": result}, True, message_id
                    )
                    await asyncio.sleep(0)
                    return True
            return True
        except Exception as e:
            logger.error(f"Error in _intercept_key_message: {e}")
            try:
                if message and message.strip():
                    await self.stream_handler.stream_tool_call(
                        "show_text", {"text": message}, True, message_id
                    )
            except Exception:
                pass
            return True

    async def _streaming_final_message(self, message_id: str, message: str, is_last: bool):
        """最终消息的流式输出"""
        try:
            result, reasoning_blocks = self._final_message_interceptor.process(message, is_last)

            for block in reasoning_blocks:
                try:
                    await self.stream_handler.stream_reasoning(
                        reasoning_id=block.uid,
                        content=block.content,
                        parent_uid=self.current_agent_id,
                        status="SUCCESS",
                    )
                except Exception as e:
                    logger.warning(f"Failed to stream final reasoning block: {e}")

            if result is not None:
                if not result.strip():
                    return True
                if self._final_message_interceptor.is_unbreakable_string(result):
                    return False
                else:
                    await self.stream_handler.stream_message(
                        message_id=message_id, delta_content=result
                    )
                    await asyncio.sleep(0)
                    return True
            return True
        except Exception as e:
            logger.error(f"Error in _streaming_final_message: {e}")
            try:
                if message and message.strip():
                    await self.stream_handler.stream_message(
                        message_id=message_id, delta_content=message
                    )
            except Exception:
                pass
            return True

    # ========== LLM 调用处理 ==========

    async def _handle_llm_call_with_logging(
        self,
        system_prompt,
        message_history,
        tool_definitions,
        step_id: int,
        purpose: str = "LLM call",
        keep_tool_result: int = -1,
        agent_type: str = "main",
        stream_message_callback: Callable = None,
    ) -> tuple[str | None, bool, Any | None]:
        """统一的 LLM 调用和日志处理，委托给 LLMCallHandler"""
        return await self.llm_handler.handle_llm_call(
            system_prompt=system_prompt,
            message_history=message_history,
            tool_definitions=tool_definitions,
            step_id=step_id,
            purpose=purpose,
            agent_type=agent_type,
            stream_message_callback=stream_message_callback,
        )

    async def _handle_summary_with_context_limit_retry(
        self,
        system_prompt,
        agent_prompt_instance,
        message_history,
        tool_definitions,
        purpose,
        task_description,
        task_failed,
        agent_type="main",
        task_guidance="",
        stream_message_callback: Callable = None,
        **kwargs,
    ):
        """处理摘要生成，委托给 SummaryHandler"""
        return await self.summary_handler.handle_summary_with_retry(
            system_prompt=system_prompt,
            agent_prompt_instance=agent_prompt_instance,
            message_history=message_history,
            tool_definitions=tool_definitions,
            purpose=purpose,
            task_description=task_description,
            task_failed=task_failed,
            agent_type=agent_type,
            task_guidance=task_guidance,
            stream_message_callback=stream_message_callback,
        )

    # ========== 子 Agent 运行 ==========

    async def run_sub_agent(self, sub_agent_name, task_description, keep_tool_result: int = -1):
        """运行子 Agent"""
        return await self.sub_agent_runner.run(
            sub_agent_name=sub_agent_name,
            task_description=task_description,
            keep_tool_result=keep_tool_result,
        )

    # ========== 主 Agent 运行 ==========

    async def run_main_agent(
        self, task_description, task_file_name=None, task_id="default_task", history=None
    ):
        """执行主 Agent 任务"""
        workflow_id = await self.stream_handler.stream_start_workflow(task_description, task_id)
        keep_tool_result = int(self.cfg.main_agent.keep_tool_result)

        logger.debug(f"\n{'=' * 20} Starting Task: {task_id} {'=' * 20}")
        logger.debug(f"Task Description: {task_description}")

        # 1. 处理输入
        initial_user_content, task_description = process_input(task_description, task_file_name)
        task_guidance = self.prompt_builder.build_task_guidance()
        initial_user_content[0]["text"] = initial_user_content[0]["text"] + task_guidance

        # 2. 生成提示词（如果启用）
        hint_notes = await self.prompt_builder.generate_hints(task_description)
        if hint_notes:
            initial_user_content[0]["text"] = initial_user_content[0]["text"] + hint_notes

        logger.info("Initial user input content: %s", initial_user_content)

        # 3. 构建消息历史
        message_history = self._build_initial_history(history)
        message_history.append({"role": "user", "content": initial_user_content})

        # 4. 获取工具定义
        _perf_t0 = time.perf_counter()
        tool_definitions = await self._get_tool_definitions()
        self.task_log.record_perf("tool_definitions_fetch", time.perf_counter() - _perf_t0)
        self.task_log.log_step("get_main_tool_definitions", f"{tool_definitions}")

        # 4.5. LLM Skill 选择
        _perf_t0 = time.perf_counter()
        selected_skill_names = await self.prompt_builder.select_skills(initial_user_content, tool_definitions)
        self.task_log.record_perf("skill_selection", time.perf_counter() - _perf_t0)

        # 5. 生成系统提示词
        _perf_t0 = time.perf_counter()
        system_prompt, main_agent_prompt_instance, deep_research_cfg = self.prompt_builder.build_system_prompt(
            tool_definitions, initial_user_content, selected_skill_names
        )
        self.task_log.record_perf("system_prompt_build", time.perf_counter() - _perf_t0)

        # 6. 主循环
        final_answer_text, is_simple_response = await self._run_main_loop(
            system_prompt=system_prompt,
            message_history=message_history,
            tool_definitions=tool_definitions,
            main_agent_prompt_instance=main_agent_prompt_instance,
            deep_research_cfg=deep_research_cfg,
            task_description=task_description,
            task_guidance=task_guidance,
            keep_tool_result=keep_tool_result,
        )

        # 7. 后处理
        _perf_t0 = time.perf_counter()
        final_summary, final_boxed_answer = await post_process_final_answer(
            cfg=self.cfg,
            final_answer_text=final_answer_text,
            task_description=task_description,
            message_history=message_history,
            system_prompt=system_prompt,
            chinese_context=self.chinese_context,
            task_log=self.task_log,
            output_formatter=self.output_formatter,
            llm_client=self.llm_client,
            is_simple_response=is_simple_response,
        )
        self.task_log.record_perf("post_process_duration", time.perf_counter() - _perf_t0)

        # 结束流式输出
        await self.stream_handler.stream_end_llm("reporter")
        await self.stream_handler.stream_end_agent("reporter", self.current_agent_id)

        main_agent_usage = self.llm_client.get_usage()
        await self.stream_handler.stream_usage_info("main", main_agent_usage, "main_agent_end")

        if self.sub_agent_llm_client and self.sub_agent_llm_client is not self.llm_client:
            sub_agent_usage = self.sub_agent_llm_client.get_usage()
            await self.stream_handler.stream_usage_info(
                "sub_agent", sub_agent_usage, "sub_agent_end"
            )

        await self.stream_handler.stream_end_workflow(workflow_id)

        logger.debug(f"\n{'=' * 20} Task {task_id} Finished {'=' * 20}")
        self.task_log.log_step("task_completed", f"Task {task_id} completed successfully")

        return final_summary, final_boxed_answer

    def _build_initial_history(self, history) -> list:
        """构建初始消息历史"""
        message_history = []
        if history:
            for turn_history in history:
                for message in turn_history["main_agent"]:
                    content = message["content"]
                    if isinstance(content, str):
                        content = re.sub(
                            r"<think>.*?</think>", "", content, flags=re.DOTALL
                        ).strip()
                    message_history.append({"role": message["role"], "content": content})
        return message_history

    async def _get_tool_definitions(self) -> list:
        """获取工具定义"""
        if not self.tool_definitions:
            tool_definitions = await self.main_agent_tool_manager.get_all_tool_definitions()
            if self.cfg.sub_agents is not None and self.cfg.sub_agents:
                tool_definitions += expose_sub_agents_as_tools(self.cfg.sub_agents)
        else:
            tool_definitions = self.tool_definitions

        if not tool_definitions:
            logger.debug("Warning: No tool definitions found.")
        return tool_definitions

    def _create_main_loop_runner(self) -> MainLoopRunner:
        """创建主循环运行器"""
        from mem_deep_research_core.core.main_loop import MainLoopContext

        ctx = MainLoopContext(
            cfg=self.cfg,
            monitor=self.monitor,
            context_manager=self.context_manager,
            stream_handler=self.stream_handler,
            tool_executor=self.tool_executor,
            sub_agent_runner=self.sub_agent_runner,
            llm_handler=self.llm_handler,
            summary_handler=self.summary_handler,
            task_planner=self.task_planner,
            inline_skill_selector=self.inline_skill_selector,
            llm_client=self.llm_client,
            output_formatter=self.output_formatter,
            task_log=self.task_log,
            context=self.context,
            chinese_context=self.chinese_context,
            response_language=self.response_language,
            handle_llm_call=self._handle_llm_call_with_logging,
            handle_summary=self._handle_summary_with_context_limit_retry,
            intercept_key_message=self._intercept_key_message,
            streaming_final_message=self._streaming_final_message,
            stream_tool_reasoning=self._stream_tool_reasoning,
            extract_recent_tool_names=self._extract_recent_tool_names,
            deduplicate_trailing_messages=self._deduplicate_trailing_messages,
        )
        return MainLoopRunner(ctx)

    async def _run_main_loop(
        self,
        system_prompt,
        message_history,
        tool_definitions,
        main_agent_prompt_instance,
        deep_research_cfg,
        task_description,
        task_guidance,
        keep_tool_result,
    ):
        """运行主执行循环（委托给 MainLoopRunner）"""
        runner = self._create_main_loop_runner()
        final_answer_text, is_simple_response = await runner.run(
            system_prompt=system_prompt,
            message_history=message_history,
            tool_definitions=tool_definitions,
            main_agent_prompt_instance=main_agent_prompt_instance,
            deep_research_cfg=deep_research_cfg,
            task_description=task_description,
            task_guidance=task_guidance,
            keep_tool_result=keep_tool_result,
        )
        # 同步 current_agent_id（runner 内部会更新）
        self.current_agent_id = runner.current_agent_id
        return final_answer_text, is_simple_response
