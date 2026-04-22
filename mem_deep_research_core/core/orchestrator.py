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
from typing import Any

from omegaconf import DictConfig

from mem_deep_research_core.core.answer_handler import post_process_final_answer
from mem_deep_research_core.core.constants import (
    DEFAULT_RESULT_OFFLOAD_THRESHOLD,
    DEFAULT_SCRAPE_MAX_LENGTH,
    ensure_dict,
    parse_bool_config,
)
from mem_deep_research_core.core.context_manager import ContextManager, ContextManagerConfig
from mem_deep_research_core.core.deferred_tools import DeferredToolManager
from mem_deep_research_core.core.agent_runtime import AgentRuntime
from mem_deep_research_core.core.hooks import HookContext
from mem_deep_research_core.core.interceptor_config import InterceptorConfig, InterceptorPresets
from mem_deep_research_core.core.llm_call_handler import (
    LLMCallHandler,
    SummaryHandler,
)
from mem_deep_research_core.core.main_loop import MainLoopRunner
from mem_deep_research_core.core.message_interceptor import MessageInterceptorHandler
from mem_deep_research_core.core.message_utils import (
    deduplicate_trailing_messages,
    extract_recent_tool_names,
)
from mem_deep_research_core.core.monitoring import (
    ExecutionMonitor,
    MonitoringConfig,
)
from mem_deep_research_core.core.prompt_builder import PromptBuilder

# 导入拆分后的模块
from mem_deep_research_core.core.stream_handler import StreamHandler
from mem_deep_research_core.core.sub_agent_runner import SubAgentRunner
from mem_deep_research_core.core.task_planner import TaskPlanner
from mem_deep_research_core.core.tool_executor import ToolExecutor
from mem_deep_research_core.core.tool_result_formatter import ToolResultFormatter
from mem_deep_research_core.llm.provider_client_base import LLMProviderClientBase
from mem_deep_research_core.mem_deep_research_logging.task_tracer import TaskTracer
from mem_deep_research_core.tool.manager import ToolManager
from mem_deep_research_core.utils.external_loader import ConfigLoader
from mem_deep_research_core.utils.io_utils import OutputFormatter, process_input
from mem_deep_research_core.utils.stream_parsing_utils import TextInterceptor
from mem_deep_research_core.utils.tool_utils import expose_sub_agents_as_tools

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


# 注意：默认钩子注册已移至 AgentRuntime.setup_hook_defaults()
# 此处不再注册到全局单例

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
        router_llm_client: LLMProviderClientBase | None = None,
        stream_queue: Any | None = None,
        tool_definitions: list[dict[str, Any]] | None = None,
        sub_agent_tool_definitions: dict[str, list[dict[str, Any]]] | None = None,
        context: dict[str, Any] | None = None,
        runtime: AgentRuntime | None = None,
    ):
        # 运行时隔离
        self.runtime = runtime or AgentRuntime()
        self._hooks = self.runtime.hooks

        # 注册所有框架默认钩子到本实例
        self.runtime.setup_hook_defaults()

        # 基础组件
        self.main_agent_tool_manager = main_agent_tool_manager
        self.sub_agent_tool_managers = sub_agent_tool_managers
        self.llm_client = llm_client
        self.sub_agent_llm_client = sub_agent_llm_client or llm_client
        self.router_llm_client = router_llm_client
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
        # Always sync to context so hooks can read language settings without KeyError.
        # In "auto" mode this will be updated by MainLoopRunner after LLM detection.
        self.context["response_language"] = self.response_language
        self.context["chinese_context"] = self.chinese_context

        self.execution_mode = self.cfg.main_agent.get("execution_mode", "auto")

        self.add_message_id = parse_bool_config(self.cfg.main_agent.get("add_message_id", False))
        logger.info(f"add_message_id: {self.add_message_id}")

        # 传递 task_log 给 LLM 客户端
        if self.llm_client and task_log:
            self.llm_client.task_log = task_log
        if self.sub_agent_llm_client and task_log and self.sub_agent_llm_client != self.llm_client:
            self.sub_agent_llm_client.task_log = task_log

        # 设置上下文到工具管理器（无条件传播，不依赖 user_id）
        if self.context:
            self.main_agent_tool_manager.set_context(self.context)
            for sub_manager in self.sub_agent_tool_managers.values():
                sub_manager.set_context(self.context)

        # ========== 初始化组合模块 ==========
        self._init_modules()

    def _init_modules(self):
        """初始化各个组合模块"""
        self._init_stream_and_interceptor()
        self._init_llm_and_summary()  # Before monitoring_and_tools (SubAgentRunner needs handlers)
        self._init_monitoring_and_tools()
        self._init_skills_and_prompt()
        self._init_context_manager()
        self._init_deferred_tools()
        self._init_transcript()
        self._init_file_state_cache()
        self.current_agent_id: str | None = None

    def _init_file_state_cache(self):
        """初始化文件状态缓存"""
        from mem_deep_research_core.core.file_state_cache import FileStateCache

        self.file_state_cache = FileStateCache(max_size=100)

    def _init_deferred_tools(self):
        """初始化延迟工具加载管理器"""
        threshold = self.cfg.main_agent.get("deferred_tools_threshold", 20)
        self.deferred_tool_manager = DeferredToolManager(threshold=threshold)

    def _init_transcript(self):
        """初始化 Transcript 事件日志"""
        from mem_deep_research_core.core.transcript import Transcript

        transcript_enabled = self.cfg.main_agent.get("transcript_enabled", True)
        if transcript_enabled:
            self.transcript = Transcript(agent_name="main")
        else:
            self.transcript = None

    def _init_stream_and_interceptor(self):
        """初始化流式处理器和消息拦截器"""
        # 流式处理器
        self.stream_handler = StreamHandler(self.stream_queue)

        # 工具结果格式化器
        self.tool_formatter = ToolResultFormatter(self.context, hooks=self._hooks)

        # 消息拦截处理器
        interceptor_config = self._load_interceptor_config()
        self.key_message_interceptor = TextInterceptor(
            interceptor_config.get_all_filter_keywords(),
            reasoning_tags=interceptor_config.reasoning_tags,
            strip_tags=interceptor_config.strip_tags,
        )
        self.message_interceptor = MessageInterceptorHandler(
            config=interceptor_config,
            stream_reasoning_callback=self.stream_handler.stream_reasoning,
            stream_tool_call_callback=self.stream_handler.stream_tool_call,
            stream_message_callback=self.stream_handler.stream_message,
            context=self.context,
            hooks=self._hooks,
        )

        # 最终消息拦截器
        self._final_message_interceptor = TextInterceptor(
            ["<use_mcp_tool>"], strip_tags=interceptor_config.strip_tags
        )

    def _init_monitoring_and_tools(self):
        """初始化执行监控器、工具执行器和子 Agent 运行器"""
        # 执行监控器（从配置读取）
        monitoring_cfg_dict = ensure_dict(self.cfg.main_agent.get("monitoring", {}))
        if monitoring_cfg_dict:
            try:
                from mem_deep_research_core.config_schema import MonitoringConfigSchema

                monitoring_schema = MonitoringConfigSchema(**monitoring_cfg_dict)
                monitoring_config = MonitoringConfig.from_schema(monitoring_schema)
            except (TypeError, ValueError) as e:
                logger.warning(f"[Orchestrator] Invalid monitoring config, using defaults: {e}")
                monitoring_config = MonitoringConfig()
        else:
            monitoring_config = MonitoringConfig()
        self.monitor = ExecutionMonitor(
            config=monitoring_config,
            stream_reasoning_callback=self._stream_tool_reasoning,  # 保留：有自定义逻辑
        )

        # 工具执行器（含自动重试配置）
        retry_cfg = ensure_dict(self.cfg.main_agent.get("tool_retry", {}))
        self.tool_executor = ToolExecutor(
            tool_manager=self.main_agent_tool_manager,
            output_formatter=self.output_formatter,
            tool_result_formatter=self.tool_formatter,
            context=self.context,
            stream_tool_call=self.stream_handler.stream_tool_call,
            stream_tool_reasoning=self._stream_tool_reasoning,  # 保留：有自定义逻辑
            stream_usage_info=self.stream_handler.stream_usage_info,
            retry_max=retry_cfg.get("max_retries", 2) if retry_cfg.get("enabled", True) else 0,
            retry_backoff_base=retry_cfg.get("backoff_base", 1.0),
            hook_registry=self._hooks,  # 注入实例级 hooks，避免全局单例
        )
        # scrape_max_length: 优先从配置读，fallback 环境变量，最后默认 20000
        scrape_max_length = monitoring_cfg_dict.get("scrape_max_length", None)
        if scrape_max_length is None:
            try:
                scrape_max_length = int(
                    os.getenv("SCRAPE_MAX_LENGTH", str(DEFAULT_SCRAPE_MAX_LENGTH))
                )
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
            handle_llm_call=self.llm_handler.handle_llm_call,
            handle_summary=self.summary_handler.handle_summary_with_retry,
            intercept_key_message=self._intercept_key_message,
            hooks=self._hooks,
            config_loader=self.runtime.config_loader,
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
            hooks=self._hooks,
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
        dr_cfg = self.cfg.main_agent.get("task_engine", {})
        planning_enabled = (
            dr_cfg and dr_cfg.get("enabled", False) and dr_cfg.get("auto_planning", False)
        )
        self.task_planner = TaskPlanner(enabled=planning_enabled)

        # TodoTracker — enabled via todo_tracker.enabled or deep_research.enabled
        from mem_deep_research_core.core.todo_tracker import TodoTracker

        todo_enabled = self.cfg.main_agent.get("todo_tracker", {}).get("enabled", False)
        if not todo_enabled:
            # Fallback: enable with deep_research
            todo_enabled = dr_cfg and dr_cfg.get("enabled", False) if dr_cfg else False
        self.todo_tracker = TodoTracker(enabled=todo_enabled)

        # Inline Skill Selector
        self.inline_skill_selector = self.runtime.config_loader.get_inline_skill_selector(
            self.cfg, chinese=self.chinese_context
        )
        # 将 <next_skills> 注册为 reasoning tag，使其被 TextInterceptor 自动提取并从输出中剥离
        if self.inline_skill_selector:
            from mem_deep_research_core.skills.inline_selector import InlineSkillSelector

            current_tags = list(self.key_message_interceptor.tag_extractor.reasoning_tags)
            if InlineSkillSelector.TAG_NAME not in current_tags:
                current_tags.append(InlineSkillSelector.TAG_NAME)
                self.key_message_interceptor.set_reasoning_tags(current_tags)

        # Skill Commands — 统一格式，传递给 MainLoopContext
        self.skill_commands = self.runtime.config_loader.get_skill_commands()

        # Prompt Builder
        self.prompt_builder = PromptBuilder(
            cfg=self.cfg,
            context=self.context,
            chinese_context=self.chinese_context,
            inline_skill_selector=self.inline_skill_selector,
            hooks=self._hooks,
            config_loader=self.runtime.config_loader,
            response_language=self.response_language,
        )

    def _init_context_manager(self):
        """初始化 Context Manager（三级 context 管理 + dedup + source registry）"""
        cm_cfg_dict = ensure_dict(self.cfg.main_agent.get("context_manager", {}))
        cm_config = ContextManagerConfig(**cm_cfg_dict) if cm_cfg_dict else ContextManagerConfig()
        self.context_manager = ContextManager(config=cm_config, hooks=self._hooks)
        # 注入 token 估算函数
        if hasattr(self.llm_client, "_estimate_tokens"):
            self.context_manager.set_token_estimator(self.llm_client._estimate_tokens)

        # Offload dir: use config if set, otherwise default to output_dir/offloaded_results
        offload_dir = cm_cfg_dict.get("result_offload_dir", "")
        if not offload_dir:
            output_dir = self.cfg.get("output_dir", "logs/")
            offload_dir = os.path.join(output_dir, "offloaded_results")
        self.context_manager.set_offload_dir(offload_dir)

        # Long-term memory (optional, persists across sessions)
        memory_cfg = self.cfg.main_agent.get("memory", {})
        if memory_cfg.get("enabled", False):
            from mem_deep_research_core.core.memory import LongTermMemory

            output_dir = self.cfg.get("output_dir", "logs/")
            memory_dir = os.path.join(output_dir, "memory")
            self.long_term_memory = LongTermMemory(storage_path=memory_dir)
        else:
            self.long_term_memory = None

    # Static utilities delegated to message_utils (backward compat)
    _extract_recent_tool_names = staticmethod(extract_recent_tool_names)
    _deduplicate_trailing_messages = staticmethod(deduplicate_trailing_messages)

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
            except Exception as fallback_err:
                logger.warning(f"Fallback streaming also failed in _intercept_key_message: {fallback_err}")
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
            except Exception as fallback_err:
                logger.warning(f"Fallback streaming also failed in _streaming_final_message: {fallback_err}")
            return True

    # ========== LLM 调用处理 ==========

    # _handle_llm_call_with_logging and _handle_summary_with_context_limit_retry
    # removed — handlers are now injected directly into MainLoopContext.

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
        self,
        task_description,
        task_file_name=None,
        task_id="default_task",
        history=None,
        resume_from: dict | None = None,
    ):
        """执行主 Agent 任务"""
        workflow_id = await self.stream_handler.stream_start_workflow(task_description, task_id)
        keep_tool_result = int(self.cfg.main_agent.keep_tool_result)

        logger.debug(f"\n{'=' * 20} Starting Task: {task_id} {'=' * 20}")
        logger.debug(f"Task Description: {task_description}")

        # 0.5. 输入编译链
        from mem_deep_research_core.core.input_compiler import InputCompiler

        input_cfg = ensure_dict(self.cfg.main_agent.get("input_process", {}))
        input_compiler = InputCompiler(
            hooks=self._hooks,
            file_ref_allowed_dirs=input_cfg.get("file_ref_allowed_dirs", []),
        )
        compile_result = input_compiler.compile(task_description, context=self.context)
        task_description = compile_result.query

        # 将提取的附件信息记录到 task_log
        if compile_result.extracted_urls:
            self.task_log.log_step(
                "input_compile",
                f"Extracted {len(compile_result.extracted_urls)} URLs from query",
            )
        if compile_result.attachments:
            self.task_log.log_step(
                "input_compile",
                f"Loaded {len(compile_result.attachments)} file attachments",
            )

        # 1. 处理输入
        initial_user_content, task_description = process_input(task_description, task_file_name)
        task_guidance = self.prompt_builder.build_task_guidance()
        initial_user_content[0]["text"] = initial_user_content[0]["text"] + task_guidance

        # 注入文件附件内容到 user content
        for attachment in compile_result.attachments:
            if attachment.get("type") == "file" and attachment.get("content"):
                initial_user_content.append(
                    {
                        "type": "text",
                        "text": f"\n\n--- File: {attachment['path']} ---\n{attachment['content']}",
                    }
                )

        # 2. 生成提示词（如果启用）
        hint_notes = await self.prompt_builder.generate_hints(task_description)
        if hint_notes:
            initial_user_content[0]["text"] = initial_user_content[0]["text"] + hint_notes

        logger.info("Initial user input content: %s", initial_user_content)

        # 3. 构建消息历史
        message_history = self._build_initial_history(history)
        from mem_deep_research_core.core.constants import MT

        message_history.append({"role": "user", "_type": MT.USER_INPUT, "content": initial_user_content})

        # 4. 获取工具定义
        _perf_t0 = time.perf_counter()
        tool_definitions = await self._get_tool_definitions()
        self.task_log.record_perf("tool_definitions_fetch", time.perf_counter() - _perf_t0)
        self.task_log.log_step("get_main_tool_definitions", f"{tool_definitions}")

        # 4.1. Deferred tools: 工具数量多时只暴露名称+描述
        tool_definitions, deferred_active = self.deferred_tool_manager.apply(tool_definitions)
        if deferred_active:
            self.task_log.log_step(
                "deferred_tools_active",
                f"Deferred tool loading enabled, {len(self.deferred_tool_manager._full_registry)} tools deferred",
            )

        # 4.5. Resume: 优先使用保存的 system_prompt（保证上下文一致性）
        _resumed_prompt = resume_from.get("system_prompt", "") if resume_from else ""
        if _resumed_prompt:
            # 跳过 skill 选择和 prompt build，直接使用保存的 system_prompt
            from mem_deep_research_core.utils.tool_utils import _load_agent_prompt

            prompt_cfg = self.cfg.main_agent.get("prompt", {})
            main_agent_prompt_instance = _load_agent_prompt(prompt_cfg)
            task_engine_cfg_raw = self.cfg.main_agent.get("task_engine", {})
            task_engine_cfg = (
                dict(task_engine_cfg_raw) if task_engine_cfg_raw else None
            )
            system_prompt = _resumed_prompt
            self.task_log.log_step(
                "resume_prompt",
                f"Using saved system_prompt ({len(system_prompt)} chars), "
                "skipped skill selection and prompt build",
            )
            logger.info(
                f"[Resume] Using saved system_prompt ({len(system_prompt)} chars)"
            )
        else:
            # 正常流程：skill 选择 + prompt build
            _perf_t0 = time.perf_counter()
            selected_skill_names = await self.prompt_builder.select_skills(
                initial_user_content, tool_definitions
            )
            self.task_log.record_perf("skill_selection", time.perf_counter() - _perf_t0)

            _perf_t0 = time.perf_counter()
            system_prompt, main_agent_prompt_instance, task_engine_cfg = (
                self.prompt_builder.build_system_prompt(
                    tool_definitions, initial_user_content, selected_skill_names
                )
            )
            self.task_log.record_perf("system_prompt_build", time.perf_counter() - _perf_t0)

        # 6. 主循环
        final_answer_text, is_simple_response = await self._run_main_loop(
            system_prompt=system_prompt,
            message_history=message_history,
            tool_definitions=tool_definitions,
            main_agent_prompt_instance=main_agent_prompt_instance,
            task_engine_cfg=task_engine_cfg,
            task_description=task_description,
            task_guidance=task_guidance,
            keep_tool_result=keep_tool_result,
            resume_from=resume_from,
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
            context=self.context,
            hooks=self._hooks,
        )
        self.task_log.record_perf("post_process_duration", time.perf_counter() - _perf_t0)

        # 结束流式输出
        await self.stream_handler.stream_end_llm("reporter")
        await self.stream_handler.stream_end_agent("reporter", self.current_agent_id)

        main_agent_usage = self.llm_client.get_usage()
        await self.stream_handler.stream_usage_info("main", main_agent_usage, "main_agent_end")
        self.task_log.record_perf("total_prompt_tokens", main_agent_usage.get("total_prompt_tokens", 0), unit="tokens")
        self.task_log.record_perf("total_completion_tokens", main_agent_usage.get("total_completion_tokens", 0), unit="tokens")
        self.task_log.record_perf("total_tokens", main_agent_usage.get("total_tokens", 0), unit="tokens")

        if self.sub_agent_llm_client and self.sub_agent_llm_client is not self.llm_client:
            sub_agent_usage = self.sub_agent_llm_client.get_usage()
            await self.stream_handler.stream_usage_info(
                "sub_agent", sub_agent_usage, "sub_agent_end"
            )

        await self.stream_handler.stream_end_workflow(workflow_id)

        # Save transcript (if enabled)
        if self.transcript and self.transcript.event_count > 0:
            output_dir = self.cfg.get("output_dir", "logs")
            transcript_path = os.path.join(output_dir, f"transcript_{task_id}.jsonl")
            try:
                self.transcript.save(transcript_path)
                self.task_log.log_step(
                    "transcript_saved",
                    f"Saved {self.transcript.event_count} events to {transcript_path}",
                )
            except Exception as e:
                logger.warning(f"[Orchestrator] Failed to save transcript: {e}")

        logger.debug(f"\n{'=' * 20} Task {task_id} Finished {'=' * 20}")
        self.task_log.log_step("task_completed", f"Task {task_id} completed successfully")

        # Cleanup offload files after task is done
        removed = self.context_manager.cleanup_offload_files()
        if removed:
            self.task_log.log_step("offload_cleanup", f"Removed {removed} offload files")

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
        """获取工具定义（含内置工具 spawn_agent / update_todo）"""
        if not self.tool_definitions:
            tool_definitions = await self.main_agent_tool_manager.get_all_tool_definitions()
        else:
            tool_definitions = list(self.tool_definitions)

        # 子 Agent 工具注入
        if getattr(self.cfg, "sub_agents", None):
            tool_definitions += expose_sub_agents_as_tools(self.cfg.sub_agents)

        # 内置工具注入
        from mem_deep_research_core.core.main_loop import (
            _get_read_result_tool_definition,
            _get_spawn_agent_tool_definition,
        )
        from mem_deep_research_core.core.todo_tracker import TodoTracker

        # auto 模式注入全部内置工具，路由后由 main_loop 按 effective_mode 裁剪
        config_mode = self.cfg.main_agent.get("execution_mode", "auto")
        if config_mode != "quick":
            tool_definitions.append(_get_spawn_agent_tool_definition())

        if self.todo_tracker and self.todo_tracker.enabled:
            tool_definitions.append(TodoTracker.get_tool_definition())

        # read_result: offload 启用时注入，让 LLM 可以回捞被压缩/卸载的工具结果
        cm_cfg = self.cfg.main_agent.get("context_manager", {})
        if cm_cfg.get("result_offload_threshold", DEFAULT_RESULT_OFFLOAD_THRESHOLD) > 0:
            tool_definitions.append(_get_read_result_tool_definition())

        if not tool_definitions:
            logger.debug("Warning: No tool definitions found.")
        return tool_definitions

    def _create_main_loop_runner(self) -> MainLoopRunner:
        """创建主循环运行器"""
        from mem_deep_research_core.core.main_loop import MainLoopContext
        from mem_deep_research_core.core.profiles import DeepResearchProfile, StandardProfile

        # Profile 路由（Phase 2a）：根据 execution_mode / task_engine 选 profile
        # - deep / auto / task_engine 启用 → DeepResearchProfile（含研究专属 strategies）
        # - 其他 → StandardProfile
        # auto 在运行时可能升级为 deep，所以 auto 默认走 DeepResearchProfile
        # （DeepResearchProfile 的 strategies 是 Standard 的超集，无副作用）
        _task_engine_enabled = False
        try:
            _te = self.cfg.main_agent.get("task_engine", {}) or {}
            _task_engine_enabled = bool(_te.get("enabled", False))
        except Exception:
            pass
        _profile_instance = (
            DeepResearchProfile()
            if self.execution_mode in ("deep", "auto") or _task_engine_enabled
            else StandardProfile()
        )

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
            execution_mode=self.execution_mode,
            todo_tracker=self.todo_tracker,
            handle_llm_call=self.llm_handler.handle_llm_call,
            handle_summary=self.summary_handler.handle_summary_with_retry,
            intercept_key_message=self._intercept_key_message,
            streaming_final_message=self._streaming_final_message,
            stream_tool_reasoning=self._stream_tool_reasoning,
            extract_recent_tool_names=self._extract_recent_tool_names,
            deduplicate_trailing_messages=self._deduplicate_trailing_messages,
            long_term_memory=self.long_term_memory,
            hooks=self._hooks,
            deferred_tool_manager=self.deferred_tool_manager,
            transcript=self.transcript,
            file_state_cache=self.file_state_cache,
            skill_commands=self.skill_commands,
            router_llm_client=self.router_llm_client,
            config_loader=self.runtime.config_loader,
            profile=_profile_instance,
        )
        return MainLoopRunner(ctx)

    async def _run_main_loop(
        self,
        system_prompt,
        message_history,
        tool_definitions,
        main_agent_prompt_instance,
        task_engine_cfg,
        task_description,
        task_guidance,
        keep_tool_result,
        resume_from=None,
    ):
        """运行主执行循环（委托给 MainLoopRunner）"""
        runner = self._create_main_loop_runner()
        final_answer_text, is_simple_response = await runner.run(
            system_prompt=system_prompt,
            message_history=message_history,
            tool_definitions=tool_definitions,
            main_agent_prompt_instance=main_agent_prompt_instance,
            task_engine_cfg=task_engine_cfg,
            task_description=task_description,
            task_guidance=task_guidance,
            keep_tool_result=keep_tool_result,
            resume_from=resume_from,
        )
        # 同步 current_agent_id（runner 内部会更新）
        self.current_agent_id = runner.current_agent_id
        return final_answer_text, is_simple_response
