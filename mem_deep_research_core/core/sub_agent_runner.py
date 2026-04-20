"""
子 Agent 运行模块

子 Agent 复用 MainLoopRunner 执行，获得与主 Agent 相同的能力：
- 三级 Context 管理（dedup + 压缩 + 紧急裁剪）
- 循环检测 + 超时监控
- Hook 系统

子 Agent 的上下文与主 Agent 完全隔离。
"""

import json
import logging
import re
from collections.abc import Callable
from typing import Any

from omegaconf import DictConfig, OmegaConf

from mem_deep_research_core.core.constants import SUB_AGENT_PREFIX, ensure_dict
from mem_deep_research_core.core.context_manager import ContextManager, ContextManagerConfig
from mem_deep_research_core.core.hooks import HookContext, HookRegistry
from mem_deep_research_core.core.llm_call_handler import LLMCallHandler, SummaryHandler
from mem_deep_research_core.core.main_loop import MainLoopContext, MainLoopRunner
from mem_deep_research_core.core.monitoring import ExecutionMonitor, MonitoringConfig
from mem_deep_research_core.core.tool_executor import ToolExecutor
from mem_deep_research_core.core.tool_result_formatter import ToolResultFormatter
from mem_deep_research_core.llm.provider_client_base import LLMProviderClientBase
from mem_deep_research_core.mem_deep_research_logging.task_tracer import TaskTracer
from mem_deep_research_core.tool.manager import ToolManager
from mem_deep_research_core.utils.external_loader import ConfigLoader
from mem_deep_research_core.utils.io_utils import OutputFormatter
from mem_deep_research_core.utils.tool_utils import _load_agent_prompt

# Regex to strip the ## Language detection section from parent system prompt.
# Sub-agents inherit a resolved language and should not re-detect.
_RE_LANGUAGE_SECTION = re.compile(
    r"\n\n## Language\n\n.*?(?=\n\n## |\Z)", re.DOTALL
)


def _strip_language_section(prompt: str) -> str:
    """Remove the ## Language section injected for auto-detection."""
    return _RE_LANGUAGE_SECTION.sub("", prompt)

logger = logging.getLogger("mem_deep_research")


class SubAgentRunner:
    """子 Agent 运行器 -- 复用 MainLoopRunner 执行，隔离上下文"""

    def __init__(
        self,
        sub_agent_tool_managers: dict[str, ToolManager],
        sub_agent_llm_client: LLMProviderClientBase,
        output_formatter: OutputFormatter,
        cfg: DictConfig,
        task_log: TaskTracer,
        context: dict[str, Any] | None = None,
        chinese_context: bool = False,
        response_language: str = "auto",
        # 流式处理器
        stream_handler=None,
        stream_tool_reasoning: Callable | None = None,
        # LLM 调用回调
        handle_llm_call: Callable | None = None,
        handle_summary: Callable | None = None,
        # 消息拦截回调
        intercept_key_message: Callable | None = None,
        streaming_final_message: Callable | None = None,
        *,
        # 运行时依赖（必传）
        hooks: HookRegistry,
        config_loader: ConfigLoader,
    ):
        self.sub_agent_tool_managers = sub_agent_tool_managers
        self.sub_agent_llm_client = sub_agent_llm_client
        self.output_formatter = output_formatter
        self.cfg = cfg
        self.task_log = task_log
        self.context = context or {}
        self.chinese_context = chinese_context
        self.response_language = response_language

        self.stream_handler = stream_handler
        self.stream_tool_reasoning = stream_tool_reasoning
        self.handle_llm_call = handle_llm_call
        self.handle_summary = handle_summary
        self.intercept_key_message = intercept_key_message
        self.streaming_final_message = streaming_final_message
        self._hooks = hooks
        self._config_loader = config_loader

        self._cached_tool_definitions: dict[str, list[dict]] | None = None

    def set_cached_tool_definitions(self, definitions: dict[str, list[dict]]):
        """设置预缓存的子 Agent 工具定义"""
        self._cached_tool_definitions = definitions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_context_manager(
        self,
        llm_client,
        *,
        parent_context_manager: ContextManager | None = None,
        sub_agent_name: str | None = None,
    ) -> ContextManager:
        """Create a sub-agent ContextManager using inherit_with_override.

        Resolution order (later entries override earlier ones):
          1. ``ContextManagerConfig()`` built-in defaults
          2. ``main_agent.context_manager`` from config
          3. Parent ContextManager runtime snapshot (spawn path only; captures
             dynamic adjustments made at runtime that aren't in the YAML)
          4. ``sub_agents.<name>.context_manager`` from config
             (configured sub-agent path only; explicit overrides win)

        The parent's resolved ``_offload_dir`` is always used when available,
        to keep offload files in a single shared directory.
        """
        import os
        import dataclasses

        merged: dict[str, Any] = {}

        # Layer 2: main agent config
        if self.cfg:
            main_agent = self.cfg.get("main_agent", {})
            main_cm = ensure_dict(
                main_agent.get("context_manager", {}) if main_agent else {}
            )
            merged.update(main_cm)

        # Layer 3: parent runtime snapshot (for spawn path only)
        if parent_context_manager is not None:
            parent_snapshot = dataclasses.asdict(parent_context_manager.config)
            merged.update(parent_snapshot)

        # Layer 4: explicit sub-agent override (for configured sub-agents)
        if sub_agent_name and self.cfg and self.cfg.get("sub_agents"):
            sub_cfg = self.cfg.sub_agents.get(sub_agent_name)
            if sub_cfg:
                sub_cm = ensure_dict(sub_cfg.get("context_manager", {}) or {})
                if sub_cm:
                    logger.info(
                        f"[SubAgent:{sub_agent_name}] context_manager overrides: "
                        f"{sorted(sub_cm.keys())}"
                    )
                    merged.update(sub_cm)

        # Filter to fields accepted by ContextManagerConfig (defensive against
        # stale keys in YAML that the dataclass no longer declares).
        valid_fields = {f.name for f in dataclasses.fields(ContextManagerConfig)}
        filtered = {k: v for k, v in merged.items() if k in valid_fields}
        cm_config = ContextManagerConfig(**filtered) if filtered else ContextManagerConfig()
        cm = ContextManager(config=cm_config, hooks=self._hooks)

        if hasattr(llm_client, "_estimate_tokens"):
            cm.set_token_estimator(llm_client._estimate_tokens)

        # Offload dir: prefer parent's resolved dir, else config, else default.
        if parent_context_manager is not None and parent_context_manager._offload_dir:
            offload_dir = parent_context_manager._offload_dir
        else:
            offload_dir = merged.get("result_offload_dir", "")
            if not offload_dir and self.cfg:
                output_dir = self.cfg.get("output_dir", "logs/")
                offload_dir = os.path.join(output_dir, "offloaded_results")
        if offload_dir:
            cm.set_offload_dir(offload_dir)

        return cm

    @staticmethod
    def _parse_task_description(raw_arguments) -> str:
        """Extract task_description from LLM tool call arguments."""
        if isinstance(raw_arguments, dict):
            return raw_arguments.get("task_description", str(raw_arguments))
        if isinstance(raw_arguments, str):
            try:
                parsed = json.loads(raw_arguments)
                if isinstance(parsed, dict):
                    return parsed.get("task_description", raw_arguments)
            except (json.JSONDecodeError, ValueError):
                pass
            return raw_arguments
        return str(raw_arguments)

    async def _get_tool_definitions(self, sub_agent_name: str) -> list[dict]:
        """获取子 Agent 的工具定义"""
        if self._cached_tool_definitions:
            return self._cached_tool_definitions.get(sub_agent_name, [])
        tool_manager = self.sub_agent_tool_managers.get(sub_agent_name)
        if tool_manager:
            return await tool_manager.get_all_tool_definitions()
        return []

    def _build_system_prompt(
        self, sub_agent_name: str, tool_definitions: list, task_description: str
    ) -> tuple:
        """Build system prompt and return (system_prompt, prompt_instance)."""
        if not self.cfg.sub_agents or sub_agent_name not in self.cfg.sub_agents:
            raise ValueError(f"Sub-agent '{sub_agent_name}' not found in configuration")
        sub_agent_cfg = self.cfg.sub_agents[sub_agent_name]
        prompt_cfg = {}
        if hasattr(sub_agent_cfg, "prompt") and sub_agent_cfg.prompt:
            prompt_cfg = dict(sub_agent_cfg.prompt)
        if "agent_type" not in prompt_cfg:
            prompt_cfg["agent_type"] = "worker"

        project_dir = self._config_loader.get_project_dir()
        prompt_instance = _load_agent_prompt(prompt_cfg, project_dir=project_dir)
        system_prompt = prompt_instance.generate_system_prompt_with_mcp_tools(
            mcp_servers=tool_definitions,
            chinese_context=self.chinese_context,
            response_language=self.response_language,
        )

        # Inject skills
        skill_injector = self._config_loader.get_skill_injector()
        if skill_injector and task_description:
            tools_to_use = [t.get("name", "") for t in tool_definitions if isinstance(t, dict)]
            system_prompt = skill_injector.inject_skills(
                base_prompt=system_prompt,
                query=task_description,
                context=self.context,
                tools_to_use=tools_to_use,
            )

        # Hook: on_system_prompt_build
        hook_result = self._hooks.call(
            "on_system_prompt_build",
            HookContext(
                hook_name="on_system_prompt_build",
                context=self.context,
                result=system_prompt,
            ),
        )
        if isinstance(hook_result, str):
            system_prompt = hook_result

        return system_prompt, prompt_instance

    @staticmethod
    def _build_sub_agent_omegaconf(sub_agent_cfg) -> DictConfig:
        """Build a minimal OmegaConf config for MainLoopRunner compatibility.

        MainLoopRunner reads ``cfg.main_agent.*`` -- wrap the sub-agent section
        so the same attribute paths resolve correctly.
        """
        wrapped = {
            "main_agent": {
                "max_turns": sub_agent_cfg.get("max_turns", 10),
                "max_tool_calls_per_turn": sub_agent_cfg.get("max_tool_calls_per_turn", 5),
                "keep_tool_result": sub_agent_cfg.get("keep_tool_result", -1),
            }
        }
        return OmegaConf.create(wrapped)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        sub_agent_name: str,
        task_description: str | dict,
        keep_tool_result: int = -1,
        *,
        parent_context_manager: ContextManager | None = None,
    ) -> str:
        """Run a sub-agent by delegating to MainLoopRunner.

        Args:
            sub_agent_name: Sub-agent name (e.g. ``"agent-search"``).
            task_description: Task description string or arguments dict from LLM.
            keep_tool_result: Number of recent tool results to keep (-1 = all).
            parent_context_manager: Parent ContextManager — its offload_dir is
                shared with the sub-agent's isolated ContextManager, and any
                offload records the sub-agent produces are merged back on exit
                so cleanup_offload_files() can reach them (prevents orphan files).

        Returns:
            Final answer text from the sub-agent.
        """
        task_description = self._parse_task_description(task_description)
        task_description += "\n\nPlease provide the answer and detailed supporting information of the subtask given to you."

        logger.debug(f"\n=== Starting Sub Agent {sub_agent_name} ===")

        self.task_log.start_sub_agent_session(sub_agent_name, task_description)

        final_answer_text = ""
        system_prompt = ""
        context_manager: ContextManager | None = None
        message_history = [
            {"role": "user", "content": [{"type": "text", "text": task_description}]}
        ]

        try:
            # Build prompt + tools
            tool_definitions = await self._get_tool_definitions(sub_agent_name)
            system_prompt, prompt_instance = self._build_system_prompt(
                sub_agent_name, tool_definitions, task_description
            )

            if not self.cfg.sub_agents or sub_agent_name not in self.cfg.sub_agents:
                raise ValueError(f"Sub-agent '{sub_agent_name}' not found in configuration")
            sub_agent_cfg = self.cfg.sub_agents[sub_agent_name]
            effective_keep = int(sub_agent_cfg.get("keep_tool_result", keep_tool_result))
            display_name = sub_agent_name.replace(SUB_AGENT_PREFIX, "")

            # --- Create isolated components for this sub-agent ---

            context_manager = self._create_context_manager(
                self.sub_agent_llm_client,
                parent_context_manager=parent_context_manager,
                sub_agent_name=sub_agent_name,
            )

            monitor = ExecutionMonitor(
                config=MonitoringConfig(),
                stream_reasoning_callback=self.stream_tool_reasoning,
            )

            tool_manager = self.sub_agent_tool_managers.get(sub_agent_name)
            if tool_manager is None:
                logger.warning(
                    f"[SubAgent] No tool manager for '{sub_agent_name}', running without tools"
                )
                tool_executor = None
            else:
                tool_executor = ToolExecutor(
                    tool_manager=tool_manager,
                    output_formatter=self.output_formatter,
                    tool_result_formatter=ToolResultFormatter(self.context, hooks=self._hooks),
                    context=self.context,
                    stream_tool_call=self.stream_handler.stream_tool_call
                    if self.stream_handler
                    else None,
                    stream_tool_reasoning=self.stream_tool_reasoning,
                    stream_usage_info=self.stream_handler.stream_usage_info
                    if self.stream_handler
                    else None,
                    hook_registry=self._hooks,
                )

            llm_handler = LLMCallHandler(
                main_llm_client=self.sub_agent_llm_client,
                task_log=self.task_log,
                add_message_id=False,
                keep_tool_result=effective_keep,
                hooks=self._hooks,
            )

            summary_handler = SummaryHandler(
                llm_call_handler=llm_handler,
                chinese_context=self.chinese_context,
                response_language=self.response_language,
            )
            summary_handler.context = self.context

            from mem_deep_research_core.core.message_utils import (
                deduplicate_trailing_messages,
                extract_recent_tool_names,
            )
            from mem_deep_research_core.core.task_planner import TaskPlanner

            # No-op callbacks for features not used by sub-agents
            _noop_async = _make_noop_async()

            ctx = MainLoopContext(
                cfg=self._build_sub_agent_omegaconf(sub_agent_cfg),
                monitor=monitor,
                context_manager=context_manager,
                stream_handler=self.stream_handler,
                tool_executor=tool_executor,
                sub_agent_runner=None,  # Sub-agents cannot spawn sub-agents
                llm_handler=llm_handler,
                summary_handler=summary_handler,
                task_planner=TaskPlanner(enabled=False),
                inline_skill_selector=None,
                llm_client=self.sub_agent_llm_client,
                output_formatter=self.output_formatter,
                task_log=self.task_log,
                context=self.context,
                chinese_context=self.chinese_context,
                response_language=self.response_language,
                agent_name=display_name,
                # Callbacks
                handle_llm_call=self.handle_llm_call or _noop_async,
                handle_summary=self.handle_summary or _noop_async,
                intercept_key_message=self.intercept_key_message or _noop_async,
                streaming_final_message=self.streaming_final_message or _noop_async,
                stream_tool_reasoning=self.stream_tool_reasoning or _noop_async,
                extract_recent_tool_names=extract_recent_tool_names,
                deduplicate_trailing_messages=deduplicate_trailing_messages,
                hooks=self._hooks,
            )

            runner = MainLoopRunner(ctx)
            final_answer_text, _ = await runner.run(
                system_prompt=system_prompt,
                message_history=message_history,
                tool_definitions=tool_definitions,
                main_agent_prompt_instance=prompt_instance,
                task_engine_cfg=None,  # No reflection for sub-agents
                task_description=task_description,
                task_guidance="",
                keep_tool_result=effective_keep,
            )

            if not final_answer_text:
                final_answer_text = f"No final answer generated by sub agent {sub_agent_name}."

        except Exception as e:
            logger.error(f"Sub-agent '{sub_agent_name}' failed: {e}", exc_info=True)
            self.task_log.log_step(
                f"sub_{sub_agent_name}_error",
                f"Sub-agent failed: {type(e).__name__}: {str(e)[:200]}",
                "failed",
            )
            final_answer_text = f"[Sub-agent Error] {sub_agent_name} failed: {str(e)[:500]}"
        finally:
            # Merge sub-agent's offload registry into parent so parent's
            # cleanup_offload_files() reaches the shared offload dir.
            # Without this, offload records die with the sub-agent's isolated
            # ContextManager and the files become orphans on disk.
            if parent_context_manager is not None and context_manager is not None:
                try:
                    parent_context_manager.merge_offload_registry(context_manager)
                except Exception as merge_err:
                    logger.warning(
                        f"[SubAgent:{sub_agent_name}] offload registry merge failed: {merge_err}"
                    )
            if message_history:
                self.task_log.sub_agent_message_history_sessions[
                    self.task_log.current_sub_agent_session_id
                ] = {"system_prompt": system_prompt, "message_history": message_history}
            self.task_log.save()
            self.task_log.end_sub_agent_session(sub_agent_name)

        return final_answer_text

    async def spawn(
        self,
        task_description: str,
        *,
        parent_llm_client,
        parent_tool_executor,
        parent_tool_definitions: list,
        parent_callbacks: dict,
        keep_tool_result: int = -1,
        spawn_depth: int = 0,
        hooks_instance=None,
        parent_system_prompt: str | None = None,
        max_turns: int | None = None,
        parent_context_manager: ContextManager | None = None,
    ) -> str:
        """Spawn a temporary agent inheriting parent's LLM client and tools.

        Unlike run() which uses pre-configured sub-agents, spawn() creates a
        lightweight agent on-the-fly using the parent's resources.

        Args:
            task_description: Task for the spawned agent.
            parent_llm_client: LLM client to inherit.
            parent_tool_executor: ToolExecutor to reuse.
            parent_tool_definitions: Tool definitions (spawn_agent may be filtered by caller).
            parent_callbacks: Dict with keys: handle_llm_call, handle_summary,
                intercept_key_message, streaming_final_message, stream_tool_reasoning.
            keep_tool_result: Number of recent tool results to keep.
            spawn_depth: Current nesting depth for the spawned agent.
            hooks_instance: HookRegistry instance (defaults to module-level singleton).
            parent_system_prompt: Reuse parent's rendered system prompt (cache optimization).
            max_turns: Maximum turns for this spawn. If None, inherits from parent config.
        """
        from omegaconf import OmegaConf

        from mem_deep_research_core.core.message_utils import (
            deduplicate_trailing_messages,
            extract_recent_tool_names,
        )
        from mem_deep_research_core.core.task_planner import TaskPlanner

        task_description = self._parse_task_description(task_description)
        task_description += "\n\nPlease provide the answer and detailed supporting information."
        display_name = f"spawned-{id(task_description) % 10000}"

        logger.debug(f"\n=== Spawning Agent: {display_name} ===")

        context_manager = self._create_context_manager(
            parent_llm_client, parent_context_manager=parent_context_manager
        )

        monitor = ExecutionMonitor(
            config=MonitoringConfig(),
            stream_reasoning_callback=self.stream_tool_reasoning,
        )

        llm_handler = LLMCallHandler(
            main_llm_client=parent_llm_client,
            task_log=self.task_log,
            add_message_id=False,
            keep_tool_result=keep_tool_result,
            hooks=hooks_instance or self._hooks,
        )

        summary_handler = SummaryHandler(
            llm_call_handler=llm_handler,
            chinese_context=self.chinese_context,
            response_language=self.response_language,
        )
        summary_handler.context = self.context

        # Resolve spawn limits: LLM-provided > parent config > hardcoded default
        # Spawn agents handle focused subtasks — keep turns tight (cap at 10).
        parent_main = self.cfg.get("main_agent", {}) if self.cfg else {}
        _default_max_tool_calls = parent_main.get("max_tool_calls_per_turn", 5)
        effective_max_turns = max(1, min(max_turns or 5, 10))

        spawn_cfg = OmegaConf.create(
            {
                "main_agent": {
                    "max_turns": effective_max_turns,
                    "max_tool_calls_per_turn": _default_max_tool_calls,
                    "keep_tool_result": keep_tool_result,
                }
            }
        )

        _noop_async = _make_noop_async()
        from mem_deep_research_core.utils.tool_utils import _load_agent_prompt

        prompt_instance = _load_agent_prompt({"agent_type": "worker", "tool_format": "xml"})
        # Reuse parent's system prompt for cache optimization (byte-exact hit)
        if parent_system_prompt:
            # Strip the ## Language detection section — sub-agent already has a resolved
            # response_language and should not emit <response_language> tags.
            system_prompt = _strip_language_section(parent_system_prompt)
            logger.debug(f"[Spawn] Reusing parent system prompt ({len(system_prompt)} chars)")
        else:
            system_prompt = prompt_instance.generate_system_prompt_with_mcp_tools(
                mcp_servers=parent_tool_definitions,
                chinese_context=self.chinese_context,
                response_language=self.response_language,
            )

        ctx = MainLoopContext(
            cfg=spawn_cfg,
            monitor=monitor,
            context_manager=context_manager,
            stream_handler=self.stream_handler,
            tool_executor=parent_tool_executor,
            sub_agent_runner=None,
            llm_handler=llm_handler,
            summary_handler=summary_handler,
            task_planner=TaskPlanner(enabled=False),
            inline_skill_selector=None,
            llm_client=parent_llm_client,
            output_formatter=self.output_formatter,
            task_log=self.task_log,
            context=self.context,
            chinese_context=self.chinese_context,
            response_language=self.response_language,
            agent_name=display_name,
            handle_llm_call=parent_callbacks.get("handle_llm_call") or _noop_async,
            handle_summary=parent_callbacks.get("handle_summary") or _noop_async,
            intercept_key_message=parent_callbacks.get("intercept_key_message") or _noop_async,
            streaming_final_message=_noop_async,  # spawned agent must NOT stream its output; result returns as tool result
            stream_tool_reasoning=parent_callbacks.get("stream_tool_reasoning") or _noop_async,
            extract_recent_tool_names=extract_recent_tool_names,
            deduplicate_trailing_messages=deduplicate_trailing_messages,
            spawn_depth=spawn_depth,
            hooks=hooks_instance or self._hooks,
        )

        # Inject language instruction so sub-agent responds in the correct language
        # (task_description from LLM is often English even when user query is Chinese)
        if self.response_language and self.response_language not in ("auto", "English"):
            task_description = (
                f"[Language: respond in {self.response_language}]\n\n{task_description}"
            )

        message_history = [
            {"role": "user", "content": [{"type": "text", "text": task_description}]}
        ]
        runner = MainLoopRunner(ctx)

        try:
            result, _ = await runner.run(
                system_prompt=system_prompt,
                message_history=message_history,
                tool_definitions=parent_tool_definitions,
                main_agent_prompt_instance=prompt_instance,
                task_engine_cfg=None,
                task_description=task_description,
                task_guidance="",
                keep_tool_result=keep_tool_result,
            )
            return result or f"No answer generated by spawned agent {display_name}."
        except Exception as e:
            logger.error(f"Spawned agent '{display_name}' failed: {e}", exc_info=True)
            return f"[Spawn Error] {display_name} failed: {str(e)[:500]}"
        finally:
            # Merge sub-agent's offload registry into parent for cleanup
            if parent_context_manager is not None:
                parent_context_manager.merge_offload_registry(context_manager)
            if message_history:
                session_key = f"spawn_{display_name}"
                self.task_log.sub_agent_message_history_sessions[session_key] = {
                    "system_prompt": system_prompt,
                    "message_history": message_history,
                }


def _make_noop_async():
    """Create a no-op async callable for optional callback defaults."""

    async def _noop(*args, **kwargs):
        return None

    return _noop
