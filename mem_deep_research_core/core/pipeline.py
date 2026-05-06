import os
import pathlib
import time
import traceback
from datetime import datetime
from typing import Any

from omegaconf import DictConfig

from mem_deep_research_core.core.hitl.exceptions import (
    HitlRejectedError,
    PendingHumanException,
)
from mem_deep_research_core.core.orchestrator import Orchestrator
from mem_deep_research_core.llm.client import LLMClient
from mem_deep_research_core.mem_deep_research_logging.logger import bootstrap_logger
from mem_deep_research_core.mem_deep_research_logging.task_tracer import TaskTracer
from mem_deep_research_core.tool.manager import ToolManager
from mem_deep_research_core.utils.io_utils import OutputFormatter
from mem_deep_research_core.utils.tool_utils import create_mcp_server_parameters

LOGGER_LEVEL = os.getenv("LOGGER_LEVEL", "INFO")
logger = bootstrap_logger(level=LOGGER_LEVEL)


def _build_llm_clients(
    cfg: DictConfig,
    task_id: str,
    *,
    main_agent_llm_client: Any | None,
    sub_agent_llm_client: Any | None,
    include_router: bool = True,
) -> tuple[Any, Any | None, Any | None]:
    """Create (or pass through) the three LLM clients a pipeline needs.

    Returns ``(main, sub, router)`` where any component that was pre-supplied
    by the caller is returned unchanged. The router is only created when
    ``include_router=True`` and ``cfg.main_agent.llm.router_model`` is set
    under the auto execution mode.

    Raises ``ValueError`` on missing LLM config so the caller can surface a
    clean startup error instead of failing deep inside the main loop.
    """
    if main_agent_llm_client is None:
        if hasattr(cfg.main_agent, "llm") and cfg.main_agent.llm is not None:
            main_agent_llm_client = LLMClient(
                task_id=task_id, llm_config=cfg.main_agent.llm
            )
        else:
            raise ValueError(
                "No LLM configuration found in main_agent. Please ensure the "
                "agent configuration includes an LLM section."
            )

    if sub_agent_llm_client is None:
        if getattr(cfg, "sub_agents", None):
            first_sub_agent = next(iter(cfg.sub_agents.values()))
            if hasattr(first_sub_agent, "llm") and first_sub_agent.llm is not None:
                sub_agent_llm_client = LLMClient(
                    task_id=f"{task_id}_sub", llm_config=first_sub_agent.llm
                )
            else:
                raise ValueError(
                    "No LLM configuration found in sub-agent. Please ensure "
                    "the agent configuration includes an LLM section."
                )
        else:
            sub_agent_llm_client = None
            logger.info("No sub agents defined, using main agent only for the task")

    router_llm_client = None
    if include_router:
        router_model = cfg.main_agent.get("llm", {}).get("router_model")
        if router_model and cfg.main_agent.get("execution_mode", "auto") == "auto":
            try:
                from copy import deepcopy

                from omegaconf import OmegaConf

                router_llm_cfg = deepcopy(cfg.main_agent.llm)
                router_llm_cfg = OmegaConf.to_container(router_llm_cfg, resolve=True)
                router_llm_cfg["model_name"] = router_model
                router_llm_cfg["max_tokens"] = 64  # router only needs a few tokens
                router_llm_cfg["enable_streaming"] = False
                router_llm_cfg = OmegaConf.create(router_llm_cfg)
                router_llm_client = LLMClient(
                    task_id=f"{task_id}_router", llm_config=router_llm_cfg
                )
                logger.info(f"[Pipeline] Router LLM client created: model={router_model}")
            except Exception as e:
                logger.warning(f"[Pipeline] Failed to create router LLM client: {e}")

    return main_agent_llm_client, sub_agent_llm_client, router_llm_client


async def _close_llm_clients(
    clients: list[tuple[bool, Any | None, str]],
) -> None:
    """Close the pipeline-owned LLM clients, tolerating close failures per-client."""
    for _owns, _client, _label in clients:
        if _owns and _client is not None:
            try:
                if hasattr(_client, "close_async"):
                    await _client.close_async()
                else:
                    _client.close()
            except Exception as e:
                logger.warning(f"Error closing {_label} LLM client: {e}")


def _get_hitl_checkpoint_dir(cfg: DictConfig, log_path: pathlib.Path) -> pathlib.Path:
    """Resolve where HITL checkpoints are written.

    Precedence: ``cfg.hitl.checkpoint_dir`` → ``cfg.output_dir`` → log file
    directory. Falling back to the log directory keeps single-run users
    working without extra configuration.
    """
    hitl_cfg = cfg.get("hitl", {}) if hasattr(cfg, "get") else {}
    explicit = hitl_cfg.get("checkpoint_dir") if hitl_cfg else None
    if explicit:
        return pathlib.Path(explicit)
    output_dir = cfg.get("output_dir") if hasattr(cfg, "get") else None
    if output_dir:
        return pathlib.Path(output_dir)
    return pathlib.Path(log_path).parent


async def execute_task_pipeline(
    cfg: DictConfig,
    task_name: str,
    task_id: str,
    task_description: str,
    task_file_name: str | None,
    main_agent_tool_manager: ToolManager,
    sub_agent_tool_managers: dict[str, ToolManager],
    output_formatter: OutputFormatter,
    log_path: pathlib.Path,
    ground_truth: str | None = None,
    metadata: dict | None = None,
    stream_queue: Any | None = None,
    tool_definitions: list[dict[str, Any]] | None = None,
    sub_agent_tool_definitions: dict[str, list[dict[str, Any]]] | None = None,
    history: list[dict[str, Any]] | None = None,
    context: dict[str, Any] | None = None,
    main_agent_llm_client: Any | None = None,
    sub_agent_llm_client: Any | None = None,
    resume_from: dict | None = None,
    runtime: Any | None = None,
) -> tuple[str, str, pathlib.Path, str, str | None, Any | None]:
    """
    Executes the full pipeline for a single task.

    Args:
        cfg: The Hydra configuration object.
        task_description: The description of the task for the LLM.
        task_file_name: The path to an associated file (optional).
        task_id: A unique identifier for this task run (used for logging).
        main_agent_tool_manager: An initialized main agent ToolManager instance.
        sub_agent_tool_managers: A dictionary of initialized sub-agent ToolManager instances.
        output_formatter: An initialized OutputFormatter instance.
        ground_truth: The ground truth for the task (optional).
        log_path: The path to save the task log.
        context: Optional user context dict with user_id, org_id, room_id, timezone, trace_id
            for passing to MCP tools.
        main_agent_llm_client: Optional pre-created LLM client. If None, a new client
            is created internally and closed after the task. If provided, the caller
            is responsible for its lifecycle (pipeline will NOT close it).
        sub_agent_llm_client: Optional pre-created sub-agent LLM client. Same
            lifecycle semantics as main_agent_llm_client.

    Returns:
        A tuple containing:
        - A string with the final execution log and summary, or an error message.
        - The final boxed answer.
        - The path to the log file.
    """
    logger.debug(f"Starting Task Execution: {task_id}")
    _perf_pipeline_start = time.perf_counter()

    # Create task log
    task_log = TaskTracer(
        log_path=log_path,
        task_name=task_name,
        task_id=task_id,
        task_file_name=task_file_name,
        ground_truth=ground_truth,
        input={
            "task_description": task_description,
            "task_file_name": task_file_name,
            "metadata": metadata or {},
        },
    )

    # Track whether we created the LLM clients (and must close them)
    _owns_main_client = main_agent_llm_client is None
    _owns_sub_client = sub_agent_llm_client is None
    final_answer, final_boxed_answer = "", ""
    # HITL outcome — populated only when the task suspends awaiting human.
    hitl_checkpoint_id: str | None = None
    hitl_pending_request: Any | None = None
    try:
        main_agent_llm_client, sub_agent_llm_client, router_llm_client = _build_llm_clients(
            cfg,
            task_id,
            main_agent_llm_client=main_agent_llm_client,
            sub_agent_llm_client=sub_agent_llm_client,
            include_router=True,
        )

        # Initialize orchestrator
        orchestrator = Orchestrator(
            main_agent_tool_manager=main_agent_tool_manager,
            sub_agent_tool_managers=sub_agent_tool_managers,
            llm_client=main_agent_llm_client,
            sub_agent_llm_client=sub_agent_llm_client,
            router_llm_client=router_llm_client,
            output_formatter=output_formatter,
            task_log=task_log,
            cfg=cfg,
            stream_queue=stream_queue,
            tool_definitions=tool_definitions,
            sub_agent_tool_definitions=sub_agent_tool_definitions,
            context=context,
            runtime=runtime,
        )

        task_log.status = "running"
        final_answer, final_boxed_answer = await orchestrator.run_main_agent(
            task_description=task_description,
            task_file_name=task_file_name,
            task_id=task_id,
            history=history,
            resume_from=resume_from,
        )

        task_log.final_boxed_answer = final_boxed_answer
        task_log.status = "completed"

    except PendingHumanException as pending:
        # HITL durable suspend — persist the RuntimeSnapshot attached by the
        # main loop and propagate enough metadata back to agent_factory for
        # the resume entry-point to recover.
        from mem_deep_research_core.core.hitl.checkpoint_store import (
            FilesystemCheckpointStore,
        )

        checkpoint_dir = _get_hitl_checkpoint_dir(cfg, log_path)
        store = FilesystemCheckpointStore(checkpoint_dir)
        snapshot = pending.snapshot
        if snapshot is None:
            # Defensive: if the main loop didn't attach a snapshot, log and
            # fall through to the generic failed path below.
            logger.error(
                "[Pipeline] PendingHumanException without snapshot — HITL "
                "suspend cannot be persisted. Treating as failed."
            )
            task_log.status = "failed"
            task_log.error = f"HITL suspend without snapshot: {pending}"
            final_answer = f"Error: HITL suspend missing snapshot ({pending})"
        else:
            hitl_checkpoint_id = await store.save(snapshot)
            pending.request.checkpoint_id = hitl_checkpoint_id
            hitl_pending_request = pending.request
            task_log.status = "awaiting_human"
            task_log.log_step(
                "hitl_suspend",
                f"Task suspended awaiting human decision "
                f"(checkpoint={hitl_checkpoint_id}, request={pending.request.request_id})",
            )
            # final_answer stays empty — no answer yet. boxed stays empty too.
            final_answer = ""
            final_boxed_answer = ""

    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"An error occurred during task {task_id}", exc_info=True)

        final_answer = (
            f"Error executing task {task_id}:\n"
            f"Description: {task_description}\n"
            f"File: {task_file_name}\n"
            f"Error Type: {type(e).__name__}\n"
            f"Error Details:\n{error_details}"
        )

        task_log.status = "interrupted"
        task_log.error = error_details

    finally:
        await _close_llm_clients(
            [
                (_owns_main_client, main_agent_llm_client, "main_agent"),
                (_owns_sub_client, sub_agent_llm_client, "sub_agent"),
                (True, router_llm_client, "router"),
            ]
        )

        task_log.end_time = datetime.now()
        task_log.record_perf("total_pipeline_duration", time.perf_counter() - _perf_pipeline_start)

        # Log performance summary
        perf_summary = task_log.get_perf_summary()
        logger.info(f"[Task {task_id}] {perf_summary}")

        # Record task summary to structured log
        task_log.log_step(
            "task_execution_finished",
            f"Task {task_id} execution completed with status: {task_log.status}",
        )
        task_log.log_step("perf_metrics", perf_summary)
        task_log.save()

        # Cleanup TaskTracer to release memory
        task_log.cleanup()

        logger.debug(f"--- Finished Task Execution: {task_id} ---")

        return (
            final_answer,
            final_boxed_answer,
            task_log.log_path,
            task_log.status,
            hitl_checkpoint_id,
            hitl_pending_request,
        )


async def execute_hitl_resume_pipeline(
    cfg: DictConfig,
    task_id: str,
    checkpoint_id: str,
    decision: Any,
    main_agent_tool_manager: ToolManager,
    sub_agent_tool_managers: dict[str, ToolManager],
    output_formatter: OutputFormatter,
    log_path: pathlib.Path,
    *,
    task_description: str | None = None,
    tool_definitions: list[dict[str, Any]] | None = None,
    stream_queue: Any | None = None,
    history: list[dict[str, Any]] | None = None,
    context: dict[str, Any] | None = None,
    main_agent_llm_client: Any | None = None,
    sub_agent_llm_client: Any | None = None,
    runtime: Any | None = None,
    task_guidance: str = "",
) -> tuple[str, str, pathlib.Path, str, str | None, Any | None]:
    """Resume a HITL-suspended task with a human decision.

    Mirrors the setup of :func:`execute_task_pipeline` but calls
    ``MainLoopRunner.run_from_tool_cursor`` instead of ``run``. The caller
    must supply the same tool-manager / LLM client surface used for the
    original task so the tool batch in the snapshot resolves correctly.

    ``task_description`` is auto-resolved from the snapshot when omitted.
    Pass it explicitly only to override (rare — useful for tests / migrating
    pre-v1.3.0 checkpoints that predate the snapshot field).

    Note: a second HITL suspend during resume is checkpointed as a fresh
    record (the old checkpoint is deleted after this call completes).
    """
    from mem_deep_research_core.core.hitl.checkpoint_store import (
        FilesystemCheckpointStore,
    )

    logger.info(f"[HITL resume] task_id={task_id} checkpoint={checkpoint_id}")
    _perf_pipeline_start = time.perf_counter()

    checkpoint_dir = _get_hitl_checkpoint_dir(cfg, log_path)
    store = FilesystemCheckpointStore(checkpoint_dir)
    snapshot = await store.load(checkpoint_id)

    # Auto-resolve task_description from snapshot when caller didn't supply.
    if not task_description:
        task_description = snapshot.task_description
    if not task_description:
        raise ValueError(
            f"Cannot resume checkpoint {checkpoint_id}: snapshot has no "
            "task_description (likely a pre-v1.3.0 checkpoint). Pass "
            "task_description= explicitly."
        )

    task_log = TaskTracer(
        log_path=log_path,
        task_name="agent_task_resume",
        task_id=task_id,
        input={"task_description": task_description, "resume_checkpoint": checkpoint_id},
    )
    task_log.start_time = datetime.now()

    _owns_main_client = main_agent_llm_client is None
    _owns_sub_client = sub_agent_llm_client is None
    final_answer, final_boxed_answer = "", ""
    hitl_checkpoint_id: str | None = None
    hitl_pending_request: Any | None = None

    try:
        # Rebuild LLM clients the same way as a fresh run. Resume does not
        # need the router — mode is resolved from the snapshot.
        main_agent_llm_client, sub_agent_llm_client, _router = _build_llm_clients(
            cfg,
            task_id,
            main_agent_llm_client=main_agent_llm_client,
            sub_agent_llm_client=sub_agent_llm_client,
            include_router=False,
        )

        # Build orchestrator the same way as the fresh-run path.
        orchestrator = Orchestrator(
            main_agent_tool_manager=main_agent_tool_manager,
            sub_agent_tool_managers=sub_agent_tool_managers,
            llm_client=main_agent_llm_client,
            sub_agent_llm_client=sub_agent_llm_client,
            router_llm_client=None,
            output_formatter=output_formatter,
            task_log=task_log,
            cfg=cfg,
            stream_queue=stream_queue,
            tool_definitions=tool_definitions,
            sub_agent_tool_definitions=None,
            context=context,
            runtime=runtime,
        )

        # Drive runner via orchestrator helpers so init parity with fresh-run
        # stays tight. The orchestrator's _create_main_loop_runner wires every
        # component the runner needs.
        runner = orchestrator._create_main_loop_runner()

        # The orchestrator's normal run does system-prompt rebuild etc; for
        # resume we can reuse the orchestrator's prompt builder directly.
        (
            system_prompt,
            main_agent_prompt_instance,
            task_engine_cfg,
        ) = orchestrator.prompt_builder.build_system_prompt(
            tool_definitions or [],
            task_description,
            [],
        )

        task_log.status = "running"

        final_answer_text, _is_simple = await runner.run_from_tool_cursor(
            snapshot,
            decision,
            system_prompt=system_prompt,
            main_agent_prompt_instance=main_agent_prompt_instance,
            task_engine_cfg=task_engine_cfg,
            task_description=task_description,
            task_guidance=task_guidance,
            tool_definitions=tool_definitions or [],
            keep_tool_result=cfg.main_agent.get("keep_tool_result", -1),
        )

        # Post-process final answer.
        from mem_deep_research_core.core.answer_handler import (
            post_process_final_answer,
        )

        final_summary, final_boxed_answer = await post_process_final_answer(
            cfg=cfg,
            final_answer_text=final_answer_text,
            task_description=task_description,
            message_history=[],  # resume already folded history into runner
            system_prompt=system_prompt,
            chinese_context=False,
            task_log=task_log,
            output_formatter=output_formatter,
            llm_client=main_agent_llm_client,
            is_simple_response=False,
            context=context,
            hooks=orchestrator._hooks,
        )
        final_answer = final_summary
        task_log.final_boxed_answer = final_boxed_answer
        task_log.status = "completed"

        # Clean up the now-consumed checkpoint (best-effort).
        try:
            await store.delete(checkpoint_id)
        except Exception as cleanup_err:  # pragma: no cover
            logger.warning(
                "[HITL resume] Failed to delete consumed checkpoint %s: %s",
                checkpoint_id,
                cleanup_err,
            )

    except PendingHumanException as pending:
        snapshot_out = pending.snapshot
        if snapshot_out is None:
            task_log.status = "failed"
            task_log.error = f"HITL suspend without snapshot on resume: {pending}"
            final_answer = f"Error: HITL resume re-suspend missing snapshot ({pending})"
        else:
            hitl_checkpoint_id = await store.save(snapshot_out)
            pending.request.checkpoint_id = hitl_checkpoint_id
            hitl_pending_request = pending.request
            task_log.status = "awaiting_human"
            task_log.log_step(
                "hitl_suspend_on_resume",
                f"Resume re-suspended awaiting human decision "
                f"(new checkpoint={hitl_checkpoint_id}, request={pending.request.request_id})",
            )

    except HitlRejectedError as rejected:
        # rejection_strategy="abort_task" path — translate to failed status.
        # Distinct from "interrupted" (unexpected error) because this is a
        # deliberate human decision, recorded for audit.
        task_log.status = "failed"
        task_log.error = (
            f"HITL rejected by {rejected.decision.decided_by or 'approver'}: "
            f"{rejected.decision.reason or 'user rejected'}"
        )
        task_log.log_step(
            "hitl_rejected_abort",
            f"Task aborted on human rejection (request={rejected.request.request_id}, "
            f"strategy=abort_task)",
        )
        final_answer = f"[HITL rejected] {rejected.decision.reason or 'user rejected'}"

    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"HITL resume failed for task {task_id}", exc_info=True)
        final_answer = f"Error resuming task {task_id}: {type(e).__name__}: {e}"
        task_log.status = "interrupted"
        task_log.error = error_details

    finally:
        await _close_llm_clients(
            [
                (_owns_main_client, main_agent_llm_client, "main_agent"),
                (_owns_sub_client, sub_agent_llm_client, "sub_agent"),
            ]
        )

        task_log.end_time = datetime.now()
        task_log.record_perf(
            "total_pipeline_duration", time.perf_counter() - _perf_pipeline_start
        )
        task_log.save()
        task_log.cleanup()

    return (
        final_answer,
        final_boxed_answer,
        task_log.log_path,
        task_log.status,
        hitl_checkpoint_id,
        hitl_pending_request,
    )


def create_pipeline_components(cfg: DictConfig, logs_dir: str | None = None, *, runtime=None):
    """
    Creates and initializes the core components of the agent pipeline.

    Args:
        cfg: The Hydra configuration object.
        logs_dir: Optional log directory path.
        runtime: AgentRuntime instance for hooks/config_loader injection.

    Returns:
        Tuple of (main_agent_tool_manager, sub_agent_tool_managers, output_formatter)
    """
    # Extract runtime dependencies
    _hook_registry = runtime.hooks if runtime else None
    _config_loader = runtime.config_loader if runtime else None

    # Create ToolManagers for main agent and sub-agents
    main_agent_mcp_server_configs, main_agent_blacklist = create_mcp_server_parameters(
        cfg, cfg.main_agent, logs_dir, config_loader=_config_loader
    )
    main_agent_tool_manager = ToolManager(
        main_agent_mcp_server_configs,
        tool_blacklist=main_agent_blacklist,
        hook_registry=_hook_registry,
    )

    sub_agent_tool_managers = {}
    if getattr(cfg, "sub_agents", None):
        for sub_agent in cfg.sub_agents:
            sub_agent_mcp_server_configs, sub_agent_blacklist = create_mcp_server_parameters(
                cfg, cfg.sub_agents[sub_agent], logs_dir, config_loader=_config_loader
            )
            sub_agent_tool_manager = ToolManager(
                sub_agent_mcp_server_configs,
                tool_blacklist=sub_agent_blacklist,
                hook_registry=_hook_registry,
            )
            sub_agent_tool_managers[sub_agent] = sub_agent_tool_manager

    # Create OutputFormatter with context-aware tool result limit
    max_context_length = cfg.main_agent.llm.get("max_context_length", -1)
    if max_context_length > 0:
        # ~5% of context budget per tool result (chars ≈ tokens * 4)
        max_tool_result_chars = max(10_000, (max_context_length * 4) // 20)
    else:
        max_tool_result_chars = 30_000  # sensible default
    output_formatter = OutputFormatter(max_tool_result_chars=max_tool_result_chars)

    return main_agent_tool_manager, sub_agent_tool_managers, output_formatter
