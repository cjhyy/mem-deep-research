import os
import pathlib
import time
import traceback
from datetime import datetime
from typing import Any

from omegaconf import DictConfig

from mem_deep_research_core.core.orchestrator import Orchestrator
from mem_deep_research_core.llm.client import LLMClient
from mem_deep_research_core.mem_deep_research_logging.logger import bootstrap_logger
from mem_deep_research_core.mem_deep_research_logging.task_tracer import TaskTracer
from mem_deep_research_core.tool.manager import ToolManager
from mem_deep_research_core.utils.io_utils import OutputFormatter
from mem_deep_research_core.utils.tool_utils import create_mcp_server_parameters

LOGGER_LEVEL = os.getenv("LOGGER_LEVEL", "INFO")
logger = bootstrap_logger(level=LOGGER_LEVEL)


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
) -> tuple[str, str, pathlib.Path]:
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
    try:
        # Initialize main agent LLM client
        if main_agent_llm_client is None:
            if hasattr(cfg.main_agent, "llm") and cfg.main_agent.llm is not None:
                main_agent_llm_client = LLMClient(task_id=task_id, llm_config=cfg.main_agent.llm)
            else:
                raise ValueError(
                    "No LLM configuration found in main_agent. Please ensure the agent configuration includes an LLM section."
                )

        # Initialize sub agent LLM client
        if sub_agent_llm_client is None:
            if cfg.sub_agents is not None and cfg.sub_agents:
                first_sub_agent = next(iter(cfg.sub_agents.values()))
                if hasattr(first_sub_agent, "llm") and first_sub_agent.llm is not None:
                    sub_agent_llm_client = LLMClient(
                        task_id=f"{task_id}_sub", llm_config=first_sub_agent.llm
                    )
                else:
                    raise ValueError(
                        "No LLM configuration found in sub-agent. Please ensure the agent configuration includes an LLM section."
                    )
            else:
                sub_agent_llm_client = None
                logger.info("No sub agents defined, using main agent only for the task")

        # Initialize orchestrator
        orchestrator = Orchestrator(
            main_agent_tool_manager=main_agent_tool_manager,
            sub_agent_tool_managers=sub_agent_tool_managers,
            llm_client=main_agent_llm_client,
            sub_agent_llm_client=sub_agent_llm_client,
            output_formatter=output_formatter,
            task_log=task_log,
            cfg=cfg,
            stream_queue=stream_queue,
            tool_definitions=tool_definitions,
            sub_agent_tool_definitions=sub_agent_tool_definitions,
            context=context,
        )

        task_log.status = "running"
        final_answer, final_boxed_answer = await orchestrator.run_main_agent(
            task_description=task_description,
            task_file_name=task_file_name,
            task_id=task_id,
            history=history,
        )

        task_log.final_boxed_answer = final_boxed_answer
        task_log.status = "completed"

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
        # Close LLM clients only if we created them (caller manages externally provided ones)
        for _owns, _client, _label in [
            (_owns_main_client, main_agent_llm_client, "main_agent"),
            (_owns_sub_client, sub_agent_llm_client, "sub_agent"),
        ]:
            if _owns and _client is not None:
                try:
                    if hasattr(_client, "close_async"):
                        await _client.close_async()
                    else:
                        _client.close()
                except Exception as e:
                    logger.warning(f"Error closing {_label} LLM client: {e}")

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

        return final_answer, final_boxed_answer, task_log.log_path


def create_pipeline_components(cfg: DictConfig, logs_dir: str | None = None):
    """
    Creates and initializes the core components of the agent pipeline.

    Args:
        cfg: The Hydra configuration object.

    Returns:
        Tuple of (main_agent_tool_manager, sub_agent_tool_managers, output_formatter)
    """
    # Create ToolManagers for main agent and sub-agents
    main_agent_mcp_server_configs, main_agent_blacklist = create_mcp_server_parameters(
        cfg, cfg.main_agent, logs_dir
    )
    main_agent_tool_manager = ToolManager(
        main_agent_mcp_server_configs,
        tool_blacklist=main_agent_blacklist,
    )

    sub_agent_tool_managers = {}
    if cfg.sub_agents is not None and cfg.sub_agents:
        for sub_agent in cfg.sub_agents:
            sub_agent_mcp_server_configs, sub_agent_blacklist = create_mcp_server_parameters(
                cfg, cfg.sub_agents[sub_agent], logs_dir
            )
            sub_agent_tool_manager = ToolManager(
                sub_agent_mcp_server_configs,
                tool_blacklist=sub_agent_blacklist,
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
