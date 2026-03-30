"""
最终答案处理模块

负责最终答案的后处理、提取和格式化输出。
从 orchestrator.py 中拆分出来以降低复杂度。
"""

import logging
from typing import Any

from mem_deep_research_core.mem_deep_research_logging.logger import truncate_for_log
from mem_deep_research_core.mem_deep_research_logging.task_tracer import TaskTracer
from mem_deep_research_core.utils.io_utils import OutputFormatter
from mem_deep_research_core.utils.summary_utils import (
    extract_browsecomp_zh_final_answer,
    extract_gaia_final_answer,
)

logger = logging.getLogger("mem_deep_research")


async def extract_final_answer(
    cfg,
    task_description: str,
    final_answer_text: str,
    message_history: list,
    chinese_context: bool,
    task_log: TaskTracer,
) -> str:
    """提取最终答案

    Args:
        cfg: Agent 配置
        task_description: 任务描述
        final_answer_text: 原始最终答案文本
        message_history: 消息历史
        chinese_context: 是否中文上下文
        task_log: 任务日志

    Returns:
        处理后的最终答案文本
    """
    try:
        final_answer_model = cfg.main_agent.output_process.get(
            "final_answer_model", "anthropic/claude-sonnet-4.5"
        )
        final_answer_base_url = cfg.main_agent.output_process.get(
            "final_answer_llm_base_url", "https://openrouter.ai/api/v1"
        )

        if "browsecomp-zh" in cfg.benchmark.name:
            extracted_answer = await extract_browsecomp_zh_final_answer(
                task_description,
                final_answer_text,
                cfg.main_agent.openai_api_key,
                final_answer_base_url,
                model=final_answer_model,
            )
            message_history.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"LLM extracted final answer:\n{extracted_answer}"}
                    ],
                }
            )
            return extracted_answer
        else:
            extracted_answer = await extract_gaia_final_answer(
                task_description,
                final_answer_text,
                cfg.main_agent.openai_api_key,
                chinese_context,
                final_answer_base_url,
                model=final_answer_model,
            )
            message_history.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"LLM extracted final answer:\n{extracted_answer}"}
                    ],
                }
            )
            return f"{final_answer_text}\n\nLLM Extracted Answer:\n{extracted_answer}"

    except Exception as e:
        logger.error(f"Final answer extraction failed: {str(e)}")
        task_log.log_step("final_answer_extraction", f"[ERROR] Failed: {str(e)}", "failed")
        return final_answer_text


async def post_process_final_answer(
    cfg,
    final_answer_text: str,
    task_description: str,
    message_history: list,
    system_prompt: str,
    chinese_context: bool,
    task_log: TaskTracer,
    output_formatter: OutputFormatter,
    llm_client: Any,
) -> tuple[str, str]:
    """后处理最终答案

    Args:
        cfg: Agent 配置
        final_answer_text: 原始最终答案
        task_description: 任务描述
        message_history: 消息历史
        system_prompt: 系统提示词
        chinese_context: 是否中文上下文
        task_log: 任务日志
        output_formatter: 输出格式化器
        llm_client: LLM 客户端

    Returns:
        (final_summary, final_boxed_answer) 格式化后的摘要和提取的答案
    """
    if final_answer_text:
        task_log.log_step("final_answer", "Final answer extracted successfully")
        task_log.log_step("final_answer_content", f"Content: {final_answer_text}")

        # 提取最终答案（如果启用）
        if cfg.main_agent.output_process.final_answer_extraction:
            final_answer_text = await extract_final_answer(
                cfg,
                task_description,
                final_answer_text,
                message_history,
                chinese_context,
                task_log,
            )
    else:
        final_answer_text = "No final answer generated."
        task_log.log_step("final_answer", "Failed to extract final answer", "failed")

    logger.debug(f"LLM Final Answer: {truncate_for_log(final_answer_text)}")

    # 保存最终消息历史
    task_log.main_agent_message_history = {
        "system_prompt": system_prompt,
        "message_history": message_history,
    }
    task_log.save()

    # 格式化输出
    task_log.log_step("format_output", "Formatting final output")
    final_summary, final_boxed_answer = output_formatter.format_final_summary_and_log(
        final_answer_text, llm_client
    )

    return final_summary, final_boxed_answer
