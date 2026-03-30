"""
最终答案处理模块

负责最终答案的后处理和格式化输出。
答案提取逻辑可通过 hooks 自定义（如项目级 GAIA/benchmark 特定提取）。
"""

import logging
import time
from typing import Any

from mem_deep_research_core.core.constants import FALLBACK_NO_ANSWER
from mem_deep_research_core.mem_deep_research_logging.logger import truncate_for_log
from mem_deep_research_core.mem_deep_research_logging.task_tracer import TaskTracer
from mem_deep_research_core.utils.io_utils import OutputFormatter

logger = logging.getLogger("mem_deep_research")


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
    is_simple_response: bool = False,
) -> tuple[str, str]:
    """后处理最终答案

    Returns:
        (final_summary, final_boxed_answer)
    """
    if final_answer_text:
        task_log.log_step("final_answer", "Final answer extracted successfully")
        task_log.log_step("final_answer_content", f"Content: {final_answer_text}")
    else:
        final_answer_text = FALLBACK_NO_ANSWER
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
