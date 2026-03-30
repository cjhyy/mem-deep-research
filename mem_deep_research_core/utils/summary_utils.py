import json
import os
import re

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from mem_deep_research_core.core.constants import generate_message_id as _generate_message_id
from mem_deep_research_core.mem_deep_research_logging.logger import bootstrap_logger
from mem_deep_research_core.prompts.template_loader import PromptTemplateLoader

LOGGER_LEVEL = os.getenv("LOGGER_LEVEL", "INFO")
logger = bootstrap_logger(level=LOGGER_LEVEL)

_DEFAULT_UTIL_TIMEOUT = 600  # seconds

# 模块级模板加载器
_loader = PromptTemplateLoader()


@retry(
    wait=wait_exponential(multiplier=5),
    stop=stop_after_attempt(3),
    retry_error_callback=lambda retry_state: logger.warning(
        f"Retry attempt {retry_state.attempt_number} for detect_response_language"
    ),
)
async def detect_response_language(
    query: str,
    api_key: str,
    base_url: str | None = None,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Detect the preferred response language based on user's query using LLM.

    Args:
        query: The user's input query/question
        api_key: OpenAI API key
        base_url: API base URL

    Returns:
        str: The detected language name (e.g., "English", "Chinese", "Japanese")
    """
    if not query or not query.strip():
        return "English"

    # Use environment variable as default, fallback to OpenAI
    if base_url is None:
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    client = AsyncOpenAI(api_key=api_key, timeout=30, base_url=base_url)

    instruction = _loader.load_template("language/detect_language")

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": instruction + query[:1000],
                },  # Truncate to avoid too long
            ],
            response_format={"type": "json_object"},
        )

        result = response.choices[0].message.content
        if not result or not result.strip():
            logger.warning("Language detection returned empty result, defaulting to English")
            return "English"

        try:
            output_lang = json.loads(result).get("language", "English")
        except (json.JSONDecodeError, KeyError):
            output_lang = "English"

        return output_lang
    except Exception as e:
        logger.warning(f"Language detection failed: {e}, defaulting to English")
        raise  # Let retry handle it


@retry(
    wait=wait_exponential(multiplier=15),
    stop=stop_after_attempt(5),
    retry_error_callback=lambda retry_state: logger.warning(
        f"Retry attempt {retry_state.attempt_number} for extract_hints"
    ),
)
async def extract_hints(
    question: str,
    api_key: str,
    chinese_context: bool,
    add_message_id: bool,
    base_url: str = "https://api.openai.com/v1",
    model: str = "o3",
) -> str:
    """Use LLM to extract task hints"""
    client = AsyncOpenAI(api_key=api_key, timeout=_DEFAULT_UTIL_TIMEOUT, base_url=base_url)

    instruction = _loader.load_template("hints/hint_instruction")

    # Add Chinese-specific instructions if enabled
    if chinese_context:
        instruction += _loader.load_template("hints/hint_chinese_supplement")

    # Add message ID for O3 messages (if configured)
    content = instruction + question
    if add_message_id:
        message_id = _generate_message_id()
        content = f"[{message_id}] {content}"

    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        reasoning_effort="high",
    )

    result = response.choices[0].message.content

    # Check if result is empty, raise exception to trigger retry if empty
    if not result or not result.strip():
        raise ValueError("Hint extraction returned empty result")

    return result


@retry(
    wait=wait_exponential(multiplier=15),
    stop=stop_after_attempt(5),
    retry_error_callback=lambda retry_state: logger.warning(
        f"Retry attempt {retry_state.attempt_number} for get_gaia_answer_type"
    ),
)
async def get_gaia_answer_type(
    task_description: str, api_key: str, base_url: str = "https://api.openai.com/v1",
    model: str = "gpt-4.1",
) -> str:
    client = AsyncOpenAI(api_key=api_key, timeout=_DEFAULT_UTIL_TIMEOUT, base_url=base_url)

    instruction = _loader.load_and_render(
        "extraction/gaia_answer_type",
        task_description=task_description,
    )
    logger.debug(f"Answer type instruction: {instruction}")

    message_id = _generate_message_id()
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": f"[{message_id}] {instruction}"}],
    )
    answer_type = response.choices[0].message.content
    # Check if result is empty, raise exception to trigger retry if empty
    if not answer_type or not answer_type.strip():
        raise ValueError("answer type returned empty result")

    logger.debug(f"Answer type: {answer_type}")

    return answer_type.strip()


@retry(
    wait=wait_exponential(multiplier=15),
    stop=stop_after_attempt(5),
    retry_error_callback=lambda retry_state: logger.warning(
        f"Retry attempt {retry_state.attempt_number} for extract_gaia_final_answer"
    ),
)
async def extract_gaia_final_answer(
    task_description_detail: str,
    summary: str,
    api_key: str,
    chinese_context: bool,
    base_url: str = "https://api.openai.com/v1",
    model: str = "anthropic/claude-sonnet-4-20250514",
) -> str:
    """Use LLM to extract final answer from summary

    Args:
        model: LLM model to use for extraction. Default is claude-sonnet-4-20250514 which is
               more cost-effective than opus while still providing good quality.
    """
    answer_type = await get_gaia_answer_type(task_description_detail, api_key, base_url)

    client = AsyncOpenAI(api_key=api_key, timeout=_DEFAULT_UTIL_TIMEOUT, base_url=base_url)

    # Build confidence section from templates
    if chinese_context:
        chinese_supplement = _loader.load_template("extraction/gaia_chinese_supplement")
    else:
        chinese_supplement = ""

    confidence_section = _loader.load_template("extraction/gaia_confidence")
    combined_confidence_section = confidence_section + "\n" + chinese_supplement

    # Select template by answer type
    template_name = (
        f"extraction/gaia_extract_{answer_type}"
        if answer_type in ["number", "time"]
        else "extraction/gaia_extract_string"
    )

    full_prompt = _loader.load_and_render(
        template_name,
        task_description=task_description_detail,
        summary=summary,
        confidence_section=combined_confidence_section,
    )

    logger.debug("Extract Final Answer Prompt:")
    logger.debug(full_prompt)

    message_id = _generate_message_id()
    logger.info(f"Using model for final answer extraction: {model}")
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": f"[{message_id}] {full_prompt}"}],
    )
    result = response.choices[0].message.content

    # Check if result is empty, raise exception to trigger retry if empty
    if not result or not result.strip():
        raise ValueError("Final answer extraction returned empty result")

    # Verify boxed answer exists
    boxed_match = re.search(r"\\boxed{([^}]*)}", result)
    if not boxed_match:
        raise ValueError("Final answer extraction returned empty answer")

    logger.debug(f"response: {result}")

    # Return the full response directly for downstream LLM processing
    # This contains all structured information: analysis, boxed answer, confidence, evidence, and weaknesses
    return result


@retry(
    wait=wait_exponential(multiplier=15),
    stop=stop_after_attempt(5),
    retry_error_callback=lambda retry_state: logger.warning(
        f"Retry attempt {retry_state.attempt_number} for extract_browsecomp_zh_final_answer"
    ),
)
async def extract_browsecomp_zh_final_answer(
    task_description_detail: str,
    summary: str,
    api_key: str,
    base_url: str = "https://api.openai.com/v1",
    model: str = "anthropic/claude-sonnet-4-20250514",
) -> str:
    """Use LLM to extract final answer from summary

    Args:
        model: LLM model to use for extraction. Default is claude-sonnet-4-20250514.
    """
    client = AsyncOpenAI(api_key=api_key, timeout=_DEFAULT_UTIL_TIMEOUT, base_url=base_url)

    full_prompt = _loader.load_and_render(
        "extraction/browsecomp_zh",
        task_description=task_description_detail,
        summary=summary,
    )

    logger.debug("Extract Final Answer Prompt:")
    logger.debug(full_prompt)

    message_id = _generate_message_id()
    logger.info(f"Using model for final answer extraction: {model}")

    # Build request params - only add reasoning_effort for o3 model
    request_params = {
        "model": model,
        "messages": [{"role": "user", "content": f"[{message_id}] {full_prompt}"}],
    }
    if model.startswith("o3") or model.startswith("openai/o3"):
        request_params["reasoning_effort"] = "medium"

    response = await client.chat.completions.create(**request_params)
    result = response.choices[0].message.content

    # Check if result is empty, raise exception to trigger retry if empty
    if not result or not result.strip():
        raise ValueError("Final answer extraction returned empty result")

    # Verify boxed answer exists
    boxed_match = re.search(r"\\boxed{([^}]*)}", result)
    if not boxed_match:
        raise ValueError("Final answer extraction returned empty answer")

    logger.debug(f"response: {result}")

    # Return the full response directly for downstream LLM processing
    # This contains all structured information: analysis, boxed answer, confidence, evidence, and weaknesses
    return result
