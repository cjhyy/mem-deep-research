#!/usr/bin/env python3
"""
Advanced usage examples for Mem Deep Research.

Demonstrates:
  - Programmatic configuration (no YAML)
  - Custom context (including secure fields)
  - Batch execution
  - Tool listing and validation
  - Progress callback
"""

import asyncio
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.resolve()

# Load .env
env_file = PROJECT_DIR / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


async def example_programmatic():
    """Example 1: Programmatic configuration — no YAML needed."""
    from mem_deep_research import DeepResearch

    dr = DeepResearch(
        llm_provider="openrouter",
        model="anthropic/claude-sonnet-4",
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        max_turns=10,
        temperature=0.3,
        tools=["tool-calculator"],
        logs_dir=str(PROJECT_DIR / "logs"),
        interceptor_preset="default",
    )

    result = await dr.run("计算 (2^10 + 3^7) * 5 的结果")
    print(f"[Programmatic] Status: {result.status}")
    print(f"[Programmatic] Answer: {result.answer}")


async def example_with_context():
    """Example 2: Pass custom context (including secure fields)."""
    from mem_deep_research import DeepResearch

    dr = DeepResearch.from_project(PROJECT_DIR)

    # Context is injected into the system prompt.
    # Fields under _secure are masked as [SECURE:xxx] in the prompt
    # but automatically restored when passed to tool arguments.
    context = {
        "user_name": "Alice",
        "task_priority": "high",
        "_secure": {
            "user_id": "usr_abc123",
            "api_token": "secret-token-value",
        },
    }

    result = await dr.run("用计算器算一下 100 的阶乘是多少", context=context)
    print(f"[Context] Answer: {result.answer}")


async def example_batch():
    """Example 3: Batch execution — run multiple tasks."""
    from mem_deep_research import DeepResearch

    dr = DeepResearch.from_project(PROJECT_DIR, config_name="agent_minimal")

    tasks = [
        "1 + 1 等于多少？",
        "2 的 20 次方是多少？",
        "100 / 7 保留三位小数",
    ]

    results = await dr.run_batch(tasks, parallel=False)

    for task, result in zip(tasks, results):
        status = "OK" if result.status == "success" else "FAIL"
        print(f"[Batch] {status} | {task} → {result.answer[:80]}")


async def example_validate_and_list_tools():
    """Example 4: Validate config and list available tools."""
    from mem_deep_research import DeepResearch

    dr = DeepResearch.from_project(PROJECT_DIR)

    # Validate configuration
    validation = await dr.validate()
    print(f"[Validate] {validation}")

    # List available tools
    tools = await dr.list_tools()
    print(f"[Tools] Available: {tools}")


async def example_progress_callback():
    """Example 5: Progress callback for real-time status updates."""
    from mem_deep_research import DeepResearch

    dr = DeepResearch.from_project(PROJECT_DIR)

    def on_progress(event):
        event_type = event.get("type", "unknown")
        print(f"  [Progress] {event_type}: {event}")

    result = await dr.run(
        "123 * 456 + 789",
        on_progress=on_progress,
    )
    print(f"[Progress] Final: {result.answer}")


async def main():
    examples = {
        "programmatic": example_programmatic,
        "context": example_with_context,
        "batch": example_batch,
        "validate": example_validate_and_list_tools,
        "progress": example_progress_callback,
    }

    if len(sys.argv) > 1 and sys.argv[1] in examples:
        await examples[sys.argv[1]]()
    else:
        print("Usage: python run_advanced.py <example>")
        print(f"Available examples: {', '.join(examples.keys())}")
        print()
        # Default: run programmatic example
        print("Running 'programmatic' example by default...\n")
        await example_programmatic()


if __name__ == "__main__":
    asyncio.run(main())
