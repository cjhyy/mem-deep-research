#!/usr/bin/env python3
"""
Universal Agent Runner — handles everything from greetings to deep research.

Usage:
    python run.py "你好"                                  # Quick greeting
    python run.py "123 * 456 + 789"                       # Calculator
    python run.py "What is quantum computing?"            # Standard research
    python run.py "深入研究 AI Agent 的最新进展" --deep     # Deep research mode

Options:
    --deep      Force deep research mode (reflection + task tracking)
    --flash     Force flash mode (single response, no tools)
    --config    Config file name (default: agent)
    --verbose   Enable debug logging
"""

import argparse
import asyncio
import logging
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


async def main():
    parser = argparse.ArgumentParser(
        description="Universal Agent — quick answers to deep research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("task", help="Task or question")
    parser.add_argument("--config", default="agent", help="Config name (default: agent)")
    parser.add_argument("--deep", action="store_true", help="Force deep research mode")
    parser.add_argument("--flash", action="store_true", help="Force flash mode (no tools)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s: %(message)s")
    else:
        log_level = os.environ.get("LOGGER_LEVEL", "INFO")
        logging.basicConfig(level=getattr(logging, log_level), format="%(name)s %(levelname)s: %(message)s")

    from mem_deep_research import DeepResearch
    from omegaconf import OmegaConf

    dr = DeepResearch.from_project(PROJECT_DIR, config_name=args.config)

    # Override execution mode from CLI flags
    if args.flash:
        OmegaConf.update(dr._cfg, "main_agent.execution_mode", "flash")
    elif args.deep:
        OmegaConf.update(dr._cfg, "main_agent.execution_mode", "deep")

    result = await dr.run(args.task)

    # Pretty output
    print(f"\n{'=' * 60}")
    print(f"Status: {result.status} | Duration: {result.duration_seconds:.1f}s")
    print(f"{'=' * 60}")
    print(f"\n{result.answer}")

    if result.error:
        print(f"\nError: {result.error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
