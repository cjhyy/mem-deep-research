#!/usr/bin/env python3
"""
Run a research task using Mem Deep Research.

Usage:
    python run.py "What is 123 * 456 + 789?"
    python run.py "Your research task" --config agent_anthropic
    python run.py "快速测试" --config agent_minimal

Available configs:
    agent           — OpenRouter + Claude (default)
    agent_anthropic — Anthropic direct API
    agent_minimal   — Minimal config for quick testing
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
        description="Run research agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See run_advanced.py for more usage patterns (context, batch, etc.)",
    )
    parser.add_argument("task", help="Research task description")
    parser.add_argument("--config", default="agent", help="Config name (default: agent)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s: %(message)s")
    else:
        log_level = os.environ.get("LOGGER_LEVEL", "INFO")
        logging.basicConfig(level=getattr(logging, log_level), format="%(name)s %(levelname)s: %(message)s")

    from mem_deep_research import DeepResearch

    dr = DeepResearch.from_project(PROJECT_DIR, config_name=args.config)
    result = await dr.run(args.task)

    print(f"\nStatus: {result.status}")
    print(f"Duration: {result.duration_seconds:.1f}s")
    print(f"\n{result.answer}")

    if result.error:
        print(f"\nError: {result.error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
