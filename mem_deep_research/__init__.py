"""
Mem Deep Research - AI Agent Orchestration Framework

A powerful framework for building deep research agents with LLM orchestration.

Usage:
    from mem_deep_research import DeepResearch

    # From project directory
    dr = DeepResearch.from_project("./my_project")
    result = await dr.run("Research the latest AI developments")

    # From config directory
    dr = DeepResearch.from_config_dir("./config")
    result = await dr.run("Your research task")

    # Programmatic configuration
    dr = DeepResearch(
        llm_provider="anthropic",
        model="claude-sonnet-4-20250514",
        api_key="your-api-key",
    )
    result = await dr.run("Your task")

    # Sync version
    result = dr.run_sync("Your task")

CLI:
    # Create a new project
    mem-deep-research init my_project

    # Run research
    python run.py "Your research task"
"""

from mem_deep_research_core.deep_research import DeepResearch, TaskResult

__version__ = "1.2.0"
__all__ = ["DeepResearch", "TaskResult", "__version__"]
