"""
CLI 入口

提供 `mem-deep-research` 命令行工具。

Usage:
    mem-deep-research run "Your research task" --project ./my_project
    mem-deep-research init my_project --provider openrouter
    mem-deep-research test --project ./my_project
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path


def cmd_init(args: argparse.Namespace) -> None:
    """初始化新项目"""
    from mem_deep_research_core.cli.templates import ProjectTemplate

    target = Path(args.name).resolve()
    if target.exists() and any(target.iterdir()):
        print(f"Directory '{target}' is not empty. Use --force to overwrite.")
        if not args.force:
            sys.exit(1)

    template = ProjectTemplate()
    template.create_project(
        target_dir=target,
        project_name=args.name,
        llm_provider=args.provider,
        chinese=args.chinese,
    )
    print(f"Project created at: {target}")
    print("\nNext steps:")
    print(f"  cd {args.name}")
    print("  # Edit .env with your API keys")
    print("  python run.py --test")
    print('  python run.py "Your research task"')


def cmd_run(args: argparse.Namespace) -> None:
    """运行研究任务"""
    project_dir = Path(args.project).resolve()
    if not (project_dir / "config").exists():
        print(f"Error: No config/ directory found in {project_dir}")
        sys.exit(1)

    # 加载 .env
    _load_env(project_dir / ".env")

    from mem_deep_research import DeepResearch

    dr = DeepResearch.from_project(project_dir)
    result = asyncio.run(dr.run(args.task))

    print(f"\nStatus: {result.status}")
    print(f"Duration: {result.duration_seconds:.1f}s")
    print(f"\n{result.answer}")

    if result.error:
        print(f"\nError: {result.error}", file=sys.stderr)
        sys.exit(1)


def cmd_test(args: argparse.Namespace) -> None:
    """测试项目配置"""
    project_dir = Path(args.project).resolve()

    print(f"Project: {project_dir}")

    # 检查配置
    config_path = project_dir / "config" / "agent.yaml"
    if config_path.exists():
        print(f"  Config: OK ({config_path})")
    else:
        print(f"  Config: MISSING ({config_path})")
        sys.exit(1)

    # 加载 .env
    env_path = project_dir / ".env"
    if env_path.exists():
        _load_env(env_path)
        print("  .env: OK")
    else:
        print("  .env: not found (using system env)")

    # 测试导入
    try:
        from mem_deep_research import DeepResearch

        print("  Import: OK")
    except ImportError as e:
        print(f"  Import: FAILED ({e})")
        sys.exit(1)

    # 测试配置加载
    try:
        DeepResearch.from_project(project_dir)
        print("  Config loading: OK")
    except Exception as e:
        print(f"  Config loading: FAILED ({e})")
        sys.exit(1)

    print("\nAll checks passed.")


def _load_env(env_file: Path) -> None:
    """加载 .env 文件"""
    if not env_file.exists():
        return
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mem-deep-research",
        description="Mem Deep Research - AI Agent Orchestration Framework",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # init
    p_init = subparsers.add_parser("init", help="Create a new research project")
    p_init.add_argument("name", help="Project name / directory")
    p_init.add_argument(
        "--provider",
        default="openrouter",
        choices=["anthropic", "openai", "openrouter", "deepseek"],
        help="LLM provider (default: openrouter)",
    )
    p_init.add_argument("--chinese", action="store_true", help="Enable Chinese context")
    p_init.add_argument("--force", action="store_true", help="Overwrite existing directory")

    # run
    p_run = subparsers.add_parser("run", help="Run a research task")
    p_run.add_argument("task", help="Research task description")
    p_run.add_argument("--project", "-p", default=".", help="Project directory (default: .)")

    # test
    p_test = subparsers.add_parser("test", help="Test project configuration")
    p_test.add_argument("--project", "-p", default=".", help="Project directory (default: .)")

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "test":
        cmd_test(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
