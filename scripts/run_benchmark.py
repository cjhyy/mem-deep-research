#!/usr/bin/env python3
"""
Dual-Mode Benchmark Runner

验证 auto 路由是否正确 + 各模式指标差异是否显著。

用法:
    # 跑全部任务 × 全部模式（auto/quick/standard/deep）
    python scripts/run_benchmark.py

    # 只跑 auto 模式（验证路由准确率）
    python scripts/run_benchmark.py --modes auto

    # 只跑特定难度
    python scripts/run_benchmark.py --difficulty quick standard

    # 指定配置
    python scripts/run_benchmark.py --project ./example_project --config agent

    # dry-run（只打印任务列表，不执行）
    python scripts/run_benchmark.py --dry-run

输出:
    1. 每个任务的详细结果（模式、耗时、token、工具调用数）
    2. 汇总对比表
    3. 路由准确率（auto 模式）
    4. JSON 结果文件（logs/benchmark_results.json）
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("benchmark")
logger.setLevel(logging.INFO)


# ============================================================
# 任务集定义
# ============================================================


@dataclass
class BenchmarkTask:
    """单个评测任务"""

    id: str
    query: str
    expected_mode: str  # quick / standard / deep
    difficulty: str  # 同 expected_mode，用于过滤
    description: str = ""


BENCHMARK_TASKS: list[BenchmarkTask] = [
    # --- Quick 任务（期望 auto → quick）---
    BenchmarkTask(
        id="q1",
        query="2 + 2 等于多少？",
        expected_mode="quick",
        difficulty="quick",
        description="简单算术",
    ),
    BenchmarkTask(
        id="q2",
        query="把 hello world 翻译成日语",
        expected_mode="quick",
        difficulty="quick",
        description="简单翻译",
    ),
    BenchmarkTask(
        id="q3",
        query="Python 的 list.sort() 是稳定排序吗？",
        expected_mode="quick",
        difficulty="quick",
        description="事实查找",
    ),
    # --- Standard 任务（期望 auto → standard）---
    BenchmarkTask(
        id="s1",
        query="搜索 OpenAI 2025-2026 年发布的主要模型，列出名称和发布日期",
        expected_mode="standard",
        difficulty="standard",
        description="多步搜索",
    ),
    BenchmarkTask(
        id="s2",
        query="比较 React 和 Vue 在 SSR 方面的支持差异",
        expected_mode="standard",
        difficulty="standard",
        description="对比分析（中等）",
    ),
    BenchmarkTask(
        id="s3",
        query="查找 Tesla 2025 年最新一个季度的财报营收数据",
        expected_mode="standard",
        difficulty="standard",
        description="数据查找",
    ),
    BenchmarkTask(
        id="s4",
        query="Python asyncio 和 threading 在 IO 密集型任务中的性能差异是什么？给出具体建议",
        expected_mode="standard",
        difficulty="standard",
        description="技术分析",
    ),
    # --- Deep 任务（期望 auto → deep）---
    BenchmarkTask(
        id="d1",
        query="全面调研 2025-2026 年 AI Agent 框架的技术演进，对比至少 5 个主流框架的架构设计、工具协议、执行模式、社区生态，撰写详细报告",
        expected_mode="deep",
        difficulty="deep",
        description="全面调研报告",
    ),
    BenchmarkTask(
        id="d2",
        query="撰写一份关于大模型推理优化技术的综述报告，覆盖 KV Cache、Speculative Decoding、量化、蒸馏等方向，要求引用具体论文和来源",
        expected_mode="deep",
        difficulty="deep",
        description="技术综述（引用）",
    ),
    BenchmarkTask(
        id="d3",
        query="深度分析 MCP (Model Context Protocol) 的设计理念、协议规范、主要实现、生态现状和未来发展方向，对比与 Function Calling 的优劣",
        expected_mode="deep",
        difficulty="deep",
        description="深度分析报告",
    ),
]

MODES = ["auto", "quick", "standard", "deep"]


# ============================================================
# 结果结构
# ============================================================


@dataclass
class TaskRunResult:
    """单次运行结果"""

    task_id: str
    mode: str  # 配置的 execution_mode
    effective_mode: str = ""  # 路由后实际执行的模式
    duration_seconds: float = 0.0
    turns: int = 0
    tool_calls: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    answer_length: int = 0
    evidence_count: int = 0
    status: str = ""
    error: str | None = None


@dataclass
class BenchmarkResults:
    """完整评测结果"""

    timestamp: str = ""
    tasks: list[dict] = field(default_factory=list)
    runs: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


# ============================================================
# Runner
# ============================================================


async def run_single_task(
    task: BenchmarkTask,
    mode: str,
    project_dir: str | None = None,
    config_name: str = "agent",
) -> TaskRunResult:
    """运行单个任务"""
    from mem_deep_research_core.deep_research import DeepResearch

    result = TaskRunResult(task_id=task.id, mode=mode)

    try:
        if project_dir:
            dr = DeepResearch.from_project(project_dir, config_name=config_name)
            # 覆盖 execution_mode
            dr._cfg.main_agent.execution_mode = mode
        else:
            dr = DeepResearch(
                execution_mode=mode,
                max_turns={"quick": 3, "standard": 8, "deep": 15, "auto": 15}.get(mode, 10),
            )

        task_result = await dr.run(task.query)

        result.duration_seconds = task_result.duration_seconds
        result.turns = task_result.turns
        result.tool_calls = task_result.tool_calls
        result.answer_length = len(task_result.answer) if task_result.answer else 0
        result.status = task_result.status
        result.error = task_result.error

        # 从 perf_metrics 提取
        perf = task_result.perf_metrics or {}
        result.effective_mode = perf.get("effective_mode", {}).get("value", mode)
        result.total_tokens = int(perf.get("total_tokens", {}).get("value", 0))
        result.prompt_tokens = int(perf.get("total_prompt_tokens", {}).get("value", 0))
        result.completion_tokens = int(perf.get("total_completion_tokens", {}).get("value", 0))

        await dr.close()

    except Exception as e:
        result.status = "error"
        result.error = str(e)
        logger.error(f"  [ERROR] {task.id}/{mode}: {e}")

    return result


async def run_benchmark(
    tasks: list[BenchmarkTask],
    modes: list[str],
    project_dir: str | None = None,
    config_name: str = "agent",
) -> list[TaskRunResult]:
    """顺序执行所有任务 × 模式"""
    all_results = []
    total = len(tasks) * len(modes)
    current = 0

    for task in tasks:
        for mode in modes:
            current += 1
            logger.info(
                f"[{current}/{total}] {task.id} ({task.difficulty}) × {mode} — {task.description}"
            )
            start = time.time()
            result = await run_single_task(task, mode, project_dir, config_name)
            elapsed = time.time() - start
            logger.info(
                f"  → {result.status} | effective={result.effective_mode} | "
                f"{elapsed:.1f}s | {result.turns} turns | {result.tool_calls} tools | "
                f"{result.total_tokens} tokens | {result.answer_length} chars"
            )
            all_results.append(result)

    return all_results


# ============================================================
# 分析与输出
# ============================================================


def analyze_results(
    tasks: list[BenchmarkTask], results: list[TaskRunResult]
) -> dict:
    """分析评测结果"""
    analysis = {
        "routing_accuracy": {},
        "mode_comparison": {},
        "per_task": {},
    }

    # 1. 路由准确率（仅 auto 模式）
    auto_runs = [r for r in results if r.mode == "auto"]
    if auto_runs:
        task_map = {t.id: t for t in tasks}
        correct = 0
        total = 0
        details = []
        for run in auto_runs:
            t = task_map.get(run.task_id)
            if not t:
                continue
            total += 1
            is_correct = run.effective_mode == t.expected_mode
            if is_correct:
                correct += 1
            details.append({
                "task_id": run.task_id,
                "expected": t.expected_mode,
                "actual": run.effective_mode,
                "correct": is_correct,
            })
        analysis["routing_accuracy"] = {
            "correct": correct,
            "total": total,
            "accuracy": f"{correct / total * 100:.0f}%" if total > 0 else "N/A",
            "details": details,
        }

    # 2. 模式维度汇总
    for mode in MODES:
        mode_runs = [r for r in results if r.mode == mode and r.status != "error"]
        if not mode_runs:
            continue
        analysis["mode_comparison"][mode] = {
            "count": len(mode_runs),
            "avg_duration": round(sum(r.duration_seconds for r in mode_runs) / len(mode_runs), 1),
            "avg_turns": round(sum(r.turns for r in mode_runs) / len(mode_runs), 1),
            "avg_tool_calls": round(sum(r.tool_calls for r in mode_runs) / len(mode_runs), 1),
            "avg_tokens": round(sum(r.total_tokens for r in mode_runs) / len(mode_runs)),
            "avg_answer_length": round(sum(r.answer_length for r in mode_runs) / len(mode_runs)),
        }

    # 3. 每个任务的跨模式对比
    for task in tasks:
        task_runs = {r.mode: r for r in results if r.task_id == task.id}
        analysis["per_task"][task.id] = {
            "description": task.description,
            "expected_mode": task.expected_mode,
            "runs": {
                mode: {
                    "effective_mode": r.effective_mode,
                    "duration": round(r.duration_seconds, 1),
                    "turns": r.turns,
                    "tool_calls": r.tool_calls,
                    "tokens": r.total_tokens,
                    "answer_length": r.answer_length,
                    "status": r.status,
                }
                for mode, r in task_runs.items()
            },
        }

    return analysis


def print_report(tasks: list[BenchmarkTask], results: list[TaskRunResult], analysis: dict):
    """打印格式化报告"""
    print("\n" + "=" * 80)
    print("  DUAL-MODE BENCHMARK REPORT")
    print("=" * 80)

    # 路由准确率
    ra = analysis.get("routing_accuracy", {})
    if ra:
        print(f"\n## Auto Routing Accuracy: {ra['accuracy']} ({ra['correct']}/{ra['total']})")
        print(f"{'Task':<8} {'Expected':<12} {'Actual':<12} {'Result'}")
        print("-" * 50)
        for d in ra.get("details", []):
            mark = "✓" if d["correct"] else "✗"
            print(f"{d['task_id']:<8} {d['expected']:<12} {d['actual']:<12} {mark}")

    # 模式对比
    mc = analysis.get("mode_comparison", {})
    if mc:
        print(f"\n## Mode Comparison (averages)")
        print(f"{'Mode':<12} {'Duration(s)':<14} {'Turns':<8} {'Tools':<8} {'Tokens':<10} {'Answer(chars)'}")
        print("-" * 70)
        for mode in MODES:
            if mode not in mc:
                continue
            m = mc[mode]
            print(
                f"{mode:<12} {m['avg_duration']:<14} {m['avg_turns']:<8} "
                f"{m['avg_tool_calls']:<8} {m['avg_tokens']:<10} {m['avg_answer_length']}"
            )

    # 每任务详情
    print(f"\n## Per-Task Details")
    for task in tasks:
        pt = analysis.get("per_task", {}).get(task.id, {})
        runs = pt.get("runs", {})
        if not runs:
            continue
        print(f"\n  {task.id} [{task.difficulty}] {task.description}")
        print(f"  {'Mode':<12} {'Effective':<12} {'Time(s)':<10} {'Turns':<8} {'Tools':<8} {'Tokens':<10} {'Status'}")
        print(f"  {'-' * 72}")
        for mode in MODES:
            if mode not in runs:
                continue
            r = runs[mode]
            print(
                f"  {mode:<12} {r['effective_mode']:<12} {r['duration']:<10} "
                f"{r['turns']:<8} {r['tool_calls']:<8} {r['tokens']:<10} {r['status']}"
            )

    print("\n" + "=" * 80)


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="Dual-Mode Benchmark Runner")
    parser.add_argument("--modes", nargs="+", default=MODES, choices=MODES, help="要测试的模式")
    parser.add_argument(
        "--difficulty", nargs="+", default=["quick", "standard", "deep"],
        choices=["quick", "standard", "deep"], help="要测试的难度"
    )
    parser.add_argument("--project", type=str, default=None, help="项目目录路径")
    parser.add_argument("--config", type=str, default="agent", help="配置文件名（不含 .yaml）")
    parser.add_argument("--output", type=str, default="logs/benchmark_results.json", help="结果输出路径")
    parser.add_argument("--dry-run", action="store_true", help="只打印任务列表")
    args = parser.parse_args()

    # 过滤任务
    tasks = [t for t in BENCHMARK_TASKS if t.difficulty in args.difficulty]

    if args.dry_run:
        print(f"Tasks ({len(tasks)}) × Modes ({len(args.modes)}) = {len(tasks) * len(args.modes)} runs\n")
        for t in tasks:
            print(f"  {t.id} [{t.difficulty}] {t.query[:60]}...")
        print(f"\nModes: {', '.join(args.modes)}")
        return

    logger.info(f"Running {len(tasks)} tasks × {len(args.modes)} modes = {len(tasks) * len(args.modes)} runs")
    logger.info(f"Modes: {', '.join(args.modes)}")
    logger.info(f"Difficulties: {', '.join(args.difficulty)}")
    if args.project:
        logger.info(f"Project: {args.project}")
    print()

    # 执行
    results = asyncio.run(run_benchmark(tasks, args.modes, args.project, args.config))

    # 分析
    analysis = analyze_results(tasks, results)

    # 输出报告
    print_report(tasks, results, analysis)

    # 保存 JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_data = BenchmarkResults(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        tasks=[asdict(t) for t in tasks],
        runs=[asdict(r) for r in results],
        summary=analysis,
    )
    output_path.write_text(json.dumps(asdict(benchmark_data), indent=2, ensure_ascii=False))
    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
