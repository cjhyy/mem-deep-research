"""
Core 模块 - Pipeline 和 Orchestrator

主要组件:
- pipeline: 任务执行流水线
- orchestrator: Agent 协调器
- stream_handler: 流式输出处理
- tool_result_formatter: 工具结果格式化
- user_context: 用户上下文构建
- llm_call_handler: LLM 调用处理
- message_interceptor: 消息拦截处理
- tool_executor: 工具执行器
- sub_agent_runner: 子 Agent 运行器
- monitoring: 执行监控
"""

from mem_deep_research_core.core.agent_factory import (
    AgentConfig,
    AgentFactory,
    TaskResult,
    run_agent,
    run_agent_from_project,
)
from mem_deep_research_core.core.constants import generate_message_id
from mem_deep_research_core.core.llm_call_handler import (
    LLMCallHandler,
    SummaryHandler,
    generate_reflection_prompt,
)
from mem_deep_research_core.core.message_interceptor import MessageInterceptorHandler
from mem_deep_research_core.core.monitoring import (
    ExecutionMonitor,
    MonitoringConfig,
    MonitoringState,
    TurnCounter,
)
from mem_deep_research_core.core.orchestrator import Orchestrator
from mem_deep_research_core.core.pipeline import (
    create_pipeline_components,
    execute_task_pipeline,
)
from mem_deep_research_core.core.stream_handler import StreamHandler
from mem_deep_research_core.core.sub_agent_runner import SubAgentRunner
from mem_deep_research_core.core.tool_executor import ToolExecutor
from mem_deep_research_core.core.tool_result_formatter import ToolResultFormatter
from mem_deep_research_core.core.user_context import UserContextBuilder, detect_language_by_chars

__all__ = [
    # Pipeline
    "execute_task_pipeline",
    "create_pipeline_components",
    # Orchestrator
    "Orchestrator",
    # Agent Factory (一体化启动)
    "AgentFactory",
    "AgentConfig",
    "TaskResult",
    "run_agent",
    "run_agent_from_project",
    # Stream
    "StreamHandler",
    # Tool formatting
    "ToolResultFormatter",
    # User context
    "UserContextBuilder",
    "detect_language_by_chars",
    # LLM call handling
    "LLMCallHandler",
    "SummaryHandler",
    "generate_message_id",
    "generate_reflection_prompt",
    # Message interception
    "MessageInterceptorHandler",
    # Tool execution
    "ToolExecutor",
    # Sub-agent
    "SubAgentRunner",
    # Monitoring
    "ExecutionMonitor",
    "MonitoringConfig",
    "MonitoringState",
    "TurnCounter",
]
