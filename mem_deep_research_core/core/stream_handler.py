"""
流式输出处理模块

处理 SSE 协议格式的流式更新，包括 workflow、agent、LLM、tool call 等事件。
"""

import logging
import uuid
from typing import Any

logger = logging.getLogger("mem_deep_research")

# Agent 名称类型：主 Agent 为 "main"，子 Agent 名称动态定义
AgentName = str


class StreamHandler:
    """处理流式输出的类"""

    def __init__(self, stream_queue: Any | None = None):
        """
        初始化流式处理器

        Args:
            stream_queue: 异步队列，用于发送流式消息
        """
        self.stream_queue = stream_queue
        self.current_agent_id: str | None = None

    async def _stream_update(self, event_type: str, data: dict):
        """Send streaming update in new SSE protocol format"""
        if self.stream_queue:
            try:
                stream_message = {
                    "event": event_type,
                    "data": data,
                }
                await self.stream_queue.put(stream_message)
            except Exception as e:
                logger.warning(f"Failed to send stream update: {e}")

    async def stream_start_workflow(self, user_input: str, workflow_id: str = None) -> str:
        """Send start_of_workflow event"""
        if not workflow_id:
            workflow_id = str(uuid.uuid4())
        await self._stream_update(
            "start_of_workflow",
            {
                "workflow_id": workflow_id,
                "input": [
                    {
                        "role": "user",
                        "content": user_input,
                    }
                ],
            },
        )
        return workflow_id

    async def stream_end_workflow(self, workflow_id: str):
        """Send end_of_workflow event"""
        await self._stream_update(
            "end_of_workflow",
            {
                "workflow_id": workflow_id,
            },
        )
        if self.stream_queue:
            try:
                await self.stream_queue.put(None)
            except Exception as e:
                logger.warning(f"Failed to send end_of_workflow: {e}")

    async def stream_show_error(self, error: str):
        """Send show_error event"""
        await self.stream_tool_call("show_error", {"error": error})
        if self.stream_queue:
            try:
                await self.stream_queue.put(None)
            except Exception as e:
                logger.warning(f"Failed to send show_error: {e}")

    async def stream_start_agent(self, agent_name: AgentName, display_name: str = None) -> str:
        """Send start_of_agent event"""
        agent_id = str(uuid.uuid4())
        self.current_agent_id = agent_id
        await self._stream_update(
            "start_of_agent",
            {
                "agent_name": agent_name,
                "display_name": display_name,
                "agent_id": agent_id,
            },
        )
        return agent_id

    async def stream_end_agent(self, agent_name: AgentName, agent_id: str):
        """Send end_of_agent event"""
        await self._stream_update(
            "end_of_agent",
            {
                "agent_name": agent_name,
                "agent_id": agent_id,
            },
        )

    async def stream_start_llm(self, agent_name: AgentName, display_name: str = None):
        """Send start_of_llm event"""
        await self._stream_update(
            "start_of_llm",
            {
                "agent_name": agent_name,
                "display_name": display_name,
            },
        )

    async def stream_end_llm(self, agent_name: AgentName):
        """Send end_of_llm event"""
        await self._stream_update(
            "end_of_llm",
            {
                "agent_name": agent_name,
            },
        )

    async def stream_message(self, message_id: str, delta_content: str):
        """Send message event"""
        await self._stream_update(
            "message",
            {
                "message_id": message_id,
                "delta": {
                    "content": delta_content,
                },
            },
        )

    async def stream_reasoning(
        self, reasoning_id: str, content: str, parent_uid: str = None, status: str = "PROCESSING"
    ):
        """
        Send reasoning event for deep research structured output.

        Args:
            reasoning_id: Unique identifier for this reasoning block
            content: The reasoning content to display
            parent_uid: Parent node UID (typically the current thinking node)
            status: Node status - "PROCESSING" or "SUCCESS"
        """
        await self._stream_update(
            "tool_call",
            {
                "tool_call_id": reasoning_id,
                "tool_name": "show_reasoning",
                "delta_input": {
                    "text": content,
                    "parent_uid": parent_uid or self.current_agent_id,
                    "status": status,
                },
            },
        )

    async def stream_tool_call(
        self, tool_name: str, payload: dict, streaming: bool = False, tool_call_id: str = None
    ) -> str:
        """Send tool_call event"""
        if not tool_call_id:
            tool_call_id = str(uuid.uuid4())

        if streaming:
            for key, value in payload.items():
                await self._stream_update(
                    "tool_call",
                    {
                        "tool_call_id": tool_call_id,
                        "tool_name": tool_name,
                        "delta_input": {key: value},
                    },
                )
        else:
            # Send complete tool call
            await self._stream_update(
                "tool_call",
                {
                    "tool_call_id": tool_call_id,
                    "tool_name": tool_name,
                    "tool_input": payload,
                },
            )

        return tool_call_id

    async def stream_usage_info(
        self, agent_name: AgentName, usage_data: dict[str, Any], scene: str
    ):
        """
        Send usage_info event

        :param agent_name: Name of the agent
        :param usage_data: Usage data dictionary
        :param scene: Scene identifier - "tool_call", "main_agent_end", or "sub_agent_end"
        """
        await self._stream_update(
            "usage_info",
            {
                "agent_name": agent_name,
                "scene": scene,
                "usage": usage_data,
            },
        )
