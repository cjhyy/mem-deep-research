"""
任务规划模块

在深度研究开始前，通过 LLM 将复杂任务分解为子问题，
并生成研究计划注入到消息历史中引导执行。
"""

import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger("mem_deep_research")

PLANNING_PROMPT = """You are a research planning assistant. Break down the following research task into 3-7 focused sub-questions that, when answered, will fully address the main question.

Return ONLY a JSON object in this exact format (no markdown, no extra text):
{
  "sub_questions": [
    {"id": 1, "question": "...", "priority": "high"},
    {"id": 2, "question": "...", "priority": "medium"}
  ]
}

Priority levels: "high", "medium", "low"

Research task: """


@dataclass
class SubQuestion:
    """研究子问题"""

    id: int
    question: str
    priority: str = "medium"  # high, medium, low
    status: str = "pending"  # pending → in_progress → completed
    findings: str = ""


@dataclass
class ResearchPlan:
    """研究计划"""

    main_question: str
    sub_questions: list[SubQuestion] = field(default_factory=list)

    def to_context_string(self) -> str:
        """生成可注入 prompt 的计划文本"""
        lines = [
            f"Main Research Question: {self.main_question}",
            "",
            "Sub-questions to investigate:",
        ]
        for sq in self.sub_questions:
            priority_marker = {"high": "!!!", "medium": "!!", "low": "!"}.get(sq.priority, "!!")
            lines.append(f"  {sq.id}. [{priority_marker}] {sq.question}")
        lines.append("")
        lines.append(
            "Instructions: Address each sub-question systematically. Start with high-priority questions."
        )
        return "\n".join(lines)

    def get_progress(self) -> float:
        """获取完成进度 (0.0 ~ 1.0)"""
        if not self.sub_questions:
            return 0.0
        completed = sum(1 for sq in self.sub_questions if sq.status == "completed")
        return completed / len(self.sub_questions)


class TaskPlanner:
    """任务规划器

    通过单轮 LLM 调用将复杂研究任务分解为子问题。
    """

    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    async def create_plan(
        self,
        task_description: str,
        llm_client,
        system_prompt: str = "",
    ) -> ResearchPlan | None:
        """创建研究计划

        Args:
            task_description: 任务描述
            llm_client: LLM 客户端实例
            system_prompt: 系统提示词

        Returns:
            ResearchPlan 或 None（如果规划失败）
        """
        if not self.enabled:
            return None

        try:
            planning_system = "You are a research planning assistant. Output valid JSON only."
            planning_messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": PLANNING_PROMPT + task_description}],
                }
            ]

            response_text = await llm_client.send_message(
                system_prompt=planning_system,
                message_history=planning_messages,
                tool_definitions=[],
                stream_message_callback=None,
            )

            # 提取 JSON（从响应文本中）
            if not response_text:
                logger.warning("[TaskPlanner] Empty LLM response, skipping planning")
                return None

            # 处理 LLM 返回的可能是 tuple 的情况
            if isinstance(response_text, tuple):
                response_text = response_text[0] if response_text[0] else ""

            return self._parse_plan(task_description, response_text)

        except Exception as e:
            logger.warning(f"[TaskPlanner] Planning failed (graceful fallback): {e}")
            return None

    def _parse_plan(self, task_description: str, response_text: str) -> ResearchPlan | None:
        """解析 LLM 返回的规划 JSON

        Args:
            task_description: 原始任务描述
            response_text: LLM 返回的文本

        Returns:
            ResearchPlan 或 None
        """
        try:
            # 尝试直接解析
            data = json.loads(response_text.strip())
        except json.JSONDecodeError:
            # 尝试从 markdown code block 提取
            try:
                import re

                match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", response_text, re.DOTALL)
                if match:
                    data = json.loads(match.group(1).strip())
                else:
                    # 尝试找到 { 开始的 JSON
                    start = response_text.find("{")
                    end = response_text.rfind("}")
                    if start >= 0 and end > start:
                        data = json.loads(response_text[start : end + 1])
                    else:
                        logger.warning("[TaskPlanner] Cannot extract JSON from response")
                        return None
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"[TaskPlanner] JSON parse failed: {e}")
                return None

        # 构建 ResearchPlan
        sub_questions = []
        raw_questions = data.get("sub_questions", [])

        for item in raw_questions:
            if not isinstance(item, dict):
                continue
            sq = SubQuestion(
                id=item.get("id", len(sub_questions) + 1),
                question=item.get("question", ""),
                priority=item.get("priority", "medium"),
            )
            if sq.question:
                sub_questions.append(sq)

        if not sub_questions:
            logger.warning("[TaskPlanner] No valid sub-questions found")
            return None

        plan = ResearchPlan(
            main_question=task_description,
            sub_questions=sub_questions,
        )

        logger.info(f"[TaskPlanner] Created plan with {len(sub_questions)} sub-questions")
        return plan
