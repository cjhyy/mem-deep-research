"""任务规划器单元测试

覆盖: SubQuestion, TaskPlan 数据结构, _parse_plan, TaskPlanner.enabled 守卫
"""

import pytest

from mem_deep_research_core.core.task_planner import SubQuestion, TaskPlan, TaskPlanner

# ========== SubQuestion 测试 ==========


class TestSubQuestion:
    def test_defaults(self):
        sq = SubQuestion(id=1, question="What is X?")
        assert sq.priority == "medium"
        assert sq.status == "pending"
        assert sq.findings == ""

    def test_custom_fields(self):
        sq = SubQuestion(
            id=2, question="Why?", priority="high", status="completed", findings="Found it"
        )
        assert sq.priority == "high"
        assert sq.status == "completed"


# ========== TaskPlan 测试 ==========


class TestTaskPlan:
    def test_to_context_string(self):
        """to_context_string 格式正确"""
        plan = TaskPlan(
            main_question="What is AI?",
            sub_questions=[
                SubQuestion(id=1, question="History of AI?", priority="high"),
                SubQuestion(id=2, question="Current state?", priority="medium"),
                SubQuestion(id=3, question="Future trends?", priority="low"),
            ],
        )
        text = plan.to_context_string()
        assert "Main Research Question: What is AI?" in text
        assert "[!!!] History of AI?" in text
        assert "[!!] Current state?" in text
        assert "[!] Future trends?" in text
        assert "high-priority" in text

    def test_progress_empty(self):
        """空计划进度为 0"""
        plan = TaskPlan(main_question="test")
        assert plan.get_progress() == 0.0

    def test_progress_partial(self):
        """部分完成进度计算"""
        plan = TaskPlan(
            main_question="test",
            sub_questions=[
                SubQuestion(id=1, question="Q1", status="completed"),
                SubQuestion(id=2, question="Q2", status="pending"),
                SubQuestion(id=3, question="Q3", status="in_progress"),
                SubQuestion(id=4, question="Q4", status="completed"),
            ],
        )
        assert plan.get_progress() == pytest.approx(0.5)

    def test_progress_all_completed(self):
        """全部完成进度为 1.0"""
        plan = TaskPlan(
            main_question="test",
            sub_questions=[
                SubQuestion(id=1, question="Q1", status="completed"),
                SubQuestion(id=2, question="Q2", status="completed"),
            ],
        )
        assert plan.get_progress() == pytest.approx(1.0)


# ========== TaskPlanner._parse_plan 测试 ==========


class TestParsePlan:
    @pytest.fixture
    def planner(self):
        return TaskPlanner(enabled=True)

    def test_parse_valid_json(self, planner):
        """解析标准 JSON"""
        response = '{"sub_questions": [{"id": 1, "question": "Q1", "priority": "high"}]}'
        plan = planner._parse_plan("main task", response)
        assert plan is not None
        assert len(plan.sub_questions) == 1
        assert plan.sub_questions[0].question == "Q1"
        assert plan.sub_questions[0].priority == "high"

    def test_parse_markdown_code_block(self, planner):
        """解析 markdown code block 中的 JSON"""
        response = (
            'Here is the plan:\n```json\n{"sub_questions": [{"id": 1, "question": "Q1"}]}\n```'
        )
        plan = planner._parse_plan("task", response)
        assert plan is not None
        assert len(plan.sub_questions) == 1

    def test_parse_json_in_text(self, planner):
        """从文本中提取 JSON"""
        response = 'The plan is: {"sub_questions": [{"id": 1, "question": "Q1"}]} end.'
        plan = planner._parse_plan("task", response)
        assert plan is not None
        assert len(plan.sub_questions) == 1

    def test_parse_invalid_json_returns_none(self, planner):
        """无效 JSON 返回 None (graceful fallback)"""
        plan = planner._parse_plan("task", "this is not json at all")
        assert plan is None

    def test_parse_empty_questions_returns_none(self, planner):
        """空 sub_questions 返回 None"""
        response = '{"sub_questions": []}'
        plan = planner._parse_plan("task", response)
        assert plan is None

    def test_parse_missing_question_field(self, planner):
        """缺少 question 字段的条目被跳过"""
        response = '{"sub_questions": [{"id": 1}, {"id": 2, "question": "Valid Q"}]}'
        plan = planner._parse_plan("task", response)
        assert plan is not None
        assert len(plan.sub_questions) == 1
        assert plan.sub_questions[0].question == "Valid Q"

    def test_parse_defaults_priority(self, planner):
        """缺少 priority 字段默认为 medium"""
        response = '{"sub_questions": [{"id": 1, "question": "Q1"}]}'
        plan = planner._parse_plan("task", response)
        assert plan.sub_questions[0].priority == "medium"


# ========== TaskPlanner enabled guard 测试 ==========


class TestTaskPlannerEnabled:
    @pytest.mark.asyncio
    async def test_disabled_returns_none(self):
        """disabled 时 create_plan 直接返回 None"""
        planner = TaskPlanner(enabled=False)
        result = await planner.create_plan("task", llm_client=None)
        assert result is None

    def test_enabled_flag(self):
        planner = TaskPlanner(enabled=True)
        assert planner.enabled is True

        planner2 = TaskPlanner(enabled=False)
        assert planner2.enabled is False
