# Skill 系统

## 概述

Skill 系统允许在运行时动态向 Agent 注入领域知识和策略指南，无需修改框架代码。

## 三种选择方式

```yaml
skill_selection:
  enabled: true
  method: inline       # rules | llm | inline
  max_skills: 3
```

| method | 说明 | 开销 | 适用场景 |
|--------|------|------|---------|
| `rules` | 基于关键词/工具/上下文评分匹配 | 零 | 规则明确的场景 |
| `llm` | 额外 LLM 调用选择最佳 Skill | 一次轻量 LLM | 复杂匹配场景 |
| `inline` | LLM 在响应中声明 `<next_skills>` 标签 | 零 | 推荐，最灵活 |

## Skill 定义格式

Skill 以 Markdown 文件定义，存放在 `config/skills/definitions/` 目录：

```markdown
---
name: search_strategy
type: knowledge
description: "搜索策略指南：如何有效使用搜索工具"
when_to_use: "当需要搜索信息时使用"
triggers:
  keywords: ["搜索", "查找", "search", "find"]
  tools_mentioned: ["web_search", "semantic_search"]
  context_conditions:
    task_type: research
metadata:
  priority: 10
  examples:
    - "搜索 AI Agent 最新论文"
    - "查找 MCP 协议文档"
---

# 搜索策略指南

## 搜索原则

1. 先用宽泛查询了解全局
2. 根据初步结果缩小范围
3. 交叉验证多个来源

## 搜索技巧

- 使用具体的关键词组合
- 尝试不同语言搜索
- 注意时效性
```

### YAML Front Matter 字段

| 字段 | 说明 |
|------|------|
| `name` | Skill 唯一标识 |
| `type` | 类型（knowledge/strategy/workflow） |
| `description` | 简短描述 |
| `when_to_use` | 使用场景说明 |
| `triggers.keywords` | 触发关键词 |
| `triggers.tools_mentioned` | 关联工具名 |
| `triggers.context_conditions` | 上下文条件 |
| `metadata.priority` | 优先级（越高越优先） |
| `metadata.examples` | 示例查询 |

## Rules 匹配模式

`SkillMatcher` 基于多维度评分：

| 维度 | 分值 | 说明 |
|------|------|------|
| 关键词命中 | +1 | query 中包含 triggers.keywords |
| 意图匹配 | +2 | description 语义匹配 |
| 工具关联 | +1.5 | 当前可用工具在 tools_mentioned 中 |
| 上下文条件 | +1 | context_conditions 满足 |

```python
class SkillMatcher:
    def match(
        self, query, context=None, tools_to_use=None,
        max_skills=3, min_score=1.0,
    ) -> list[MatchedSkill]:
        """返回按分数排序的匹配结果"""
```

## LLM 选择模式

向 LLM 发送 Skill 摘要列表，由 LLM 选择最相关的 Skill：

```python
class LLMSkillSelector:
    async def select(self, query, tool_definitions, max_skills=3) -> list[str]:
        """返回选中的 Skill 名称列表"""
```

使用 `prompts/templates/skills/select_skills.md` 模板构建选择提示。

## Inline 选择模式（推荐）

LLM 在响应末尾声明下一轮需要的 Skill：

```xml
<!-- LLM 响应示例 -->
根据搜索结果，我需要进一步分析...

<next_skills>search_strategy, data_analysis</next_skills>
```

### InlineSkillSelector

```python
class InlineSkillSelector:
    def build_skill_catalog_prompt(self) -> str
        """生成 Skill 目录，注入 System Prompt"""

    def parse_next_skills(self, text) -> list[str]
        """从响应中提取 <next_skills> 标签"""

    def update_pending_skills(self, response_text) -> None
        """解析并验证 Skill"""

    def inject_pending_skills(self, base_prompt) -> str
        """将待注入 Skill 内容合并到 System Prompt"""

    def consume_pending_skills(self) -> list[str]
        """获取并清空待注入列表"""
```

### 工作流程

```
Turn N:
  1. System Prompt 中包含 Skill 目录
  2. LLM 回复末尾声明 <next_skills>search_strategy</next_skills>
  3. 框架解析并验证 Skill 名称

Turn N+1:
  1. 将 search_strategy 内容注入 System Prompt
  2. LLM 使用新注入的知识继续工作
  3. 可声明新的 <next_skills> 或不声明
```

## SkillInjector

将匹配的 Skill 注入 System Prompt：

```python
class SkillInjector:
    def inject_skills(
        self, base_prompt, query, context=None,
        tools_to_use=None, max_skills=3,
    ) -> str:
        """匹配并注入 Skill，返回增强后的 Prompt"""

    def inject_selected_skills(
        self, base_prompt, skill_names, include_examples=False,
    ) -> str:
        """按名称注入指定 Skill"""
```

注入位置：在工具定义部分之前插入 Skill 内容。

## 配置加载

Skill 从两个位置加载，项目级优先：

```
1. project_dir/config/skills/definitions/  (项目级)
2. 框架 config/skills/definitions/          (框架默认)
```

两个目录的 Skill 会合并，同名 Skill 项目级覆盖框架级。

## 自定义 Skill

在项目目录创建 Skill 定义：

```
my_project/config/skills/definitions/
├── my_analysis.md
├── my_writing.md
└── my_domain_knowledge.md
```

框架会自动发现和加载这些 Skill。

## 配置参考

```yaml
skill_selection:
  enabled: true              # 启用 Skill 系统
  method: inline             # rules | llm | inline
  max_skills: 3              # 每轮最大 Skill 数
  model: null                # LLM 选择模式使用的模型（默认同主模型）
```
