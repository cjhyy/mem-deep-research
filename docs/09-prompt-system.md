# Prompt 系统

## 概述

Prompt 系统通过 `AgentPrompt` 统一类和 `PromptTemplateLoader` 模板加载器实现配置驱动的 Prompt 生成，无需继承即可组合不同模板。

## AgentPrompt

核心 Prompt 生成类，通过配置组合模板：

```python
class AgentPrompt:
    def __init__(
        self,
        agent_type: str = "main",       # main | worker
        tool_format: str = "xml",        # xml | native
        presets: list[str] = [],         # 预设列表
        templates_dir: str = None,       # 自定义模板目录
        custom_system_template: str = None,
        custom_summarize_template: str = None,
    )
```

### System Prompt 生成

```python
prompt = AgentPrompt(agent_type="main", tool_format="xml", presets=["research"])

system_prompt = prompt.generate_system_prompt_with_mcp_tools(
    mcp_servers=tool_definitions,        # 工具定义列表
    chinese_context=False,               # 是否中文语境
    deep_research_cfg=deep_research_config,
    extra_context="附加上下文",
)
```

### Prompt 组成结构

```
┌─────────────────────────────────────────┐
│ 1. System Intro（系统介绍 + 日期时间）    │
│    模板: base/system_intro.md            │
├─────────────────────────────────────────┤
│ 2. Tool Format（工具格式说明）            │
│    模板: base/tool_format_xml.md         │
│          base/tool_format_native.md      │
├─────────────────────────────────────────┤
│ 3. Tool Definitions（工具定义 JSON）      │
│    动态生成                              │
├─────────────────────────────────────────┤
│ 4. Objective（任务目标说明）              │
│    模板: base/objective_main.md          │
│          base/objective_worker.md        │
├─────────────────────────────────────────┤
│ 5. Presets（预设模块）                    │
│    模板: presets/research.md             │
│          presets/time_sensitive.md       │
│          presets/research_planning.md    │
├─────────────────────────────────────────┤
│ 6. Chinese Context（中文语境，可选）      │
│    模板: base/chinese_context.md         │
├─────────────────────────────────────────┤
│ 7. SecureContext 说明（如有敏感数据）     │
│    动态生成                              │
├─────────────────────────────────────────┤
│ 8. Skill 内容（如已选择 Skill）          │
│    动态注入                              │
└─────────────────────────────────────────┘
```

### 摘要 Prompt 生成

```python
summarize_prompt = prompt.generate_summarize_prompt(
    task_description="研究任务描述",
    task_failed=False,
    chinese_context=False,
    target_language="zh",
)
```

### 子 Agent 工具定义

```python
tool_def = prompt.expose_agent_as_tool("worker")
# 返回工具定义 dict，主 Agent 可通过调用 "agent-worker" 工具创建子 Agent
```

## PromptTemplateLoader

模板加载和渲染引擎：

```python
class PromptTemplateLoader:
    def __init__(self, custom_dir: str = None)

    def load_template(self, name: str) -> str
    def render_template(self, template: str, **variables) -> str
    def load_and_render(self, name: str, **variables) -> str
    def template_exists(self, name: str) -> bool
    def list_templates(self) -> list[str]
    def add_search_path(self, path: str, priority: bool = False)
```

### 模板搜索顺序

```
1. 自定义目录 (custom_dir)
2. 项目目录 (project_dir/config/prompts/)
3. 框架内置 (mem_deep_research_core/prompts/templates/)
```

### 模板语法

使用 `{{variable}}` 占位符：

```markdown
# System Introduction

You are an AI research assistant. Today is {{date}}.

## Your Objective

{{objective_description}}

## Available Tools

{{tool_definitions}}
```

渲染：

```python
loader = PromptTemplateLoader()
result = loader.load_and_render("base/system_intro", date="2026-03-11")
```

## 内置模板目录

```
prompts/templates/
├── base/                              # 基础模板
│   ├── system_intro.md                # 系统介绍
│   ├── objective_main.md              # 主 Agent 目标
│   ├── objective_worker.md            # 子 Agent 目标
│   ├── tool_format_xml.md             # XML 工具格式
│   ├── tool_format_native.md          # Native 工具格式
│   ├── sub_agent_tool_description.md  # 子 Agent 工具描述
│   ├── summarize.md                   # 摘要模板
│   ├── chinese_context.md             # 中文语境
│   └── chinese_worker.md              # 中文子 Agent
├── presets/                           # 预设组合
│   ├── research.md                    # 研究模式
│   ├── research_planning.md           # 研究规划
│   └── time_sensitive.md              # 时间敏感
├── extraction/                        # 答案提取
│   ├── gaia_answer_type.md            # 答案类型判断
│   ├── gaia_extract_number.md         # 数字提取
│   ├── gaia_extract_string.md         # 字符串提取
│   ├── gaia_extract_time.md           # 时间提取
│   ├── gaia_confidence.md             # 置信度评估
│   ├── gaia_chinese_supplement.md     # 中文补充
│   └── browsecomp_zh.md              # 浏览理解（中文）
├── guidance/                          # 任务指导
│   └── task_guidance_chinese.md       # 中文任务指导
├── hints/                             # 提示模板
│   ├── hint_prefix.md                 # 提示前缀
│   ├── hint_instruction.md            # 提示指令
│   └── hint_chinese_supplement.md     # 中文提示补充
├── reflection/                        # 反思模板
│   ├── reflection.md                  # 英文反思
│   └── reflection_chinese.md          # 中文反思
├── skills/                            # Skill 选择
│   └── select_skills.md              # Skill 选择提示
└── language/                          # 语言检测
    └── detect_language.md             # 语言检测提示
```

## 配置参考

```yaml
prompt:
  agent_type: main                    # main | worker
  tool_format: xml                    # xml | native
  presets: [research, time_sensitive]  # 预设模板列表
  templates_dir: null                 # 自定义模板目录
  custom_system_template: null        # 自定义 system prompt 模板
  custom_summarize_template: null     # 自定义摘要模板
```

## 自定义模板

1. 在项目目录创建 `config/prompts/` 目录
2. 放入自定义模板文件（.md）
3. 在配置中引用：

```yaml
prompt:
  templates_dir: "config/prompts"
  custom_system_template: "my_system_prompt"  # 对应 my_system_prompt.md
```

或直接创建与内置模板同名的文件来覆盖：

```
my_project/config/prompts/
└── base/
    └── objective_main.md    # 覆盖框架内置的 objective_main
```
