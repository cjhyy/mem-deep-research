# 开发指南

## 环境搭建

```bash
# 克隆仓库
git clone https://github.com/cjhyy/mem-deep-research.git
cd mem-deep-research

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 安装开发依赖
pip install -e ".[dev]"
```

## 技术栈

| 技术 | 用途 |
|------|------|
| Python 3.12+ | 运行时 |
| asyncio | 全异步设计 |
| Pydantic | 配置验证 |
| OmegaConf (Hydra) | 运行时配置 |
| MCP | 工具协议 |
| FastMCP | MCP Server 实现 |
| tenacity | 重试机制 |
| tiktoken | Token 估算 |
| Rich | 终端输出 |

## 测试

```bash
# 运行全部测试
python -m pytest tests/ -v

# 运行单个测试文件
python -m pytest tests/test_hooks.py -v

# 带覆盖率
python -m pytest tests/ --cov=mem_deep_research_core --cov-report=term-missing
```

### 测试文件

| 文件 | 测试内容 |
|------|---------|
| `test_context_manager.py` | 上下文管理、去重、压缩 |
| `test_exceptions.py` | 异常定义 |
| `test_hooks.py` | 钩子注册、执行、链式调用 |
| `test_inline_skill_selector.py` | Inline Skill 选择 |
| `test_interceptor_config.py` | 拦截器配置 |
| `test_monitoring.py` | 执行监控、循环检测 |
| `test_secure_context.py` | SecureContext 占位符替换 |
| `test_task_planner.py` | 任务分解 |
| `test_window_strategy.py` | 窗口压缩策略 |

## 代码质量

```bash
# Lint 检查
ruff check .

# 自动格式化
ruff format .

# 类型检查
mypy mem_deep_research_core/
```

### Ruff 配置

- 目标: Python 3.12
- 行宽: 100
- 启用规则: E, W, F, I, B, C4, UP, SIM

## 项目结构

```
mem-deep-research/
├── mem_deep_research/               # 包装包（re-export）
│   └── __init__.py                  # 导出 DeepResearch, TaskResult
├── mem_deep_research_core/          # 框架核心代码
│   ├── deep_research.py             # 主入口
│   ├── config_schema.py             # Pydantic 配置
│   ├── exceptions.py                # 异常定义
│   ├── core/                        # 核心模块（16+ 文件）
│   ├── llm/                         # LLM 客户端
│   ├── prompts/                     # Prompt 系统
│   ├── tool/                        # 工具系统
│   ├── skills/                      # Skill 系统
│   ├── utils/                       # 工具函数
│   └── mem_deep_research_logging/   # 日志系统
├── config/                          # 框架默认配置
├── tests/                           # 单元测试
├── docs/                            # 文档
├── pyproject.toml                   # 包配置
├── CLAUDE.md                        # AI 辅助开发指令
├── CHANGELOG.md                     # 版本历史
├── CONTRIBUTING.md                  # 贡献指南
└── LICENSE                          # Apache 2.0
```

## 开发规范

### 新增核心功能

1. 在 `config_schema.py` 添加 Pydantic 配置模型
2. 设置合理默认值
3. 在 `core/` 下实现功能模块
4. 添加对应的测试文件
5. 更新 CLAUDE.md 和文档

### 新增 LLM Provider

1. 继承 `OpenAICompatibleClient` 或 `LLMProviderClientBase`
2. 实现必要的抽象方法
3. 在 `llm/providers/__init__.py` 中注册
4. 添加配置支持

### 新增工具

1. 创建 MCP Server 文件（`tool/mcp_servers/`）
2. 创建工具配置 YAML（`config/tool/`）
3. 在 Agent 配置中引用

### 新增 Skill

1. 创建 Markdown 文件（`config/skills/definitions/`）
2. 编写 YAML front matter（触发条件、元数据）
3. 编写 Skill 内容

### 新增 Hook

1. 在 `hooks.py` 中添加 Hook 名称（如有新类型）
2. 在框架中适当位置调用 `hooks.call()`
3. 更新文档

## 包发布

```bash
# 构建
python -m build

# 上传到 PyPI
python -m twine upload dist/*
```

构建系统使用 hatchling，wheel 包含 `mem_deep_research_core` 和 `mem_deep_research` 两个包。

## CLI 入口

```bash
# 通过 pyproject.toml 注册的入口点
mem-deep-research init my_project    # 初始化项目
mem-deep-research run "任务"         # 运行任务
```

CLI 实现在 `mem_deep_research_core/cli/main.py`。
