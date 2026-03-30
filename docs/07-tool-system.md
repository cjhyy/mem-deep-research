# 工具系统

工具系统完全基于 MCP (Model Context Protocol) 规范，支持多种传输方式。

## 传输模式

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| **stdio** | 本地子进程（npx、python） | 本地工具、CLI 工具 |
| **streamable-http** | HTTP 远程服务（推荐） | 远程 API、微服务 |
| **sse** | Server-Sent Events | 实时流式服务 |

## 工具配置

### 本地工具 (stdio)

```yaml
# config/tool/tool-searching-serper.yaml
name: "tool-searching-serper"
tool_command: "npx"
args:
  - "-y"
  - "@anthropic/tool-searching"
env:
  SERPER_API_KEY: "${oc.env:SERPER_API_KEY,}"
```

### 远程工具 (streamable-http)

```yaml
# config/tool/tool-remote.yaml
name: "tool-remote"
url: "https://api.example.com/mcp"
transport: "streamable-http"
headers:
  Authorization: "Bearer ${oc.env:API_TOKEN}"
```

### SSE 工具

```yaml
# config/tool/tool-sse.yaml
name: "tool-sse"
url: "https://api.example.com/sse"
transport: "sse"
```

### Python MCP Server

```yaml
# config/tool/tool-calculator.yaml
name: "tool-calculator"
tool_command: "python"
args:
  - "mem_deep_research_core/tool/mcp_servers/calculator_server.py"
```

## 在 Agent 配置中引用

```yaml
main_agent:
  tool_config:
    - tool-searching-serper    # YAML 文件名（不含 .yaml）
    - tool-calculator
  max_tool_calls_per_turn: 10  # 每轮最大调用数
```

## ToolManager

核心工具管理类：

```python
class ToolManager:
    # 工具发现
    async def get_all_tool_definitions(self, parallel=True) -> list[dict]

    # 工具执行
    async def execute_tool_call(
        self, server_name, tool_name, arguments, context=None,
    ) -> dict:
        """返回 {"result": ...} 或 {"error": ...}"""

    # 上下文管理
    def set_context(self, context: dict) -> None

    # 缓存管理
    def clear_cache(self, server_name=None) -> None
```

### 工具发现流程

```
get_all_tool_definitions()
  ├─ 遍历所有配置的 server
  ├─ 建立 MCP 连接（stdio/http/sse）
  ├─ 调用 list_tools() 获取工具定义
  ├─ 缓存定义（避免重复连接）
  └─ 返回合并后的工具列表
```

### 工具执行流程

```
execute_tool_call(server_name, tool_name, arguments)
  ├─ 查找工具所在的 server
  ├─ 如果 server 名称不匹配，尝试自动纠正
  ├─ 建立 MCP 连接
  ├─ 调用 call_tool(tool_name, arguments)
  ├─ 超时控制（默认 900 秒）
  └─ 返回结果或错误
```

### 上下文注入

ToolManager 支持向 MCP Server 注入上下文：

```python
tool_manager.set_context({
    "user_id": "uid-123",
    "org_id": "org-456",
    "timezone": "Asia/Shanghai",
})
```

注入的上下文通过 `_mcp_context` 参数传递给 MCP Server。

## 内置 MCP Server

### calculator_server.py

基础计算器工具，支持数学表达式求值。

### searching_mcp_server.py

搜索工具 MCP Server，封装 Serper API。

### browser_session.py

浏览器自动化工具，支持网页抓取。

### code_executor_server.py

沙箱化的代码执行工具。支持 Python 代码和 shell 命令执行。

```yaml
# config/tool/tool-code-executor.yaml
name: "tool-code-executor"
transport: "inprocess"
module: "mem_deep_research_core.tool.mcp_servers.code_executor_server"
object: "mcp"
```

提供两个工具：
- `execute_python(code, timeout=30)` — 执行 Python 代码，返回 stdout/stderr
- `execute_command(command, timeout=30)` — 执行 shell 命令（受白名单限制）

安全限制：
- Python 代码写入临时文件后执行，非 `exec()` 方式
- Shell 命令白名单：`ls`, `cat`, `head`, `tail`, `wc`, `grep`, `find`, `echo`, `pwd`, `date`, `python`, `pip`
- 执行超时保护（默认 30 秒）
- 工作目录隔离

### filesystem_server.py

文件系统读写工具，支持路径白名单访问控制。

```yaml
# config/tool/tool-filesystem.yaml
name: "tool-filesystem"
transport: "inprocess"
module: "mem_deep_research_core.tool.mcp_servers.filesystem_server"
object: "mcp"
env:
  MCP_ALLOWED_DIRS: "${oc.env:MCP_ALLOWED_DIRS,./}"  # 逗号分隔的允许目录
```

提供工具：
- `read_file(path, encoding="utf-8")` — 读取文件（大文件截断到 1MB）
- `write_file(path, content, encoding="utf-8")` — 写入文件
- `list_directory(path, max_entries=200)` — 列出目录内容

安全限制：
- 路径白名单：通过 `MCP_ALLOWED_DIRS` 环境变量配置允许访问的目录
- 所有路径操作前校验 `Path.resolve()` 后是否在白名单内
- 默认允许当前工作目录

## Hook 集成

工具系统与 Hook 系统深度集成：

### on_tool_start

工具执行前触发，可修改参数：

```python
@hooks.register("on_tool_start")
def modify_args(ctx: HookContext, original_fn):
    if ctx.tool_name == "web_search":
        ctx.arguments["safe_search"] = True
    return original_fn(ctx)
```

### on_tool_end

工具执行后触发，可修改结果：

```python
@hooks.register("on_tool_end")
def filter_result(ctx: HookContext, original_fn):
    result = original_fn(ctx)
    if ctx.tool_name == "web_search":
        # 过滤或转换结果
        pass
    return result
```

### on_tool_result_format

工具结果格式化时触发：

```python
@hooks.register("on_tool_result_format")
def custom_format(ctx: HookContext, original_fn):
    if ctx.tool_name == "my_tool":
        return "自定义格式化结果"
    return original_fn(ctx)
```

### on_env_inject

MCP 子进程环境变量注入：

```python
@hooks.register("on_env_inject")
def inject_env(ctx: HookContext, original_fn):
    params = original_fn(ctx)
    params.env["MY_SECRET"] = os.environ.get("MY_SECRET", "")
    return params
```

## SecureContext 集成

工具执行时自动替换 SecureContext 占位符：

```python
# LLM 生成的工具调用参数
arguments = {"user_id": "[SECURE:user_id]", "query": "搜索内容"}

# 执行前自动替换
resolved_args = {"user_id": "real-uid-123", "query": "搜索内容"}
```

详见 [08-hook-and-secure-context.md](08-hook-and-secure-context.md)。

## 子 Agent 作为工具

Orchestrator 可以将子 Agent 暴露为工具供主 Agent 调用：

```python
# 自动生成工具定义
{
    "name": "agent-worker",
    "description": "调用子 Agent 执行子任务",
    "input_schema": {
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": "子任务描述"}
        }
    }
}
```

主 Agent 调用 `agent-worker` 工具时，框架自动创建 SubAgentRunner 执行。
