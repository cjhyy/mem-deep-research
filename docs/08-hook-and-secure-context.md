# 钩子系统与 SecureContext

## 钩子系统

### 概述

HookRegistry 是全局单例，支持在 Agent 生命周期的关键节点注入自定义逻辑。

### 支持的钩子

| 钩子 | 时机 | 可修改内容 |
|------|------|-----------|
| `on_agent_start` | Agent 开始执行 | — |
| `on_agent_end` | Agent 执行结束 | — |
| `on_turn_start` | 每轮开始 | — |
| `on_turn_end` | 每轮结束 | — |
| `on_tool_start` | 工具调用前 | arguments |
| `on_tool_end` | 工具调用后 | tool_result |
| `on_final_answer` | 最终答案后处理 | 返回值（str） |
| `on_tool_result_format` | 结果格式化 | 返回值 |
| `on_thinking_generate` | thinking 描述生成 | 返回值 |
| `on_env_inject` | MCP 环境变量注入 | server_params |
| `on_message_intercept` | 消息拦截 | — |

### 注册方式

#### 装饰器注册

```python
from mem_deep_research_core.core.hooks import hooks, HookContext

@hooks.register("on_tool_end", priority=10)
def my_hook(ctx: HookContext, original_fn):
    result = original_fn(ctx)   # 调用链中的下一个
    # 修改 result
    return result
```

#### 直接注册

```python
hooks.register_fn("on_tool_end", my_function, priority=10)
```

#### 便捷装饰器

```python
from mem_deep_research_core.core.hooks import on_tool_end, on_agent_start

@on_tool_end(priority=10)
def my_hook(ctx, original_fn):
    return original_fn(ctx)

@on_agent_start()
def startup(ctx, original_fn):
    print("Agent 启动")
    return original_fn(ctx)
```

### HookContext

```python
@dataclass
class HookContext:
    hook_name: str              # 当前钩子名
    query: str = ""             # Agent 查询
    result: str = ""            # Agent 结果
    context: dict = None        # 用户上下文

    # 工具相关
    tool_name: str = ""
    server_name: str = ""
    arguments: dict = None      # 工具参数（on_tool_start 可修改）
    tool_result: Any = None     # 工具结果（on_tool_end 可修改）
    duration_ms: float = 0

    # 轮次相关
    turn_number: int = 0
    tool_calls_count: int = 0

    # 环境注入
    server_params: Any = None   # on_env_inject 可修改

    extra: Dict[str, Any] = None  # 附加数据
```

### 执行链

钩子按优先级排序，形成调用链：

```
Hook(priority=10) → Hook(priority=5) → Hook(priority=0) → 框架默认逻辑
```

每个 Hook 接收 `(ctx, next_fn)`，调用 `next_fn(ctx)` 传递给链中的下一个。不调用 `next_fn` 则完全替换逻辑。

### 项目级钩子

在项目目录下创建 `hooks.py`，框架自动加载：

```python
# my_project/hooks.py
import os
from mem_deep_research_core.core.hooks import hooks, HookContext

@hooks.register("on_env_inject", priority=10)
def inject_env(ctx: HookContext, original_fn):
    params = original_fn(ctx)
    params.env["MY_KEY"] = os.environ.get("MY_KEY", "")
    return params

@hooks.register("on_tool_result_format")
def format_result(ctx: HookContext, original_fn):
    if ctx.tool_name == "my_tool":
        return "自定义格式"
    return original_fn(ctx)
```

加载函数：

```python
from mem_deep_research_core.core.hooks import load_project_hooks
load_project_hooks("./my_project")
```

### Hook API

```python
class HookRegistry:
    def register(self, hook_name, priority=0)       # 装饰器
    def register_fn(self, hook_name, fn, priority=0) # 直接注册
    def set_default(self, hook_name, default_fn)     # 设置默认逻辑
    def call(self, hook_name, ctx)                   # 执行钩子链
    def has_hooks(self, hook_name) -> bool            # 是否有注册
    def clear(self, hook_name=None)                   # 清除注册
    def list_hooks() -> dict                          # 列出所有钩子
```

---

## SecureContext

### 概述

SecureContext 实现敏感数据的自动隔离：在 System Prompt 中显示为占位符，工具执行前自动替换为真实值。

### 使用方式

```python
context = {
    "user_name": "张三",              # 正常显示给 LLM
    "timezone": "Asia/Shanghai",      # 正常显示给 LLM
    "_secure": {
        "user_id": "uid-real-123",    # LLM 看到 [SECURE:user_id]
        "api_token": "sk-secret",     # LLM 看到 [SECURE:api_token]
        "org_id": "org-456",          # LLM 看到 [SECURE:org_id]
    }
}
```

### 工作流程

```
1. System Prompt 构建
   context["_secure"]["user_id"] → "[SECURE:user_id]"

2. LLM 生成工具调用
   {"tool": "query_db", "arguments": {"user": "[SECURE:user_id]"}}

3. 工具执行前替换
   {"user": "[SECURE:user_id]"} → {"user": "uid-real-123"}

4. 工具执行
   实际使用真实值调用
```

### API

```python
from mem_deep_research_core.core.secure_context import (
    get_secure_fields,          # 获取所有 _secure 字段
    has_secure_fields,          # 检查是否有 _secure
    make_placeholder,           # 生成 [SECURE:xxx] 占位符
    get_display_value,          # System Prompt 用（_secure 返回占位符）
    get_real_value,             # 工具执行用（始终返回真实值）
    resolve_placeholders_in_args,  # 递归替换工具参数中的占位符
    build_secure_usage_prompt,  # 生成 LLM 使用说明
)
```

### 占位符替换

`resolve_placeholders_in_args()` 支持深层嵌套结构的递归替换：

```python
# 输入
arguments = {
    "user_id": "[SECURE:user_id]",
    "filters": {
        "org": "[SECURE:org_id]",
        "tags": ["public", "[SECURE:api_token]"]
    }
}

# 输出
resolved = {
    "user_id": "uid-real-123",
    "filters": {
        "org": "org-456",
        "tags": ["public", "sk-secret"]
    }
}
```

### LLM 使用说明

`build_secure_usage_prompt()` 生成注入 System Prompt 的说明，告知 LLM：
- 哪些占位符可用
- 必须使用占位符引用敏感数据
- 禁止猜测真实值
- 支持中英文

```python
prompt = build_secure_usage_prompt(context, chinese=True)
# 输出示例:
# 以下敏感信息以占位符形式提供，请在工具调用中使用占位符：
# - [SECURE:user_id]: 用户ID
# - [SECURE:api_token]: API令牌
# 请勿猜测或推断真实值。
```
