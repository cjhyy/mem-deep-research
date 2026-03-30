# LLM Provider 系统

框架支持多种 LLM Provider，通过统一接口 `LLMProviderClientBase` 实现。

## 支持的 Provider

| Provider | 类名 | 基类 | 说明 |
|----------|------|------|------|
| Anthropic (原生) | `ClaudeAnthropicClient` | `LLMProviderClientBase` | 直连 Anthropic API |
| OpenAI (原生) | `GPTOpenAIClient` | `LLMProviderClientBase` | 直连 OpenAI API |
| OpenRouter Claude | `ClaudeOpenRouterClient` | `OpenAICompatibleClient` | 通过 OpenRouter |
| OpenRouter GPT-5 | `GPT5OpenRouterClient` | `OpenAICompatibleClient` | 通过 OpenRouter |
| OpenAI GPT-5 | `GPT5OpenAIClient` | `GPT5OpenRouterClient` | 直连 OpenAI |
| DeepSeek | `DeepSeekOpenRouterClient` | `OpenAICompatibleClient` | 通过 OpenRouter |
| OpenAI Compatible | `OpenAICompatibleClient` | `LLMProviderClientBase` | 通用 OpenAI 兼容 |

## 配置方式

```yaml
llm:
  provider_class: "ClaudeOpenRouterClient"  # Provider 类名
  model_name: "anthropic/claude-sonnet-4"   # 模型标识
  temperature: 0.3
  max_tokens: 32000
  max_context_length: 128000

  # Provider 特定凭证
  openrouter_api_key: "${oc.env:OPENROUTER_API_KEY,}"
  anthropic_api_key: "${oc.env:ANTHROPIC_API_KEY,}"
  openai_api_key: "${oc.env:OPENAI_API_KEY,}"
```

## Provider 基类

### LLMProviderClientBase

所有 Provider 的抽象基类：

```python
@dataclass
class LLMProviderClientBase:
    task_id: str
    cfg: DictConfig

    # 子类必须实现
    def _create_client(self, config): ...
    async def _create_message(self, system_prompt, messages, tools_definitions, **kwargs): ...
    def process_llm_response(self, response, message_history): ...
    def extract_tool_calls_info(self, response, text): ...
    def update_message_history(self, history, tool_call_info, tool_calls_exceeded=False): ...
```

### 关键方法

| 方法 | 说明 |
|------|------|
| `create_message()` | 统一异步消息创建接口 |
| `process_llm_response()` | 处理 LLM 响应，提取文本 |
| `extract_tool_calls_info()` | 从响应中提取工具调用 |
| `update_message_history()` | 更新消息历史（含工具结果） |
| `get_retry_decorator()` | 创建 tenacity 重试装饰器 |
| `get_effective_temperature()` | 获取当前温度（含 boost） |
| `set_temperature_boost()` | 设置温度提升（循环打破） |
| `get_usage()` | 获取用量统计 |
| `close()` / `close_async()` | 关闭连接 |

### 消息历史管理

```python
# 过滤旧消息
_filter_message_history(message_history)

# 保留最近 N 个工具结果
_remove_tool_result_from_messages(message_history, keep_tool_result)

# 添加缓存控制
_apply_cache_control(messages)
```

## OpenAICompatibleClient

大多数 Provider 的共用基类，实现 OpenAI API 协议：

### 上下文预检

调用 LLM 前自动检测上下文是否超限：

```python
async def _preflight_context_check(self, system_prompt, messages):
    """
    Phase 1: 替换旧消息内容为 "[已压缩]"
    Phase 2: 删除旧消息对
    """
```

### Token 估算

使用 tiktoken 库进行 token 估算：

```python
def _estimate_tokens(self, text: str) -> int:
    """使用 o200k_base 或 cl100k_base 编码器估算"""
```

### 工具结果去重

```python
def _deduplicate_tool_results(self, tool_call_info):
    """Hash 相同的工具结果，合并为 1 条 + 去重说明"""
```

### 子类扩展点

| 方法 | 说明 |
|------|------|
| `_get_api_credentials()` | 返回 (api_key, base_url) |
| `_build_extra_body()` | 构建 OpenRouter provider routing |
| `_customize_params()` | 自定义 API 参数 |
| `_post_response_hook()` | 响应后处理 |
| `_use_cache_control()` | 是否启用缓存控制 |

## ClaudeAnthropicClient

直连 Anthropic API 的专用实现：

- 原生 tool_use block 支持
- Anthropic 专用缓存控制（`cache_control: {"type": "ephemeral"}`）
- 支持同步和异步调用

## 重试机制

```yaml
llm:
  retry_max_attempts: 3         # 最大重试次数
  retry_strategy: "exponential" # exponential | fixed
  retry_multiplier: 1.0         # 指数退避乘数
  retry_wait_seconds: 2.0       # 固定等待时间
```

使用 tenacity 库实现：
- 指数退避（exponential）: 等待时间倍增
- 固定等待（fixed）: 固定间隔重试
- 自动跳过不可重试的异常

## LLMClient 工厂

```python
from mem_deep_research_core.llm.client import LLMClient

client = LLMClient(
    task_id="task-001",
    cfg=config,
    llm_config=llm_config,
    task_log=task_log,
)
```

工厂函数动态导入 `provider_class` 指定的类并实例化。

## 自定义 Provider

继承 `OpenAICompatibleClient` 或 `LLMProviderClientBase`：

```python
from mem_deep_research_core.llm.providers.openai_compatible_client import OpenAICompatibleClient

class MyProviderClient(OpenAICompatibleClient):
    def _get_api_credentials(self):
        return (self._get_config_value("my_api_key"), "https://api.my-provider.com/v1")

    def _customize_params(self, params):
        params["extra_param"] = "value"
        return params
```

然后在配置中引用：

```yaml
llm:
  provider_class: "MyProviderClient"
  model_name: "my-model"
  my_api_key: "${oc.env:MY_API_KEY}"
```
