"""
SecureContext — 隐私数据脱敏模块

将 context 中的敏感字段与 LLM 可见内容分离：
- _secure 字段：LLM 看到占位符 [SECURE:key]，工具调用时自动还原真实值
- 其他字段：正常暴露给 LLM

用法：
    # 用户传入
    context = {
        "user_name": "张三",          # 公开，LLM 可见
        "timezone": "Asia/Shanghai",  # 公开，LLM 可见
        "_secure": {
            "user_id": "real-123",    # LLM 看到 [SECURE:user_id]
            "org_id": "org-456",      # LLM 看到 [SECURE:org_id]
            "api_key": "sk-xxx",      # LLM 看到 [SECURE:api_key]
        },
    }

    dr.run(task, context=context)

框架自动处理：
1. system prompt 中 _secure 字段显示为占位符
2. LLM 生成的工具参数中的占位符在执行前自动替换回真实值
3. _mcp_context 注入使用真实值
"""

import logging
import re
from typing import Any

logger = logging.getLogger("mem_deep_research")

# 占位符格式: [SECURE:field_name]
SECURE_PLACEHOLDER_PATTERN = re.compile(r"\[SECURE:(\w+)\]")
SECURE_KEY = "_secure"


def get_secure_fields(context: dict[str, Any] | None) -> dict[str, Any]:
    """获取 context 中的 _secure 字段"""
    if not context:
        return {}
    return context.get(SECURE_KEY, {})


def has_secure_fields(context: dict[str, Any] | None) -> bool:
    """检查 context 是否包含 _secure 字段"""
    return bool(get_secure_fields(context))


def make_placeholder(field_name: str) -> str:
    """生成占位符"""
    return f"[SECURE:{field_name}]"


def resolve_secure_value(context: dict[str, Any] | None, field_name: str) -> str | None:
    """从 _secure 中获取真实值，回退到 context 顶层"""
    if not context:
        return None
    secure = context.get(SECURE_KEY, {})
    if field_name in secure:
        return str(secure[field_name])
    # 回退：顶层字段也可能被标记为 secure（向后兼容）
    val = context.get(field_name)
    return str(val) if val is not None else None


def get_display_value(context: dict[str, Any] | None, field_name: str, default: str = "") -> str:
    """获取用于 LLM 展示的值

    如果字段在 _secure 中，返回占位符；否则返回真实值。
    """
    if not context:
        return default
    secure = context.get(SECURE_KEY, {})
    if field_name in secure:
        return make_placeholder(field_name)
    val = context.get(field_name)
    if val is not None:
        return str(val)
    return default


def get_real_value(context: dict[str, Any] | None, field_name: str, default: str = "") -> str:
    """获取真实值（优先 _secure，回退顶层）

    用于 _mcp_context 注入、env 注入等需要真实值的场景。
    """
    if not context:
        return default
    secure = context.get(SECURE_KEY, {})
    if field_name in secure:
        return str(secure[field_name])
    val = context.get(field_name)
    if val is not None:
        return str(val)
    return default


def build_secure_usage_prompt(
    context: dict[str, Any] | None,
    chinese: bool = False,
) -> str:
    """根据 _secure 中实际注册的字段，自动生成 LLM 使用说明

    告诉 LLM：
    1. 哪些占位符可用
    2. 必须原样传递给工具参数
    3. 不能猜测或硬编码真实值

    Args:
        context: 包含 _secure 的用户 context
        chinese: 是否使用中文

    Returns:
        prompt 段落字符串，无 _secure 字段时返回空字符串
    """
    secure = get_secure_fields(context)
    if not secure:
        return ""

    [make_placeholder(k) for k in secure]

    if chinese:
        lines = [
            "\n## 安全占位符（必须遵守）\n",
            "部分敏感值隐藏在 `[SECURE:xxx]` 占位符后面。在工具调用参数中**必须原样使用**这些占位符，系统会在执行前自动替换为真实值。\n",
            "**可用占位符：**\n",
        ]
        for key in secure:
            lines.append(f"- `{make_placeholder(key)}`\n")
        lines.append("\n**规则：**\n")
        lines.append("- **禁止**猜测或硬编码这些字段的真实值\n")
        lines.append("- **禁止**省略这些字段 — 没有它们工具调用会失败\n")
        lines.append("- 必须完全按照上面的格式复制占位符\n")
    else:
        lines = [
            "\n## Secure Placeholders (MUST follow)\n",
            "Some sensitive values are hidden behind `[SECURE:xxx]` placeholders. "
            "You MUST use these exact placeholder strings when passing them as tool arguments. "
            "The system will automatically replace them with real values before execution.\n",
            "\n**Available placeholders:**\n",
        ]
        for key in secure:
            lines.append(f"- `{make_placeholder(key)}`\n")
        lines.append("\n**Rules:**\n")
        lines.append("- NEVER guess or hardcode real values for these fields\n")
        lines.append("- NEVER omit these fields — without them, tool calls will fail\n")
        lines.append("- Always copy the placeholder exactly as shown\n")

    return "".join(lines)


def resolve_placeholders_in_args(
    arguments: dict[str, Any],
    context: dict[str, Any] | None,
) -> dict[str, Any]:
    """将工具参数中的 [SECURE:xxx] 占位符替换回真实值

    递归处理字符串值（包括嵌套 dict/list）。
    LLM 可能在生成的工具参数中引用占位符，需要在执行前还原。

    Args:
        arguments: LLM 生成的工具参数
        context: 包含 _secure 的用户 context

    Returns:
        替换后的参数（新 dict，不修改原始）
    """
    if not context or not has_secure_fields(context):
        return arguments
    return _resolve_recursive(arguments, context)


def _resolve_recursive(obj: Any, context: dict[str, Any]) -> Any:
    """递归替换占位符"""
    if isinstance(obj, str):
        return _resolve_string(obj, context)
    if isinstance(obj, dict):
        return {k: _resolve_recursive(v, context) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_recursive(item, context) for item in obj]
    return obj


def _resolve_string(text: str, context: dict[str, Any]) -> str:
    """替换字符串中的 [SECURE:xxx] 占位符"""

    def replacer(match):
        field_name = match.group(1)
        real_value = resolve_secure_value(context, field_name)
        if real_value is not None:
            return real_value
        # 找不到真实值，保留占位符
        return match.group(0)

    return SECURE_PLACEHOLDER_PATTERN.sub(replacer, text)
