"""
用户上下文构建模块

处理用户身份信息、对话历史等上下文的构建。
支持 _secure 字段：敏感数据在 system prompt 中显示为占位符。

NOTE: This module is NOT used by the framework core. It is provided as a utility
for users who want to inject user identity via hooks (e.g., on_system_prompt_build).
See example_project/hooks.py for usage examples.
"""

from typing import Any

from mem_deep_research_core.core.secure_context import build_secure_usage_prompt, get_display_value


class UserContextBuilder:
    """用户上下文构建器

    Optional utility class — not used by framework core.
    Import and use in your hooks to inject user identity into the system prompt.
    """

    def __init__(self, context: dict[str, Any] | None = None, chinese_context: bool = False):
        """
        初始化用户上下文构建器

        Args:
            context: 用户上下文字典，包含 user_id, user_name, org_id, timezone, mode 等
            chinese_context: 是否使用中文上下文
        """
        self.context = context or {}
        self.chinese_context = chinese_context

    def build_user_identity_context(self) -> str:
        """
        Build user identity context string to inject into the system prompt.

        Returns:
            A formatted string with user identity information, or empty string if no context.
        """
        if not self.context:
            return ""

        # _secure 字段自动显示为占位符 [SECURE:xxx]，非 _secure 字段显示真实值
        user_id = get_display_value(self.context, "user_id")
        user_name = get_display_value(self.context, "user_name")
        org_id = get_display_value(self.context, "org_id")
        timezone = get_display_value(self.context, "timezone", "UTC")
        mode = self.context.get("mode", "default")
        meta_chat_history = self.context.get("meta_chat_history")

        if not user_id:
            return ""

        # Mirror mode: Direct identity - the AI IS the user
        if mode == "mirror":
            return self._build_mirror_context(
                user_id, user_name, meta_chat_history
            )

        # Default mode: Service-oriented context
        return self._build_default_context(
            user_id, user_name, org_id, timezone, meta_chat_history
        )

    def _build_mirror_context(
        self, user_id: str, user_name: str, meta_chat_history: list | None
    ) -> str:
        """Mirror mode: the AI IS the user."""
        if user_name:
            if self.chinese_context:
                identity_str = f"你是 {user_name}。以第一人称进行所有操作。\n"
            else:
                identity_str = f"You are {user_name}. Perform all actions in the first person.\n"
        else:
            if self.chinese_context:
                identity_str = f"你的用户ID是 {user_id}。\n"
            else:
                identity_str = f"Your user ID is {user_id}.\n"

        # Add role purpose context
        role_purpose = self.context.get("role_purpose")
        if role_purpose:
            if self.chinese_context:
                identity_str += f"\n## 你的角色与职责\n\n{role_purpose}\n"
            else:
                identity_str += f"\n## Your Role & Responsibilities\n\n{role_purpose}\n"

        # Add conversation history
        history_str = self._build_chat_history_context(meta_chat_history)
        if history_str:
            identity_str += "\n" + history_str

        # Append secure placeholder usage guide
        identity_str += build_secure_usage_prompt(self.context, chinese=self.chinese_context)

        return identity_str

    def _build_default_context(
        self,
        user_id: str,
        user_name: str,
        org_id: str,
        timezone: str,
        meta_chat_history: list | None,
    ) -> str:
        """Default mode: service-oriented context."""
        # Build user fields
        fields = []
        if self.chinese_context:
            if user_name:
                fields.append(f"- **用户名称**: {user_name}")
            fields.append(f"- **用户ID**: {user_id}")
            if org_id:
                fields.append(f"- **组织ID**: {org_id}")
            fields.append(f"- **时区**: {timezone}")
            header = "## 当前用户信息"
        else:
            if user_name:
                fields.append(f"- **User Name**: {user_name}")
            fields.append(f"- **User ID**: {user_id}")
            if org_id:
                fields.append(f"- **Organization ID**: {org_id}")
            fields.append(f"- **Timezone**: {timezone}")
            header = "## Current User Information"

        user_fields = "\n".join(fields)
        result = f"{header}\n\n{user_fields}\n"

        # Add conversation history
        history_str = self._build_chat_history_context(meta_chat_history)
        if history_str:
            result += "\n" + history_str

        # Append secure placeholder usage guide
        secure_prompt = build_secure_usage_prompt(self.context, chinese=self.chinese_context)
        if secure_prompt:
            result += secure_prompt

        return result

    def _build_chat_history_context(self, meta_chat_history: list) -> str:
        """Build conversation history context string."""
        if not meta_chat_history or len(meta_chat_history) == 0:
            return ""

        history_parts = []
        for item in meta_chat_history:
            # Support both dict and object (MetaChatHistoryItem) formats
            if hasattr(item, "role"):
                role = item.role or "user"
                content = item.content or ""
                name = getattr(item, "name", None)
            else:
                role = item.get("role", "user")
                content = item.get("content", "")
                name = item.get("name")

            role_labels = self._get_role_labels()
            if role == "user":
                label = name if name else role_labels["user"]
            elif role == "assistant":
                label = role_labels["assistant"]
            elif role == "system":
                label = role_labels["system"]
            else:
                label = role

            history_parts.append(f"**{label}**: {content}\n")

        history_content = "\n".join(history_parts) + "\n---\n"

        if self.chinese_context:
            header = "## 对话历史"
        else:
            header = "## Conversation History"

        return f"{header}\n\n{history_content}\n"

    def _get_role_labels(self) -> dict[str, str]:
        """Get role display labels based on language."""
        if self.chinese_context:
            return {"user": "用户", "assistant": "助手", "system": "系统"}
        return {"user": "User", "assistant": "Assistant", "system": "System"}

    def get_prompt_class_for_mode(self, mode: str, default_prompt_class: str) -> str:
        """
        Get the appropriate prompt class name based on the operation mode.

        Args:
            mode: The operation mode from context ("default", "mirror", etc.)
            default_prompt_class: The default prompt class from config

        Returns:
            The prompt class name to use.
        """
        mode_prompt_mapping = {
            "mirror": "MainAgentPrompt_Mirror",
            "default": default_prompt_class,
        }
        return mode_prompt_mapping.get(mode, default_prompt_class)


def detect_language_by_chars(text: str) -> str:
    """Fallback: detect language by character analysis."""
    chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    japanese_chars = sum(1 for c in text if "\u3040" <= c <= "\u309f" or "\u30a0" <= c <= "\u30ff")
    korean_chars = sum(1 for c in text if "\uac00" <= c <= "\ud7af" or "\u1100" <= c <= "\u11ff")

    total_len = len(text)
    if total_len == 0:
        return "English"

    if japanese_chars / total_len > 0.05:
        return "Japanese"
    if korean_chars / total_len > 0.05:
        return "Korean"
    if chinese_chars / total_len > 0.05:
        return "Chinese"

    return "English"
