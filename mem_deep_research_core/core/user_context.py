"""
用户上下文构建模块

处理用户身份信息、对话历史等上下文的构建。
支持 _secure 字段：敏感数据在 system prompt 中显示为占位符。
"""

from typing import Any

from mem_deep_research_core.core.secure_context import build_secure_usage_prompt, get_display_value


class UserContextBuilder:
    """用户上下文构建器"""

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

        This allows the LLM to know who the current user is when answering questions
        like "我是谁" (Who am I?).

        For mirror mode: Direct identity statement (e.g., "你是 XXX")
        For default mode: Service-oriented context (e.g., "你正在为 XXX 提供服务")

        Also includes conversation history (meta_chat_history) when provided.

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
            if self.chinese_context:
                if user_name:
                    identity_str = (
                        f"你是 **{user_name}**。当别人问你是谁时，直接说「我是{user_name}」。\n"
                    )
                else:
                    identity_str = f"你的用户ID是 {user_id}。\n"
            else:
                if user_name:
                    identity_str = f'You are **{user_name}**. When asked who you are, simply say "I\'m {user_name}".\n'
                else:
                    identity_str = f"Your user ID is {user_id}.\n"

            # Add role purpose context for mirror mode (auto-fetched)
            role_purpose = self.context.get("role_purpose")
            if role_purpose:
                if self.chinese_context:
                    identity_str += f"\n## 你的角色与职责\n\n{role_purpose}\n"
                else:
                    identity_str += f"\n## Your Role & Responsibilities\n\n{role_purpose}\n"

            # Add conversation history for mirror mode
            history_str = self._build_chat_history_context(meta_chat_history)
            if history_str:
                identity_str += "\n" + history_str

            # Append secure placeholder usage guide
            identity_str += build_secure_usage_prompt(self.context, chinese=self.chinese_context)

            return identity_str

        # Default mode: Service-oriented context
        if self.chinese_context:
            identity_parts = ["## 当前用户信息\n"]
            identity_parts.append("你正在为以下用户提供服务：\n")
            if user_name:
                identity_parts.append(f"- **用户名称**: {user_name}\n")
            identity_parts.append(f"- **用户ID**: {user_id}\n")
            if org_id:
                identity_parts.append(f"- **组织ID**: {org_id}\n")
            identity_parts.append(f"- **时区**: {timezone}\n")
            identity_parts.append(
                "\n当用户询问「我是谁」或类似身份相关问题时，你可以使用上述信息回答。"
            )
            identity_parts.append("你还可以使用 search_contacts 等工具查询该用户的更多详细信息。\n")
        else:
            identity_parts = ["## Current User Information\n"]
            identity_parts.append("You are serving the following user:\n")
            if user_name:
                identity_parts.append(f"- **User Name**: {user_name}\n")
            identity_parts.append(f"- **User ID**: {user_id}\n")
            if org_id:
                identity_parts.append(f"- **Organization ID**: {org_id}\n")
            identity_parts.append(f"- **Timezone**: {timezone}\n")
            identity_parts.append(
                "\nWhen the user asks 'Who am I?' or similar identity questions, you can use the above information to answer."
            )
            identity_parts.append(
                "You can also use tools like search_contacts to get more detailed information about this user.\n"
            )

        # Add conversation history for default mode
        history_str = self._build_chat_history_context(meta_chat_history)
        if history_str:
            identity_parts.append("\n" + history_str)

        # Append secure placeholder usage guide
        secure_prompt = build_secure_usage_prompt(self.context, chinese=self.chinese_context)
        if secure_prompt:
            identity_parts.append(secure_prompt)

        return "".join(identity_parts)

    def _build_chat_history_context(self, meta_chat_history: list) -> str:
        """
        Build conversation history context string to inject into the system prompt.

        This provides the LLM with context from previous conversation rounds.

        Args:
            meta_chat_history: List of chat history items with role, content, name fields.
                Format: [{"role": "user/assistant", "content": "...", "name": "..."}]

        Returns:
            A formatted string with conversation history, or empty string if no history.
        """
        if not meta_chat_history or len(meta_chat_history) == 0:
            return ""

        if self.chinese_context:
            history_parts = ["## 对话历史\n"]
            history_parts.append("以下是之前的对话记录，请参考这些上下文来理解当前问题：\n\n")
        else:
            history_parts = ["## Conversation History\n"]
            history_parts.append(
                "The following is the previous conversation history. Please refer to this context to understand the current question:\n\n"
            )

        for item in meta_chat_history:
            # Support both dict and object (MetaChatHistoryItem) formats
            if hasattr(item, "role"):
                # Object format (MetaChatHistoryItem)
                role = item.role or "user"
                content = item.content or ""
                name = getattr(item, "name", None)
            else:
                # Dict format
                role = item.get("role", "user")
                content = item.get("content", "")
                name = item.get("name")

            if role == "user":
                if self.chinese_context:
                    if name:
                        history_parts.append(f"**{name}**: {content}\n\n")
                    else:
                        history_parts.append(f"**用户**: {content}\n\n")
                else:
                    if name:
                        history_parts.append(f"**{name}**: {content}\n\n")
                    else:
                        history_parts.append(f"**User**: {content}\n\n")
            elif role == "assistant":
                if self.chinese_context:
                    history_parts.append(f"**助手**: {content}\n\n")
                else:
                    history_parts.append(f"**Assistant**: {content}\n\n")
            elif role == "system":
                if self.chinese_context:
                    history_parts.append(f"**系统**: {content}\n\n")
                else:
                    history_parts.append(f"**System**: {content}\n\n")

        history_parts.append("---\n\n")

        return "".join(history_parts)

    def get_prompt_class_for_mode(self, mode: str, default_prompt_class: str) -> str:
        """
        Get the appropriate prompt class name based on the operation mode.

        This allows different modes (e.g., 'mirror' vs 'default') to use different
        system prompts for specialized behavior.

        Args:
            mode: The operation mode from context ("default", "mirror", etc.)
            default_prompt_class: The default prompt class from config

        Returns:
            The prompt class name to use.
        """
        # Mode to prompt class mapping
        mode_prompt_mapping = {
            "mirror": "MainAgentPrompt_Mirror",  # Mirror chat uses specialized prompt
            "default": default_prompt_class,  # Default uses config value
        }

        # Return the mapped prompt class, or fall back to config default
        return mode_prompt_mapping.get(mode, default_prompt_class)


def detect_language_by_chars(text: str) -> str:
    """Fallback: detect language by character analysis."""
    chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    japanese_chars = sum(1 for c in text if "\u3040" <= c <= "\u309f" or "\u30a0" <= c <= "\u30ff")
    korean_chars = sum(1 for c in text if "\uac00" <= c <= "\ud7af" or "\u1100" <= c <= "\u11ff")

    total_len = len(text)
    if total_len == 0:
        return "English"

    # Check ratios (threshold: 5%)
    if japanese_chars / total_len > 0.05:
        return "Japanese"
    if korean_chars / total_len > 0.05:
        return "Korean"
    if chinese_chars / total_len > 0.05:
        return "Chinese"

    return "English"
