"""
SecureContext 测试

覆盖：
- 占位符生成与解析
- get_display_value: _secure 字段返回占位符，普通字段返回真实值
- get_real_value: 优先 _secure，回退顶层
- resolve_placeholders_in_args: 递归替换工具参数中的占位符
- UserContextBuilder 集成: system prompt 中的 secure 字段显示为占位符
"""

from mem_deep_research_core.core.secure_context import (
    SECURE_PLACEHOLDER_PATTERN,
    get_display_value,
    get_real_value,
    get_secure_fields,
    has_secure_fields,
    make_placeholder,
    resolve_placeholders_in_args,
)

# ========== 基础函数测试 ==========


class TestBasicFunctions:
    def test_get_secure_fields_present(self):
        ctx = {"user_name": "test", "_secure": {"user_id": "123"}}
        assert get_secure_fields(ctx) == {"user_id": "123"}

    def test_get_secure_fields_absent(self):
        ctx = {"user_name": "test"}
        assert get_secure_fields(ctx) == {}

    def test_get_secure_fields_none(self):
        assert get_secure_fields(None) == {}

    def test_has_secure_fields(self):
        assert has_secure_fields({"_secure": {"k": "v"}})
        assert not has_secure_fields({"_secure": {}})
        assert not has_secure_fields({"user_id": "123"})
        assert not has_secure_fields(None)

    def test_make_placeholder(self):
        assert make_placeholder("user_id") == "[SECURE:user_id]"
        assert make_placeholder("api_key") == "[SECURE:api_key]"

    def test_placeholder_pattern(self):
        text = "User is [SECURE:user_id] from org [SECURE:org_id]"
        matches = SECURE_PLACEHOLDER_PATTERN.findall(text)
        assert matches == ["user_id", "org_id"]


# ========== get_display_value 测试 ==========


class TestGetDisplayValue:
    def test_secure_field_returns_placeholder(self):
        ctx = {"_secure": {"user_id": "real-123"}}
        assert get_display_value(ctx, "user_id") == "[SECURE:user_id]"

    def test_normal_field_returns_real_value(self):
        ctx = {"user_name": "张三"}
        assert get_display_value(ctx, "user_name") == "张三"

    def test_missing_field_returns_default(self):
        ctx = {"user_name": "test"}
        assert get_display_value(ctx, "org_id") == ""
        assert get_display_value(ctx, "org_id", "N/A") == "N/A"

    def test_none_context(self):
        assert get_display_value(None, "user_id") == ""
        assert get_display_value(None, "user_id", "default") == "default"

    def test_secure_overrides_top_level(self):
        """_secure 中的字段优先于顶层同名字段"""
        ctx = {"user_id": "top-level", "_secure": {"user_id": "secure-value"}}
        assert get_display_value(ctx, "user_id") == "[SECURE:user_id]"

    def test_non_string_value(self):
        ctx = {"count": 42}
        assert get_display_value(ctx, "count") == "42"


# ========== get_real_value 测试 ==========


class TestGetRealValue:
    def test_secure_field_returns_real(self):
        ctx = {"_secure": {"user_id": "real-123"}}
        assert get_real_value(ctx, "user_id") == "real-123"

    def test_fallback_to_top_level(self):
        ctx = {"user_id": "top-123"}
        assert get_real_value(ctx, "user_id") == "top-123"

    def test_secure_takes_priority(self):
        ctx = {"user_id": "top-level", "_secure": {"user_id": "secure-value"}}
        assert get_real_value(ctx, "user_id") == "secure-value"

    def test_missing_returns_default(self):
        ctx = {"other": "val"}
        assert get_real_value(ctx, "user_id") == ""
        assert get_real_value(ctx, "user_id", "fallback") == "fallback"

    def test_none_context(self):
        assert get_real_value(None, "user_id") == ""


# ========== resolve_placeholders_in_args 测试 ==========


class TestResolvePlaceholders:
    def test_simple_string_replacement(self):
        ctx = {"_secure": {"user_id": "real-123"}}
        args = {"query": "Find user [SECURE:user_id]"}
        result = resolve_placeholders_in_args(args, ctx)
        assert result["query"] == "Find user real-123"

    def test_multiple_placeholders(self):
        ctx = {"_secure": {"user_id": "u123", "org_id": "o456"}}
        args = {"text": "[SECURE:user_id] in [SECURE:org_id]"}
        result = resolve_placeholders_in_args(args, ctx)
        assert result["text"] == "u123 in o456"

    def test_nested_dict(self):
        ctx = {"_secure": {"user_id": "real-123"}}
        args = {"filter": {"owner": "[SECURE:user_id]", "active": True}}
        result = resolve_placeholders_in_args(args, ctx)
        assert result["filter"]["owner"] == "real-123"
        assert result["filter"]["active"] is True

    def test_nested_list(self):
        ctx = {"_secure": {"user_id": "real-123"}}
        args = {"ids": ["[SECURE:user_id]", "other-id"]}
        result = resolve_placeholders_in_args(args, ctx)
        assert result["ids"] == ["real-123", "other-id"]

    def test_no_secure_context(self):
        args = {"query": "hello [SECURE:user_id]"}
        result = resolve_placeholders_in_args(args, {"user_name": "test"})
        # 没有 _secure，不做任何替换
        assert result["query"] == "hello [SECURE:user_id]"

    def test_none_context(self):
        args = {"query": "hello"}
        result = resolve_placeholders_in_args(args, None)
        assert result == args

    def test_unknown_placeholder_preserved(self):
        ctx = {"_secure": {"user_id": "real-123"}}
        args = {"text": "[SECURE:unknown_field]"}
        result = resolve_placeholders_in_args(args, ctx)
        assert result["text"] == "[SECURE:unknown_field]"

    def test_non_string_values_unchanged(self):
        ctx = {"_secure": {"user_id": "real-123"}}
        args = {"count": 42, "flag": True, "data": None}
        result = resolve_placeholders_in_args(args, ctx)
        assert result == args

    def test_original_not_modified(self):
        ctx = {"_secure": {"user_id": "real-123"}}
        args = {"query": "[SECURE:user_id]"}
        result = resolve_placeholders_in_args(args, ctx)
        assert result["query"] == "real-123"
        assert args["query"] == "[SECURE:user_id]"  # 原始未修改

    def test_fallback_to_top_level(self):
        """_secure 中没有但顶层有的字段也能解析"""
        ctx = {"user_id": "top-123", "_secure": {"org_id": "org-456"}}
        args = {"text": "[SECURE:user_id] [SECURE:org_id]"}
        result = resolve_placeholders_in_args(args, ctx)
        assert result["text"] == "top-123 org-456"


# ========== UserContextBuilder 集成测试 ==========


class TestUserContextBuilderIntegration:
    def test_secure_user_id_shows_placeholder(self):
        from mem_deep_research_core.core.user_context import UserContextBuilder

        ctx = {
            "user_name": "张三",
            "_secure": {"user_id": "real-123", "org_id": "org-456"},
        }
        builder = UserContextBuilder(context=ctx, chinese_context=True)
        result = builder.build_user_identity_context()

        assert "[SECURE:user_id]" in result
        assert "[SECURE:org_id]" in result
        assert "real-123" not in result
        assert "org-456" not in result
        assert "张三" in result

    def test_secure_user_id_english(self):
        from mem_deep_research_core.core.user_context import UserContextBuilder

        ctx = {
            "_secure": {"user_id": "real-123", "org_id": "org-456"},
            "timezone": "UTC",
        }
        builder = UserContextBuilder(context=ctx, chinese_context=False)
        result = builder.build_user_identity_context()

        assert "[SECURE:user_id]" in result
        assert "[SECURE:org_id]" in result
        assert "real-123" not in result

    def test_no_secure_shows_real_values(self):
        from mem_deep_research_core.core.user_context import UserContextBuilder

        ctx = {"user_id": "visible-123", "org_id": "visible-org"}
        builder = UserContextBuilder(context=ctx, chinese_context=True)
        result = builder.build_user_identity_context()

        assert "visible-123" in result
        assert "visible-org" in result
        assert "[SECURE:" not in result

    def test_mirror_mode_secure(self):
        from mem_deep_research_core.core.user_context import UserContextBuilder

        ctx = {
            "mode": "mirror",
            "_secure": {"user_id": "real-123"},
        }
        builder = UserContextBuilder(context=ctx, chinese_context=True)
        result = builder.build_user_identity_context()

        assert "[SECURE:user_id]" in result
        assert "real-123" not in result

    def test_mirror_mode_with_user_name_not_secure(self):
        """user_name 不在 _secure 中，正常显示"""
        from mem_deep_research_core.core.user_context import UserContextBuilder

        ctx = {
            "mode": "mirror",
            "user_name": "张三",
            "_secure": {"user_id": "real-123"},
        }
        builder = UserContextBuilder(context=ctx, chinese_context=True)
        result = builder.build_user_identity_context()

        assert "张三" in result
        assert "real-123" not in result
