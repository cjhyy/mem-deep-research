"""拦截器配置单元测试"""

from mem_deep_research_core.core.interceptor_config import InterceptorConfig, InterceptorPresets


class TestInterceptorConfig:
    def test_default_config(self):
        """默认配置"""
        config = InterceptorConfig()
        assert isinstance(config.filter_tags, list)
        assert isinstance(config.reasoning_tags, list)
        assert config.show_reasoning is True
        assert config.show_tool_calls is True
        assert config.show_text_output is True

    def test_custom_filter_tags(self):
        """自定义过滤标签"""
        config = InterceptorConfig(filter_tags=["custom_tag"])
        assert "custom_tag" in config.filter_tags

    def test_get_all_filter_keywords(self):
        """获取所有过滤关键字"""
        config = InterceptorConfig(filter_tags=["my_tag"])
        keywords = config.get_all_filter_keywords()
        assert isinstance(keywords, list)
        # 应该包含格式化后的标签
        assert any("my_tag" in kw for kw in keywords)


class TestInterceptorPresets:
    def test_default_preset(self):
        """默认预设"""
        config = InterceptorPresets.from_name("default")
        assert isinstance(config, InterceptorConfig)

    def test_verbose_preset(self):
        """详细预设"""
        config = InterceptorPresets.from_name("verbose")
        assert config.show_reasoning is True
        assert config.show_tool_calls is True

    def test_minimal_preset(self):
        """精简预设"""
        config = InterceptorPresets.from_name("minimal")
        assert isinstance(config, InterceptorConfig)

    def test_unknown_preset_fallback(self):
        """未知预设回退到默认"""
        config = InterceptorPresets.from_name("nonexistent_preset")
        assert isinstance(config, InterceptorConfig)
