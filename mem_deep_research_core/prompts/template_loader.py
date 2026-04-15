"""
Prompt 模板加载器

从 Markdown 文件加载模板并支持变量替换。
使用 {{variable_name}} 语法作为占位符。

模板搜索顺序:
1. 用户指定的目录
2. 项目 config/templates/ 目录
3. 框架内置模板目录

Usage:
    from mem_deep_research_core.prompts.template_loader import PromptTemplateLoader

    loader = PromptTemplateLoader()
    template = loader.load_template("boxed_answer_system")
    prompt = loader.render_template(template, date="2025-01-15", mcp_tools="...")
"""

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 内置模板目录（框架自带）
BUILTIN_TEMPLATES_DIR = Path(__file__).parent / "templates"

# 默认项目模板目录
DEFAULT_PROJECT_TEMPLATES_DIR = "config/templates"


class PromptTemplateLoader:
    """从 Markdown 文件加载和渲染 Prompt 模板"""

    def __init__(
        self,
        templates_dir: Path | None = None,
        fallback_to_builtin: bool = True,
    ):
        """
        初始化模板加载器

        Args:
            templates_dir: 自定义模板目录
            fallback_to_builtin: 找不到模板时是否回退到内置模板
        """
        self.templates_dir = templates_dir
        self.fallback_to_builtin = fallback_to_builtin
        self._cache: dict[str, str] = {}

        # 构建搜索路径
        self._search_paths: list[Path] = []
        if templates_dir:
            self._search_paths.append(Path(templates_dir))
        self._search_paths.append(BUILTIN_TEMPLATES_DIR)

    def add_search_path(self, path: Path, priority: int = 0):
        """
        添加模板搜索路径

        Args:
            path: 目录路径
            priority: 优先级（0 最高）
        """
        path = Path(path)
        if path not in self._search_paths:
            self._search_paths.insert(priority, path)

    def _find_template_file(self, name: str) -> Path | None:
        """查找模板文件"""
        for search_path in self._search_paths:
            template_path = search_path / f"{name}.md"
            if template_path.exists():
                return template_path
        return None

    def load_template(self, name: str, use_cache: bool = True) -> str:
        """
        按名称加载模板文件

        Args:
            name: 模板名称（不含 .md 扩展名）
            use_cache: 是否使用缓存

        Returns:
            模板内容字符串

        Raises:
            FileNotFoundError: 模板文件不存在
        """
        if use_cache and name in self._cache:
            return self._cache[name]

        template_path = self._find_template_file(name)

        if not template_path:
            search_dirs = [str(p) for p in self._search_paths]
            logger.warning(
                f"[TemplateLoader] Template '{name}' NOT FOUND | "
                f"search_paths={search_dirs} cwd={Path.cwd()}"
            )
            raise FileNotFoundError(f"Template '{name}' not found in: {search_dirs}")

        with open(template_path, encoding="utf-8") as f:
            content = f.read()

        if use_cache:
            self._cache[name] = content

        logger.debug(f"[TemplateLoader] Loaded template: {name} from {template_path}")
        return content

    def render_template(self, template: str, **variables: Any) -> str:
        """
        渲染模板，替换变量

        使用 {{variable_name}} 语法。
        缺失的变量替换为空字符串。

        Args:
            template: 模板内容字符串
            **variables: 要替换的变量

        Returns:
            渲染后的字符串
        """

        missing_vars = []

        def replace_var(match):
            var_name = match.group(1).strip()
            if var_name not in variables:
                missing_vars.append(var_name)
            value = variables.get(var_name, "")
            return str(value) if value is not None else ""

        pattern = r"\{\{([^}]+)\}\}"
        rendered = re.sub(pattern, replace_var, template)

        if missing_vars:
            logger.warning(f"[TemplateLoader] Missing template variables: {missing_vars}")

        return rendered

    def load_and_render(self, name: str, **variables: Any) -> str:
        """
        加载并渲染模板（一步完成）

        Args:
            name: 模板名称
            **variables: 要替换的变量

        Returns:
            渲染后的字符串
        """
        template = self.load_template(name)
        return self.render_template(template, **variables)

    def template_exists(self, name: str) -> bool:
        """检查模板是否存在"""
        return self._find_template_file(name) is not None

    def list_templates(self) -> list[str]:
        """列出所有可用模板"""
        templates = set()
        for search_path in self._search_paths:
            if search_path.exists():
                for f in search_path.glob("*.md"):
                    templates.add(f.stem)
        return sorted(templates)

    def get_template_path(self, name: str) -> Path | None:
        """获取模板文件的完整路径"""
        return self._find_template_file(name)

    def clear_cache(self):
        """清除缓存"""
        self._cache.clear()


# 全局实例
template_loader = PromptTemplateLoader()
