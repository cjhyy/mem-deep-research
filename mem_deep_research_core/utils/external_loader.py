"""
配置加载器

负责从框架目录和项目目录加载各类配置：
- 提示词 (src/prompts/)
- 提示词模板 (src/prompts/templates/)
- 工具配置 (config/tool/ 或 项目目录/config/tool/)
- Agent 配置 (config/)
- Skills (config/skills/)

使用方式:
    from mem_deep_research_core.utils.config_loader import config_loader

    # 设置项目目录（启用从项目目录加载工具）
    config_loader.set_project_dir("/path/to/my_project")

    # 加载配置
    prompt_class = config_loader.load_prompt_class("MyPrompt")
    tool_config = config_loader.load_tool_config("tool-name")
    skill_injector = config_loader.get_skill_injector()

    # 加载模板化 prompt
    prompt_instance = config_loader.load_template_prompt(
        system_template="boxed_answer_system",
        summarize_template="boxed_answer_summarize"
    )
"""

import importlib
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# 框架 config 目录路径（包内）
CONFIG_DIR = Path(__file__).parent.parent / "config"  # mem_deep_research_core/config


class ConfigLoader:
    """框架配置加载器

    支持从两个位置加载配置：
    1. 项目目录 (优先) - 通过 set_project_dir() 设置
    2. 框架内置目录 (回退)
    """

    def __init__(self):
        self._skill_injector: Any | None = None
        self._skill_injector_initialized = False
        self._llm_skill_selector: Any | None = None
        self._llm_skill_selector_initialized = False
        self._project_dir: Path | None = None

    def reset(self) -> None:
        """Reset all cached state for a fresh start."""
        self._skill_injector = None
        self._skill_injector_initialized = False
        self._llm_skill_selector = None
        self._llm_skill_selector_initialized = False
        self._project_dir = None

    def set_project_dir(self, project_dir: str | Path | None) -> None:
        """设置项目目录，用于加载项目级别的工具配置

        Args:
            project_dir: 项目目录路径，设置为 None 清除
        """
        new_dir = Path(project_dir) if project_dir is not None else None
        if new_dir != self._project_dir:
            # Project changed — invalidate cached skills/selectors
            self._skill_injector = None
            self._skill_injector_initialized = False
            self._llm_skill_selector = None
            self._llm_skill_selector_initialized = False
        if new_dir is not None:
            self._project_dir = new_dir
            logger.info(f"[ConfigLoader] Project directory set to: {self._project_dir}")
        else:
            self._project_dir = None
            logger.info("[ConfigLoader] Project directory cleared")

    def get_project_dir(self) -> Path | None:
        """获取当前项目目录"""
        return self._project_dir

    def load_prompt_class(self, name: str) -> type:
        """
        加载提示词类。

        Args:
            name: 提示词类名

        Returns:
            提示词类

        Raises:
            ValueError: 如果找不到提示词类
        """
        try:
            prompts_module = importlib.import_module("mem_deep_research_core.prompts")
            if hasattr(prompts_module, name):
                logger.debug(f"Loading prompt '{name}'")
                return getattr(prompts_module, name)
        except (ImportError, AttributeError) as e:
            logger.warning(f"Failed to load prompt class '{name}': {e}")

        raise ValueError(f"Prompt class not found: {name}")

    def load_tool_config(self, name: str) -> dict:
        """
        加载工具配置。

        优先从项目目录加载，如果不存在则回退到框架内置配置。

        查找顺序：
        1. {project_dir}/config/tool/{name}.yaml
        2. {framework}/config/tool/{name}.yaml

        Args:
            name: 工具名称

        Returns:
            工具配置字典 (已解析环境变量插值)

        Raises:
            ValueError: 如果找不到工具配置
        """
        from omegaconf import OmegaConf

        # 1. 优先从项目目录加载
        if self._project_dir is not None:
            project_tool_path = self._project_dir / "config" / "tool" / f"{name}.yaml"
            if project_tool_path.exists():
                logger.info(
                    f"[ConfigLoader] Loading tool '{name}' from project: {project_tool_path}"
                )
                tool_cfg = OmegaConf.load(project_tool_path)
                resolved = OmegaConf.to_container(tool_cfg, resolve=True)

                # 处理相对路径的 args（相对于项目目录）
                if "args" in resolved:
                    resolved["args"] = self._resolve_tool_args(resolved["args"], self._project_dir)

                return resolved

        # 2. 回退到框架内置配置
        tool_path = CONFIG_DIR / "tool" / f"{name}.yaml"
        if tool_path.exists():
            logger.debug(f"[ConfigLoader] Loading tool '{name}' from framework: {tool_path}")
            tool_cfg = OmegaConf.load(tool_path)
            return OmegaConf.to_container(tool_cfg, resolve=True)

        raise ValueError(f"Tool config not found: {name}")

    def _resolve_tool_args(self, args: list, project_dir: Path) -> list:
        """解析工具参数中的相对路径

        将相对于项目目录的路径转换为绝对路径。

        Args:
            args: 工具参数列表
            project_dir: 项目目录

        Returns:
            解析后的参数列表
        """
        resolved_args = []
        for arg in args:
            if isinstance(arg, str):
                # 检查是否是相对路径（不是以 - 开头的选项）
                if not arg.startswith("-") and not arg.startswith("/"):
                    potential_path = (project_dir / arg).resolve()
                    # Prevent path traversal: resolved path must be under project_dir
                    if str(potential_path).startswith(str(project_dir.resolve())):
                        if potential_path.exists() or arg.endswith(".py"):
                            arg = str(potential_path)
                    else:
                        logger.warning(
                            f"[ConfigLoader] Blocked path traversal in tool arg: {arg}"
                        )
            resolved_args.append(arg)
        return resolved_args

    def load_agent_config(self, name: str) -> dict:
        """
        加载 Agent 配置。

        Args:
            name: Agent 配置名称

        Returns:
            Agent 配置字典

        Raises:
            ValueError: 如果找不到 Agent 配置
        """
        agent_path = CONFIG_DIR / f"{name}.yaml"
        if agent_path.exists():
            logger.debug(f"Loading agent '{name}' from {agent_path}")
            with open(agent_path, encoding="utf-8") as f:
                return yaml.safe_load(f)

        raise ValueError(f"Agent config not found: {name}")

    def get_skill_injector(self) -> Any | None:
        """
        获取 Skill 注入器。

        加载顺序：
        1. 框架内置 skills (config/skills/)
        2. 项目目录 skills (project_dir/config/skills/) — 合并到同一个 matcher

        Returns:
            SkillInjector 实例，如果 skills 目录存在的话
        """
        if self._skill_injector_initialized:
            return self._skill_injector

        self._skill_injector_initialized = True

        try:
            from mem_deep_research_core.skills import SkillInjector, SkillMatcher

            matcher = None

            # 1. 加载框架内置 skills
            framework_skills_dir = CONFIG_DIR / "skills"
            if framework_skills_dir.exists() and (framework_skills_dir / "definitions").exists():
                matcher = SkillMatcher(framework_skills_dir)
                logger.info(
                    f"Loaded {len(matcher.skills)} framework skills from {framework_skills_dir}"
                )

            # 2. 加载项目目录 skills（合并）
            if self._project_dir is not None:
                project_skills_dir = self._project_dir / "config" / "skills"
                if project_skills_dir.exists() and (project_skills_dir / "definitions").exists():
                    if matcher is None:
                        matcher = SkillMatcher(project_skills_dir)
                    else:
                        project_matcher = SkillMatcher(project_skills_dir)
                        matcher.skills.update(project_matcher.skills)
                    logger.info(
                        f"Loaded project skills from {project_skills_dir}, total: {len(matcher.skills)}"
                    )

            if matcher and matcher.skills:
                self._skill_injector = SkillInjector(matcher)
                return self._skill_injector

        except Exception as e:
            logger.warning(f"Failed to load skill injector: {e}")

        return None

    def get_llm_skill_selector(self, cfg) -> Any | None:
        """
        获取 LLM Skill 选择器。

        需要已初始化的 skill_injector（用于获取 matcher）和有效的 API key。

        Args:
            cfg: Agent 配置（DictConfig），需要包含 main_agent.openai_api_key
                 和 main_agent.skill_selection 配置

        Returns:
            LLMSkillSelector 实例，如果条件不满足则返回 None
        """
        if self._llm_skill_selector_initialized:
            return self._llm_skill_selector

        self._llm_skill_selector_initialized = True

        try:
            import os

            from mem_deep_research_core.skills import LLMSkillSelector

            # 检查 skill_selection 配置
            skill_selection_cfg = cfg.main_agent.get("skill_selection", {})
            if not skill_selection_cfg.get("enabled", True):
                logger.info("[ConfigLoader] LLM skill selection is disabled")
                return None

            # 获取 API key
            api_key = cfg.main_agent.get("openai_api_key")
            if not api_key:
                logger.info("[ConfigLoader] No openai_api_key, LLM skill selector not available")
                return None

            # 获取已初始化的 skill injector 来取 matcher
            injector = self.get_skill_injector()
            if not injector:
                logger.info(
                    "[ConfigLoader] No skill injector available, LLM skill selector not created"
                )
                return None

            base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
            model = skill_selection_cfg.get("model", "gpt-4o-mini")
            max_skills = skill_selection_cfg.get("max_skills", 3)
            fallback_to_rules = skill_selection_cfg.get("fallback_to_rules", True)

            self._llm_skill_selector = LLMSkillSelector(
                matcher=injector.matcher,
                api_key=api_key,
                base_url=base_url,
                model=model,
                max_skills=max_skills,
                fallback_to_rules=fallback_to_rules,
            )
            logger.info(
                f"[ConfigLoader] LLM skill selector initialized (model={model}, "
                f"max_skills={max_skills}, fallback={fallback_to_rules})"
            )
            return self._llm_skill_selector

        except Exception as e:
            logger.warning(f"Failed to create LLM skill selector: {e}")
            return None

    def get_inline_skill_selector(self, cfg, chinese: bool = False) -> Any | None:
        """
        获取 Inline Skill 选择器。

        Args:
            cfg: Agent 配置
            chinese: 是否使用中文提示

        Returns:
            InlineSkillSelector 实例，如果条件不满足则返回 None
        """
        try:
            skill_selection_cfg = cfg.main_agent.get("skill_selection", {})
            if not skill_selection_cfg.get("enabled", True):
                return None
            if skill_selection_cfg.get("method", "rules") != "inline":
                return None

            injector = self.get_skill_injector()
            if not injector:
                return None

            from mem_deep_research_core.skills import InlineSkillSelector

            progressive = skill_selection_cfg.get("progressive", True)
            selector = InlineSkillSelector(
                matcher=injector.matcher,
                chinese=chinese,
                progressive=progressive,
            )
            logger.info(
                f"[ConfigLoader] Inline skill selector initialized (progressive={progressive})"
            )
            return selector

        except Exception as e:
            logger.warning(f"Failed to create inline skill selector: {e}")
            return None

    def get_skills_dir(self) -> Path:
        """获取 Skills 目录路径"""
        return CONFIG_DIR / "skills"

    def get_config_dir(self) -> Path:
        """获取 config 目录路径"""
        return CONFIG_DIR

    def get_prompt_templates_dir(self) -> Path:
        """获取 prompt templates 目录路径"""
        from mem_deep_research_core.prompts import BUILTIN_TEMPLATES_DIR

        return BUILTIN_TEMPLATES_DIR

    def list_prompt_templates(self) -> list[str]:
        """列出所有可用的 prompt 模板"""
        templates_dir = self.get_prompt_templates_dir()
        if not templates_dir.exists():
            return []
        return [f.stem for f in templates_dir.glob("*.md")]


# 全局实例
config_loader = ConfigLoader()

# 保持向后兼容的别名
external_loader = config_loader


# ============================================================
# 通用配置工具函数
# ============================================================


def load_env_file(env_file: Path) -> None:
    """加载 .env 文件中的键值对到环境变量

    仅在环境变量未设置时生效（os.environ.setdefault）。

    Args:
        env_file: .env 文件路径
    """
    import os

    if not env_file.exists():
        return
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def load_yaml_config(config_path: Path) -> Any:
    """加载 YAML 配置文件并解析环境变量插值

    如果环境变量缺失导致解析失败，会回退到不解析模式。

    Args:
        config_path: YAML 配置文件路径

    Returns:
        解析后的 OmegaConf DictConfig

    Raises:
        FileNotFoundError: 配置文件不存在
    """
    from omegaconf import OmegaConf

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = OmegaConf.load(config_path)
    try:
        cfg = OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        cfg = OmegaConf.to_container(cfg, resolve=False)
    cfg = OmegaConf.create(cfg)
    return cfg
