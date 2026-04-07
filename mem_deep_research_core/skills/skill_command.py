"""
SkillCommand — 统一 Skill 数据模型

兼容两种格式:
1. Claude Code 格式: {name}/SKILL.md (frontmatter + markdown)
2. 遗留格式: config/skills/definitions/*.md (自定义 frontmatter)

核心能力:
- 延迟渲染: $ARGUMENTS/$0/$name 替换 + ${CLAUDE_SKILL_DIR} + !`cmd` 动态内容
- Budget-aware catalog: 截断描述到指定字符数
- 条件激活: paths glob 匹配
"""

import asyncio
import fnmatch
import logging
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

logger = logging.getLogger("mem_deep_research")

def _glob_match(filepath: str, pattern: str) -> bool:
    """Gitignore-style glob match. Supports ** for recursive directory matching."""
    # Tokenize then convert to regex to avoid replacement collisions
    i = 0
    regex_parts = []
    while i < len(pattern):
        if pattern[i:].startswith("**/"):
            regex_parts.append("(?:.+/)*")  # zero or more dir segments
            i += 3
        elif pattern[i:].startswith("**"):
            regex_parts.append(".*")  # match anything
            i += 2
        elif pattern[i] == "*":
            regex_parts.append("[^/]*")  # match within single segment
            i += 1
        elif pattern[i] == "?":
            regex_parts.append("[^/]")
            i += 1
        elif pattern[i] in r"\.+^${}()|[]":
            regex_parts.append(re.escape(pattern[i]))
            i += 1
        else:
            regex_parts.append(re.escape(pattern[i]))
            i += 1
    regex_str = "".join(regex_parts)
    return bool(re.match(f"^{regex_str}$", filepath))


# --- Frontmatter parsing ---

_FRONTMATTER_PATTERN = re.compile(r"^---\s*\n([\s\S]*?)---\s*\n?", re.MULTILINE)

# --- Argument substitution patterns (matches Claude Code argumentSubstitution.ts) ---

# $ARGUMENTS[0], $ARGUMENTS[1], ...
_INDEXED_ARG_PATTERN = re.compile(r"\$ARGUMENTS\[(\d+)\]")
# $0, $1, ... (not followed by word chars)
_SHORTHAND_ARG_PATTERN = re.compile(r"\$(\d+)(?!\w)")
# $ARGUMENTS (full string)
_FULL_ARG_PATTERN = re.compile(r"\$ARGUMENTS(?!\[)")

# Dynamic content: !`command`
_DYNAMIC_CMD_PATTERN = re.compile(r"!`([^`]+)`")

# ${CLAUDE_SKILL_DIR}, ${CLAUDE_SESSION_ID}
_TEMPLATE_VAR_PATTERN = re.compile(r"\$\{(\w+)\}")


def parse_frontmatter(markdown: str) -> tuple[dict[str, Any], str]:
    """解析 YAML frontmatter，返回 (frontmatter_dict, markdown_body)"""
    match = _FRONTMATTER_PATTERN.match(markdown)
    if not match:
        return {}, markdown

    frontmatter_text = match.group(1)
    content = markdown[match.end() :]

    try:
        fm = yaml.safe_load(frontmatter_text)
    except yaml.YAMLError:
        # Retry: quote values with special YAML characters
        lines = []
        for line in frontmatter_text.splitlines():
            if ":" in line and not line.strip().startswith("#"):
                key, _, val = line.partition(":")
                val = val.strip()
                if val and any(c in val for c in "{}[]|>*&!#%@`"):
                    val = f'"{val}"'
                lines.append(f"{key}: {val}")
            else:
                lines.append(line)
        try:
            fm = yaml.safe_load("\n".join(lines))
        except yaml.YAMLError:
            logger.warning("[SkillCommand] Failed to parse frontmatter, using empty")
            fm = {}

    if not isinstance(fm, dict):
        fm = {}
    return fm, content


def _parse_allowed_tools(value: Any) -> list[str]:
    """Parse allowed-tools from string or list."""
    if not value:
        return []
    if isinstance(value, list):
        return [str(t).strip() for t in value if str(t).strip()]
    return [t.strip() for t in str(value).split() if t.strip()]


def _parse_arguments(value: Any) -> list[str]:
    """Parse argument names from string or list."""
    if not value:
        return []
    if isinstance(value, list):
        names = [str(a).strip() for a in value if str(a).strip()]
    else:
        names = str(value).split()
    # Filter out numeric-only names (conflict with $0, $1)
    return [n for n in names if n and not n.isdigit()]


def _parse_paths(value: Any) -> list[str]:
    """Parse paths field (comma-separated string or list)."""
    if not value:
        return []
    if isinstance(value, list):
        raw = [str(p).strip() for p in value]
    else:
        # Split by comma, respecting braces
        raw = [p.strip() for p in str(value).split(",")]

    patterns = []
    for p in raw:
        if not p:
            continue
        # Normalize trailing /**
        if p.endswith("/**"):
            p = p[:-3]
        if p and p != "**":
            patterns.append(p)
    return patterns


def _parse_bool(value: Any, default: bool = False) -> bool:
    """Parse boolean from frontmatter (only true/True/"true" → True)."""
    if value is True or value == "true":
        return True
    if value is False or value == "false" or value is None:
        return default
    return default


def _coerce_description(value: Any) -> str:
    """Coerce description to string, handle non-scalar gracefully."""
    if value is None:
        return ""
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, str):
        return value.strip()
    return ""


def _extract_description_from_markdown(content: str) -> str:
    """Extract first meaningful paragraph from markdown as fallback description."""
    for line in content.splitlines():
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("```"):
            return line[:250]
    return ""


# --- Argument substitution ---


def substitute_arguments(
    content: str,
    args: str,
    argument_names: list[str] | None = None,
    append_if_no_placeholder: bool = True,
) -> str:
    """Substitute argument placeholders in skill content.

    Order (matches Claude Code):
    1. Named args: $name (from argument_names list)
    2. Indexed: $ARGUMENTS[0], $ARGUMENTS[1]
    3. Shorthand: $0, $1
    4. Full: $ARGUMENTS
    5. Append: if no placeholders found and args non-empty
    """
    if not args:
        return content

    try:
        parsed_args = shlex.split(args)
    except ValueError:
        parsed_args = args.split()

    has_placeholder = False
    result = content

    # 1. Named arguments
    if argument_names:
        for i, name in enumerate(argument_names):
            pattern = re.compile(rf"\${re.escape(name)}(?![\[\w])")
            value = parsed_args[i] if i < len(parsed_args) else ""
            if pattern.search(result):
                has_placeholder = True
                result = pattern.sub(value, result)

    # 2. Indexed: $ARGUMENTS[N]
    def replace_indexed(m):
        nonlocal has_placeholder
        has_placeholder = True
        idx = int(m.group(1))
        return parsed_args[idx] if idx < len(parsed_args) else ""

    result = _INDEXED_ARG_PATTERN.sub(replace_indexed, result)

    # 3. Shorthand: $N
    def replace_shorthand(m):
        nonlocal has_placeholder
        has_placeholder = True
        idx = int(m.group(1))
        return parsed_args[idx] if idx < len(parsed_args) else ""

    result = _SHORTHAND_ARG_PATTERN.sub(replace_shorthand, result)

    # 4. Full: $ARGUMENTS
    if _FULL_ARG_PATTERN.search(result):
        has_placeholder = True
        result = _FULL_ARG_PATTERN.sub(args, result)

    # 5. Append if no placeholders found
    if not has_placeholder and append_if_no_placeholder and args.strip():
        result += f"\n\nARGUMENTS: {args}"

    return result


# --- Dynamic content execution ---


async def _execute_dynamic_commands(content: str, shell: str = "bash") -> str:
    """Execute !`cmd` patterns and replace with stdout."""

    async def run_cmd(match):
        cmd = match.group(1)
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            output = stdout.decode("utf-8", errors="replace").strip()
            if proc.returncode != 0 and not output:
                err = stderr.decode("utf-8", errors="replace").strip()
                return f"[command failed: {err[:200]}]"
            return output
        except TimeoutError:
            return f"[command timeout: {cmd[:100]}]"
        except Exception as e:
            return f"[command error: {e}]"

    # Find all matches and execute
    matches = list(_DYNAMIC_CMD_PATTERN.finditer(content))
    if not matches:
        return content

    # Execute all commands concurrently
    replacements = await asyncio.gather(*[run_cmd(m) for m in matches])

    # Replace in reverse order to preserve positions
    result = content
    for match, replacement in zip(reversed(matches), reversed(replacements), strict=True):
        result = result[: match.start()] + replacement + result[match.end() :]

    return result


# --- SkillCommand ---


@dataclass
class SkillCommand:
    """统一 Skill 数据模型，兼容 Claude Code 和遗留格式"""

    # Identity
    name: str
    description: str
    when_to_use: str = ""
    source_path: Path | None = None
    skill_dir: Path | None = None

    # Claude Code fields
    allowed_tools: list[str] = field(default_factory=list)
    context_mode: Literal["inline", "fork"] = "inline"
    agent: str | None = None
    user_invocable: bool = True
    disable_model_invocation: bool = False
    argument_names: list[str] = field(default_factory=list)
    argument_hint: str = ""
    model: str | None = None
    effort: str | None = None
    paths: list[str] = field(default_factory=list)
    hooks: dict = field(default_factory=dict)
    shell: str = "bash"

    # Legacy compat
    skill_type: str = "knowledge"
    triggers: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    priority: int = 0

    # Internal
    raw_content: str = ""
    format: Literal["legacy", "claude_code"] = "legacy"

    async def get_prompt(self, arguments: str = "") -> str:
        """延迟渲染 skill 内容。

        执行顺序:
        1. 前缀 skill 目录路径
        2. $ARGUMENTS/$0/$name 替换
        3. ${CLAUDE_SKILL_DIR} 替换
        4. !`cmd` 动态内容执行
        """
        content = self.raw_content

        # 1. Prepend base directory
        if self.skill_dir:
            content = f"Base directory for this skill: {self.skill_dir}\n\n{content}"

        # 2. Argument substitution
        if arguments:
            content = substitute_arguments(
                content, arguments, self.argument_names, append_if_no_placeholder=True
            )

        # 3. Template variables
        if self.skill_dir:
            content = content.replace("${CLAUDE_SKILL_DIR}", str(self.skill_dir))

        # 4. Dynamic content execution
        if _DYNAMIC_CMD_PATTERN.search(content):
            content = await _execute_dynamic_commands(content, self.shell)

        return content

    def get_catalog_entry(self, max_chars: int = 250) -> str:
        """返回用于 catalog 的截断描述"""
        desc = self.description
        if self.when_to_use:
            desc = f"{desc} - {self.when_to_use}"
        if len(desc) > max_chars:
            return desc[: max_chars - 1] + "\u2026"
        return desc

    def matches_paths(self, touched_files: list[str]) -> bool:
        """检查是否有文件匹配 paths glob 模式。paths 为空时总是匹配。

        支持 ** 递归匹配（gitignore 风格：src/** 匹配 src/ 下所有文件）。
        """
        if not self.paths:
            return True
        for pattern in self.paths:
            for f in touched_files:
                if _glob_match(f, pattern):
                    return True
        return False

    @classmethod
    def from_claude_code(cls, skill_dir: Path) -> "SkillCommand":
        """从 Claude Code 格式目录加载 SKILL.md"""
        skill_file = skill_dir / "SKILL.md"
        raw = skill_file.read_text(encoding="utf-8")
        fm, content = parse_frontmatter(raw)

        name = str(fm.get("name", skill_dir.name))
        description = _coerce_description(fm.get("description"))
        if not description:
            description = _extract_description_from_markdown(content)
        if not description:
            description = f"Skill: {name}"

        context_val = fm.get("context")
        context_mode = "fork" if context_val == "fork" else "inline"

        return cls(
            name=name,
            description=description,
            when_to_use=_coerce_description(fm.get("when_to_use", "")),
            source_path=skill_file,
            skill_dir=skill_dir,
            allowed_tools=_parse_allowed_tools(fm.get("allowed-tools")),
            context_mode=context_mode,
            agent=fm.get("agent"),
            user_invocable=_parse_bool(fm.get("user-invocable"), default=True),
            disable_model_invocation=_parse_bool(fm.get("disable-model-invocation"), default=False),
            argument_names=_parse_arguments(fm.get("arguments")),
            argument_hint=str(fm.get("argument-hint", "")),
            model=fm.get("model"),
            effort=fm.get("effort"),
            paths=_parse_paths(fm.get("paths")),
            hooks=fm.get("hooks") if isinstance(fm.get("hooks"), dict) else {},
            shell=fm.get("shell", "bash") if fm.get("shell") in ("bash", "powershell") else "bash",
            skill_type=str(fm.get("type", "knowledge")),
            raw_content=content,
            format="claude_code",
        )

    @classmethod
    def from_legacy(cls, name: str, data: dict, source_path: Path) -> "SkillCommand":
        """从遗留格式 dict 转换"""
        triggers = data.get("triggers", {})
        meta = data.get("metadata", {})

        return cls(
            name=name,
            description=data.get("description", ""),
            when_to_use=data.get("when_to_use", ""),
            source_path=source_path,
            skill_dir=None,
            skill_type=data.get("type", "knowledge"),
            triggers=triggers,
            metadata=meta,
            priority=meta.get("priority", 0),
            raw_content=data.get("content", ""),
            format="legacy",
        )
