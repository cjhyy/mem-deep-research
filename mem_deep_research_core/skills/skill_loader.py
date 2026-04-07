"""
SkillLoader — 多源 Skill 扫描与加载

扫描顺序（后覆盖前，同名去重）:
1. 框架 config/skills/definitions/ (legacy 格式)
2. 项目 config/skills/definitions/ (legacy 格式)
3. 项目 .claude/skills/ (Claude Code 格式)
4. 用户 ~/.claude/skills/ (Claude Code 格式)
5. 额外指定目录 (Claude Code 格式)
"""

import logging
from pathlib import Path

from mem_deep_research_core.skills.skill_command import SkillCommand, parse_frontmatter

logger = logging.getLogger("mem_deep_research")


class SkillLoader:
    """多源 Skill 扫描与加载器"""

    def __init__(
        self,
        framework_dir: Path | None = None,
        project_dir: Path | None = None,
        extra_dirs: list[Path] | None = None,
        scan_user_home: bool = True,
    ):
        self._framework_dir = framework_dir
        self._project_dir = project_dir
        self._extra_dirs = extra_dirs or []
        self._scan_user_home = scan_user_home

    def load_all(self) -> dict[str, SkillCommand]:
        """扫描所有目录，加载两种格式，按优先级去重（后覆盖前）。"""
        skills: dict[str, SkillCommand] = {}

        for directory, format_type in self._build_search_order():
            if not directory.is_dir():
                continue
            try:
                if format_type == "legacy":
                    loaded = self._scan_legacy_dir(directory)
                else:
                    loaded = self._scan_claude_code_dir(directory)

                for sc in loaded:
                    if sc.name in skills:
                        logger.debug(f"[SkillLoader] Skill '{sc.name}' overridden by {directory}")
                    skills[sc.name] = sc

            except Exception as e:
                logger.warning(f"[SkillLoader] Error scanning {directory}: {e}")

        logger.info(
            f"[SkillLoader] Loaded {len(skills)} skills "
            f"({sum(1 for s in skills.values() if s.format == 'claude_code')} claude_code, "
            f"{sum(1 for s in skills.values() if s.format == 'legacy')} legacy)"
        )
        return skills

    def _build_search_order(self) -> list[tuple[Path, str]]:
        """构建扫描顺序列表: [(dir, format_type), ...]"""
        order: list[tuple[Path, str]] = []

        # 1. 框架内置 (legacy)
        if self._framework_dir:
            legacy_dir = self._framework_dir / "config" / "skills" / "definitions"
            order.append((legacy_dir, "legacy"))

        # 2. 项目级 (legacy)
        if self._project_dir:
            project_legacy = self._project_dir / "config" / "skills" / "definitions"
            order.append((project_legacy, "legacy"))

        # 3. 项目级 .claude/skills/ (Claude Code)
        if self._project_dir:
            project_cc = self._project_dir / ".claude" / "skills"
            order.append((project_cc, "claude_code"))

        # 4. 用户级 ~/.claude/skills/ (Claude Code)
        if self._scan_user_home:
            user_cc = Path.home() / ".claude" / "skills"
            order.append((user_cc, "claude_code"))

        # 5. 额外目录 (Claude Code)
        for d in self._extra_dirs:
            order.append((Path(d), "claude_code"))

        return order

    def _scan_claude_code_dir(self, base_dir: Path) -> list[SkillCommand]:
        """扫描 Claude Code 格式: {name}/SKILL.md"""
        skills = []
        try:
            entries = sorted(base_dir.iterdir())
        except PermissionError:
            logger.warning(f"[SkillLoader] Permission denied: {base_dir}")
            return []

        for entry in entries:
            if not entry.is_dir():
                continue
            skill_file = entry / "SKILL.md"
            if not skill_file.is_file():
                continue
            try:
                sc = SkillCommand.from_claude_code(entry)
                skills.append(sc)
                logger.debug(f"[SkillLoader] Loaded Claude Code skill: {sc.name} from {entry}")
            except Exception as e:
                logger.warning(f"[SkillLoader] Failed to load skill from {entry}: {e}")

        return skills

    def _scan_legacy_dir(self, definitions_dir: Path) -> list[SkillCommand]:
        """扫描遗留格式: *.md 文件"""
        skills = []
        try:
            md_files = sorted(definitions_dir.glob("*.md"))
        except PermissionError:
            logger.warning(f"[SkillLoader] Permission denied: {definitions_dir}")
            return []

        for md_file in md_files:
            try:
                raw = md_file.read_text(encoding="utf-8")
                fm, content = parse_frontmatter(raw)

                name = fm.get("name", md_file.stem)
                data = {
                    "name": name,
                    "description": fm.get("description", ""),
                    "when_to_use": fm.get("when_to_use", ""),
                    "type": fm.get("type", "knowledge"),
                    "triggers": fm.get("triggers", {}),
                    "metadata": fm.get("metadata", {}),
                    "content": content,
                }
                sc = SkillCommand.from_legacy(name, data, md_file)
                skills.append(sc)
                logger.debug(f"[SkillLoader] Loaded legacy skill: {sc.name} from {md_file}")
            except Exception as e:
                logger.warning(f"[SkillLoader] Failed to load skill from {md_file}: {e}")

        return skills
