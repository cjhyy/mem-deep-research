"""
输入编译链

用户的原始 query 经过编译处理后再进入 Agent 循环。
参考 Claude Code 的输入编译链（ihz → BU4 → AU8）。

处理流程：
1. URL 检测 → 提取为 attachments
2. 文件引用 (@file) → 展开内容
3. Hook: on_query_compile → 允许业务侧修改 query
4. 语言检测预处理

Usage:
    compiler = InputCompiler(hooks=hooks)
    result = await compiler.compile(query, context)
    # result.query — 处理后的 query
    # result.attachments — 提取的附件
    # result.metadata — 编译元数据
"""

import logging
import re
from dataclasses import dataclass, field

from mem_deep_research_core.core.hooks import HookContext, HookRegistry

logger = logging.getLogger("mem_deep_research")

# URL 正则（简洁版，覆盖常见 http/https URL）
_URL_PATTERN = re.compile(
    r'https?://[^\s<>"\')\]]+',
    re.IGNORECASE,
)

# 文件引用正则：@path/to/file 或 @"path with spaces"
_FILE_REF_PATTERN = re.compile(
    r'@"([^"]+)"|@(\S+\.\w+)',
)


@dataclass
class CompileResult:
    """编译结果"""

    query: str  # 处理后的 query
    original_query: str  # 原始 query
    attachments: list[dict] = field(default_factory=list)  # 附件列表
    extracted_urls: list[str] = field(default_factory=list)  # 提取的 URL
    file_refs: list[str] = field(default_factory=list)  # 文件引用
    metadata: dict = field(default_factory=dict)  # 编译元数据


class InputCompiler:
    """输入编译器

    对用户 query 做轻量预处理，提取 URL 和文件引用等。

    Args:
        hooks: HookRegistry 实例
        enable_url_extraction: 是否启用 URL 提取
        enable_file_refs: 是否启用文件引用展开
        file_ref_allowed_dirs: Allowlist of directories for @file refs.
            Empty list = unrestricted (backward compatible).
            When set, only files under these directories can be read.
    """

    # Filenames always blocked regardless of allowlist
    _SENSITIVE_PATTERNS = {
        ".env",
        ".env.local",
        ".env.production",
        ".env.staging",
        "credentials.json",
        "id_rsa",
        "id_dsa",
        "id_ed25519",
        "id_ecdsa",
        "id_ed448",
    }

    def __init__(
        self,
        *,
        hooks: HookRegistry,
        enable_url_extraction: bool = True,
        enable_file_refs: bool = True,
        file_ref_allowed_dirs: list[str] | None = None,
    ):
        self.hooks = hooks
        self.enable_url_extraction = enable_url_extraction
        self.enable_file_refs = enable_file_refs
        self._file_ref_allowed_dirs = file_ref_allowed_dirs or []

    def compile(
        self,
        query: str,
        context: dict | None = None,
    ) -> CompileResult:
        """编译用户输入

        Args:
            query: 原始用户输入
            context: 用户上下文

        Returns:
            CompileResult
        """
        result = CompileResult(
            query=query,
            original_query=query,
        )

        # 1. 提取 URL
        if self.enable_url_extraction:
            urls = _URL_PATTERN.findall(query)
            if urls:
                result.extracted_urls = list(dict.fromkeys(urls))  # 去重保序
                result.metadata["url_count"] = len(result.extracted_urls)
                logger.debug(f"[InputCompiler] Extracted {len(urls)} URLs from query")

        # 2. 提取文件引用
        if self.enable_file_refs:
            file_refs = []
            for match in _FILE_REF_PATTERN.finditer(query):
                ref = match.group(1) or match.group(2)
                if ref:
                    file_refs.append(ref)
            if file_refs:
                result.file_refs = file_refs
                result.metadata["file_ref_count"] = len(file_refs)
                logger.debug(f"[InputCompiler] Found {len(file_refs)} file references")

                # 尝试展开文件内容
                for ref in file_refs:
                    content = self._read_file(ref)
                    if content is not None:
                        result.attachments.append(
                            {
                                "type": "file",
                                "path": ref,
                                "content": content,
                            }
                        )
                        # 替换 query 中的文件引用为简短标记
                        pattern = f'@"{ref}"' if " " in ref else f"@{ref}"
                        replacement = f"[attached: {ref}]"
                        result.query = result.query.replace(pattern, replacement)

        # 3. Hook: on_query_compile — 允许业务侧修改
        if self.hooks.has_hooks("on_query_compile"):
            hook_result = self.hooks.call(
                "on_query_compile",
                HookContext(
                    hook_name="on_query_compile",
                    query=result.query,
                    context=context or {},
                    extra={
                        "urls": result.extracted_urls,
                        "file_refs": result.file_refs,
                        "attachments": result.attachments,
                    },
                ),
            )
            if isinstance(hook_result, str):
                result.query = hook_result
            elif isinstance(hook_result, dict):
                # Hook 可以返回 {"query": "...", "attachments": [...]}
                if "query" in hook_result:
                    result.query = hook_result["query"]
                if "attachments" in hook_result:
                    result.attachments.extend(hook_result["attachments"])

        return result

    def _read_file(self, path: str) -> str | None:
        """尝试读取文件内容（带安全检查）"""
        import os

        resolved = os.path.realpath(path)
        basename = os.path.basename(resolved)

        # Block known sensitive files
        if basename in self._SENSITIVE_PATTERNS or basename.startswith(".env"):
            logger.warning(f"[InputCompiler] Blocked sensitive file reference: {path}")
            return None

        # Block SSH private keys (exact match already in _SENSITIVE_PATTERNS,
        # this catches variants like id_rsa_work without blocking id_mapping.csv)
        _SSH_KEY_PREFIXES = ("id_rsa", "id_dsa", "id_ed25519", "id_ecdsa", "id_ed448")
        if basename.startswith(_SSH_KEY_PREFIXES) and not basename.endswith(".pub"):
            logger.warning(f"[InputCompiler] Blocked private key reference: {path}")
            return None

        # Enforce allowlist if configured
        if self._file_ref_allowed_dirs:
            allowed = False
            for allowed_dir in self._file_ref_allowed_dirs:
                allowed_resolved = os.path.realpath(allowed_dir)
                if resolved.startswith(allowed_resolved + os.sep) or resolved == allowed_resolved:
                    allowed = True
                    break
            if not allowed:
                logger.warning(
                    f"[InputCompiler] File ref '{path}' outside allowed directories, skipping"
                )
                return None

        if not os.path.isfile(resolved):
            return None
        try:
            with open(resolved, encoding="utf-8", errors="replace") as f:
                content = f.read()
            # 限制文件内容大小
            if len(content) > 50000:
                content = content[:50000] + f"\n... [truncated, total {len(content)} chars]"
            return content
        except Exception as e:
            logger.debug(f"[InputCompiler] Failed to read file {path}: {e}")
            return None
