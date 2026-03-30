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

from mem_deep_research_core.core.hooks import HookContext
from mem_deep_research_core.core.hooks import hooks as _default_hooks

logger = logging.getLogger("mem_deep_research")

# URL 正则（简洁版，覆盖常见 http/https URL）
_URL_PATTERN = re.compile(
    r'https?://[^\s<>"\')\]]+',
    re.IGNORECASE,
)

# 文件引用正则：@path/to/file 或 @"path with spaces"
_FILE_REF_PATTERN = re.compile(
    r'@"([^"]+)"|@(\S+\.\w{1,10})',
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
    """

    def __init__(
        self,
        hooks=None,
        enable_url_extraction: bool = True,
        enable_file_refs: bool = True,
    ):
        self.hooks = hooks or _default_hooks
        self.enable_url_extraction = enable_url_extraction
        self.enable_file_refs = enable_file_refs

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
                        result.attachments.append({
                            "type": "file",
                            "path": ref,
                            "content": content,
                        })
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

    @staticmethod
    def _read_file(path: str) -> str | None:
        """尝试读取文件内容"""
        import os

        if not os.path.isfile(path):
            return None
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                content = f.read()
            # 限制文件内容大小
            if len(content) > 50000:
                content = content[:50000] + f"\n... [truncated, total {len(content)} chars]"
            return content
        except Exception as e:
            logger.debug(f"[InputCompiler] Failed to read file {path}: {e}")
            return None
