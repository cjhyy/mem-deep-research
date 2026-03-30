"""
消息历史工具函数

从 Orchestrator 提取的纯函数，用于消息历史操作。
无状态、无副作用（除了直接修改传入的 message_history 列表）。
"""

import hashlib
import logging

from mem_deep_research_core.core.constants import (
    FALLBACK_LOOP_TERMINATED,
    RECENT_TOOL_LOOKBACK,
    SYSTEM_MESSAGE_KEYWORDS,
)

logger = logging.getLogger("mem_deep_research")


def extract_recent_tool_names(message_history: list, lookback: int = RECENT_TOOL_LOOKBACK) -> list:
    """从最近消息中提取 tool_use 的 name 列表"""
    names = []
    for msg in message_history[-lookback:]:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    name = block.get("name")
                    if name and name not in names:
                        names.append(name)
    return names


def deduplicate_trailing_messages(message_history: list) -> int:
    """移除消息历史末尾重复的 assistant 响应，保留第一次出现。

    当循环检测终止时，message_history 可能包含多轮相同的 assistant 响应，
    这会导致摘要 LLM 困惑或生成空内容。此方法从末尾向前扫描，
    移除连续重复的 assistant 消息（基于文本内容 hash），
    并在末尾追加一条说明，引导摘要 LLM 基于已有信息作答。

    Returns:
        int: 移除的消息数量
    """
    if len(message_history) < 4:
        return 0

    def _text_hash(msg: dict) -> int:
        content = msg.get("content", "")
        if isinstance(content, list):
            texts = [
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            text = "".join(texts)
        elif isinstance(content, str):
            text = content
        else:
            text = str(content)
        if not text:
            return 0
        return int(hashlib.md5(text[:500].encode("utf-8", errors="replace")).hexdigest(), 16)

    # 从末尾收集连续 assistant 消息的 hash
    i = len(message_history) - 1
    tail_hashes = []
    while i >= 0 and message_history[i].get("role") == "assistant":
        tail_hashes.append((i, _text_hash(message_history[i])))
        i -= 1
        # 跳过中间的 user 消息（如 INJECT_HINT）
        if i >= 0 and message_history[i].get("role") == "user":
            content_str = str(message_history[i].get("content", ""))
            if any(kw in content_str for kw in SYSTEM_MESSAGE_KEYWORDS):
                i -= 1

    if len(tail_hashes) < 2:
        return 0

    # 找出重复 hash 的索引，保留最早出现的（最低索引 = 最老的消息）
    seen_hashes: dict[int, int] = {}  # hash -> first index
    indices_to_remove = []
    for idx, h in sorted(tail_hashes, key=lambda x: x[0]):  # 按索引从小到大
        if h in seen_hashes:
            indices_to_remove.append(idx)  # 后续重复的标记删除
        else:
            seen_hashes[h] = idx  # 记录首次出现

    if not indices_to_remove:
        return 0

    # 同时移除紧跟在被删 assistant 消息后面的 INJECT_HINT user 消息
    all_remove = set(indices_to_remove)
    for idx in indices_to_remove:
        # 检查 idx+1 和 idx-1 是否为注入的指令消息
        for neighbor in (idx + 1, idx - 1):
            if 0 <= neighbor < len(message_history) and neighbor not in all_remove:
                msg = message_history[neighbor]
                if msg.get("role") == "user":
                    content_str = str(msg.get("content", ""))
                    if any(kw in content_str for kw in SYSTEM_MESSAGE_KEYWORDS):
                        all_remove.add(neighbor)

    # 按索引从大到小移除
    for idx in sorted(all_remove, reverse=True):
        if idx < len(message_history):
            message_history.pop(idx)

    removed = len(all_remove)
    if removed > 0:
        # 追加引导消息，帮助摘要 LLM 生成有效输出
        message_history.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": FALLBACK_LOOP_TERMINATED,
                    }
                ],
            }
        )
        logger.info(
            f"[DEDUP] Removed {removed} duplicate/injected messages from history tail, "
            f"history now {len(message_history)} messages"
        )

    return removed
