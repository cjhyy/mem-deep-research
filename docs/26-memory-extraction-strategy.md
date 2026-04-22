# Memory Extraction Strategy 设计

> 状态基线：2026-04-22
> 文档定位：v1.4.0 "大工具结果细节保鲜" 的可插拔 strategy 层最终设计
> 配套阅读：`docs/22-profile-boundary.md`（Runtime/Profile 边界）、`docs/25-profile-contract.md`（Profile 契约）

## 问题

长任务 + 大工具结果场景下，关键细节容易丢失：

- 工具返回 20KB 结果 → offload 到文件 → 后续压缩时，如果没提前"留痕"，细节不可恢复
- LLM 在回复里引用工具结果 → 如果没结构化存下来，只是自然语言，后续难以检索
- Context 压缩时细节被摘要掉 → summary 可能遗漏重要事实

当前框架用"`<evidence>` tag 提取"作为解法，但这只是**一种**。业界还有几种主流方案：

| 方案 | 机制 | 优点 | 缺点 |
|------|------|------|------|
| Evidence tag | prompt 引导 LLM 在回复里产 `<evidence>` tag → 框架抽取 | 零额外 LLM 成本、实时 | 依赖 prompt 遵循度 |
| Summarize-on-compact | context 压缩时 LLM 把旧消息压成 summary，细节留在 summary 里 | 通用、强模型效果好 | 额外 LLM 调用、触发才做 |
| Vector store retrieval | 结果进 vector store，需要时检索 | 无 context 压力 | 依赖 embedding、外部存储 |
| Structured citation | 工具返回就做结构化拆分 + citation ID | 细节和位置天然结构化 | 工具侧改动大 |
| Pagination + 二次查询 | 大结果分页返回，LLM 用 grep/read 二次查询 | 不占 context | 只适合文件型工具 |
| Runtime fact extraction | 工具结果回来后轻量 LLM 抽 facts 存储 | 不依赖 prompt 遵循 | 每次工具额外 LLM 成本 |

每种方案有 trade-off，没有最佳。**所以框架应该把"提取策略"做成可插拔层**，用户 / profile 按需选实现。

## 设计原则

1. **Strategies 归属 Profile 层**（单层心智）— Runtime 本身不持有 strategies 列表，所有 strategy 都在 `profile.extraction_strategies` 里
2. **Strategy 可组合**（多个 strategy 并行工作，不互斥）
3. **Strategy 必须是无状态或显式状态化**（以支持 HITL resume）
4. **Strategy 原则上只读**：接口允许返回修改后的 text，但 tag 清理 / 输出卫生由 runtime 统一做，不是 strategy 职责
5. **接口稳定，实现可替换**：用户自定义 strategy 不需要改框架

## 接口定义

### `MemoryExtractionStrategy` ABC

```python
from abc import ABC
from dataclasses import dataclass
from typing import Any


@dataclass
class ExtractionContext:
    """Strategy 方法可访问的 runtime 状态视图。"""
    turn_number: int
    task_description: str
    mode: str
    session_memory: Any
    context_manager: Any  # 完整实例暴露，strategy 自律使用
    llm_client: Any       # strategy 需要跑 LLM 时用（如 fact extraction）


class MemoryExtractionStrategy(ABC):
    """从 LLM 响应 / 工具结果 / context 压缩中抽取长期细节的策略。

    所有方法都有默认 no-op 实现，子类按需覆盖。每个 strategy 只需关心自己的触发点。

    Strategy 通过 session_memory / context_manager 的写 API 持久化抽取的细节。
    Strategy 不负责输出卫生（tag 清理），那由 runtime 在 strategy 链执行完后统一做。
    """

    name: str = "base"

    async def on_llm_response(
        self,
        assistant_text: str,
        ctx: ExtractionContext,
    ) -> str:
        """LLM 响应后触发。返回可能被修改的 text。

        典型用途：EvidenceTagStrategy 抽 <evidence> tag 存 session_memory。

        原则：strategy 只读取并存储结构化数据。Tag 清理由 runtime 在 strategy
        链执行完后统一做，strategy 不应清理 tag。返回原 text 即可。
        """
        return assistant_text

    async def on_tool_result(
        self,
        tool_name: str,
        tool_result: Any,
        ctx: ExtractionContext,
    ) -> None:
        """工具结果回来后触发（执行后、注入 message_history 前）。

        并发工具场景：**每个工具分别触发一次**（不做 batch）。需要 batch
        处理的 strategy 在内部维护 buffer，在 on_compact 等时机 flush。

        典型用途：FactExtractionStrategy 用轻量 LLM 从工具结果抽 facts。
        """
        return None

    async def on_compact(
        self,
        summary: str,
        up_to_turn: int,
        ctx: ExtractionContext,
    ) -> None:
        """Context 压缩完成后触发（LLMSummarize 产出 summary）。

        典型用途：SummaryEvidenceStrategy 从 summary 的 ## Evidence 段抽细节。
        """
        return None

    async def on_offload(
        self,
        ref: str,
        tool_name: str,
        original_content: str,
        ctx: ExtractionContext,
    ) -> None:
        """工具结果被 offload 到文件时触发。

        典型用途：用户自定义 strategy 把 offload 内容同步入外部存储
        （vector store / 全文索引 / 外部 blob store 等）。
        """
        return None

    # ========== Snapshot（HITL resume 支持）==========

    def snapshot(self) -> dict:
        """返回 strategy 内部状态，用于 checkpoint。默认空（无状态）。"""
        return {}

    def restore(self, state: dict) -> None:
        """从 snapshot 恢复内部状态。默认 no-op。"""
        return None
```

### 触发点说明

按当前框架 evidence / 细节抽取的实际发生位置，归纳出 4 个触发点：

| 触发点 | 当前框架位置 | 典型 strategy | 状态 |
|-------|------------|-------------|------|
| `on_llm_response` | `main_loop.py` `_extract_evidence_tags` + `_extract_offload_evidence` | EvidenceTagStrategy, OffloadEvidenceStrategy | Phase 2a |
| `on_compact` | `window_strategy.py:798` `_extract_evidence_from_summary` | SummaryEvidenceStrategy | Phase 2a |
| `on_tool_result` | 当前框架**无**，需加触发点 | FactExtractionStrategy | Phase 2b |
| `on_offload` | 当前框架**无**，需加触发点 | 用户自定义 vector store 等 | Phase 2c |

## Runtime 与 Profile 的职责分工

**核心决策**：Runtime 不持有 strategies 列表，全部 strategies 通过 `profile.extraction_strategies` 暴露。

```
Runtime（MainLoopRunner）
  │
  │ 触发点调用：await self.profile.run_strategies(hook_name, ...)
  │
  └── Profile
        └── extraction_strategies: list[MemoryExtractionStrategy]

              StandardProfile 默认: [OffloadEvidenceStrategy, SummaryEvidenceStrategy]
                                    (细节保鲜是 base profile 基础能力)

              DeepResearchProfile 默认: 继承 base + 追加 [EvidenceTagStrategy]
```

### 为什么 StandardProfile 默认带 Offload/Summary Evidence

"防止长任务细节丢失" 对所有 profile 都有价值，不是研究专属。StandardProfile 把这两个作为默认 strategies 体现"base profile 的实用基线"。

用户如果想**完全自定义**（例如只用 VectorStore，不用 tag 机制），覆盖 `extraction_strategies`：

```python
DeepResearch(
    profile="standard",
    profile_config={
        "extraction_strategies": [MyVectorStoreStrategy(pinecone_client)],  # 用户自定义
    },
)
```

这种"全覆盖"是 strategies 在 profile 层的自然结果，不需要"高级用法禁用开关"。

### Profile 基类的 `run_strategies` 方法

Profile 基类提供统一的 strategy 链调用方法，避免每个触发点重复遍历代码：

```python
class Profile(ABC):
    extraction_strategies: list[MemoryExtractionStrategy] = []

    async def run_strategies_on_llm_response(
        self, assistant_text: str, ctx: ExtractionContext,
    ) -> str:
        """按 list 顺序依次调用 strategy.on_llm_response，串联返回值。"""
        for strat in self.extraction_strategies:
            assistant_text = await strat.on_llm_response(assistant_text, ctx)
        return assistant_text

    async def run_strategies_on_tool_result(
        self, tool_name: str, tool_result: Any, ctx: ExtractionContext,
    ) -> None:
        for strat in self.extraction_strategies:
            await strat.on_tool_result(tool_name, tool_result, ctx)

    async def run_strategies_on_compact(
        self, summary: str, up_to_turn: int, ctx: ExtractionContext,
    ) -> None:
        for strat in self.extraction_strategies:
            await strat.on_compact(summary, up_to_turn, ctx)

    async def run_strategies_on_offload(
        self, ref: str, tool_name: str, original_content: str, ctx: ExtractionContext,
    ) -> None:
        for strat in self.extraction_strategies:
            await strat.on_offload(ref, tool_name, original_content, ctx)
```

### Strategy 调用顺序

**按 `extraction_strategies` list 的顺序依次调用**。用户配置顺序 = 执行顺序。

对于 `on_llm_response` 这类有返回值串联的方法，list 顺序影响下游 strategy 看到的 text。设计约定：**strategy 应视 text 为只读**（返回原值），如果非要修改，按 list 顺序明确谁先谁后。

### Runtime 层的 tag 清理

Strategy 链执行完后，Runtime 统一对 assistant_text 做 tag 清理（剥离 `<evidence>` / `<offload_evidence>` / `<next_skills>` 等 tag），保证最终输出干净。

```python
# main_loop.py 伪代码
assistant_text = await self.profile.run_strategies_on_llm_response(
    assistant_text, ext_ctx,
)
# Runtime 统一清理 tag（strategy 链不负责这步）
assistant_text = _strip_framework_tags(assistant_text)
```

这样分工：
- **Strategy 只读 + 抽取存储**
- **Runtime 负责输出卫生**

StandardProfile 下即使没任何 strategy 抽 `<evidence>` tag，tag 依然会被 runtime 清理，用户看到干净输出。

## 内置 Strategy

### Phase 2a 首批

#### `OffloadEvidenceStrategy`（StandardProfile 默认）

当前 `_extract_offload_evidence` 的逻辑封装。

```python
class OffloadEvidenceStrategy(MemoryExtractionStrategy):
    """绑定 LLM 产出的 <offload_evidence ref="..."> 到 offload registry。

    工作流：
    1. 主循环在 offload 发生时注入 [OFFLOAD PREP] sidecar
    2. LLM 下轮回复里产 <offload_evidence ref="abc.txt">关键行</offload_evidence>
    3. 本 strategy 抽取并绑定到 context_manager._offload_registry

    是大工具结果"细节保鲜"的核心机制，StandardProfile 默认启用。
    """

    name = "offload_evidence"

    async def on_llm_response(self, assistant_text, ctx):
        # 封装当前 _extract_offload_evidence 的逻辑
        ctx.context_manager.extract_and_bind_offload_evidence(assistant_text)
        return assistant_text
```

#### `SummaryEvidenceStrategy`（StandardProfile 默认）

当前 `window_strategy.py:_extract_evidence_from_summary` 的逻辑封装。

```python
class SummaryEvidenceStrategy(MemoryExtractionStrategy):
    """从 LLM 压缩 summary 的 ## Evidence 段抽取细节存入 session_memory。

    是"细节保鲜"的第二道防线：即使原消息被 compact 掉，关键事实
    仍以 EvidenceItem 形式留在 session_memory 里。
    """

    name = "summary_evidence"

    async def on_compact(self, summary, up_to_turn, ctx):
        # 封装当前 _extract_evidence_from_summary 的逻辑
        ctx.session_memory.add_evidence(EvidenceItem(...))
```

#### `EvidenceTagStrategy`（DeepResearchProfile 默认，追加在 base 之后）

当前 `_extract_evidence_tags` 的逻辑封装。依赖 prompt 引导 LLM 产出 `<evidence>` tag（研究场景 prompt 模板的默认指令）。

```python
class EvidenceTagStrategy(MemoryExtractionStrategy):
    """抽取 LLM 回复中的 <evidence> tag 存 session_memory。

    DeepResearchProfile 的 prompt 模板默认包含 "请用 <evidence> 标记引用" 指令；
    其他 profile 除非自行引导否则 LLM 不会产出此 tag，strategy 执行但抽不到（无副作用）。
    """

    name = "evidence_tag"

    async def on_llm_response(self, assistant_text, ctx):
        # 封装当前 _extract_evidence_tags 的逻辑
        # 只读取 + 写 session_memory，不清理 tag
        return assistant_text
```

### Phase 2b 第二批

#### `FactExtractionStrategy`（可选）

工具结果回来后用轻量 LLM 抽事实，不依赖主 LLM 产 tag。

```python
class FactExtractionStrategy(MemoryExtractionStrategy):
    """工具结果回来后用轻量 LLM 抽 facts 存 session_memory。

    适合主 LLM 不可控（弱模型 / 用户 prompt 自由度大）的场景，
    以轻量 LLM 的成本换 extraction 质量的确定性。
    """

    name = "fact_extraction"

    def __init__(
        self,
        extractor_llm_client,
        prompt_template: str = DEFAULT_FACT_EXTRACTION_PROMPT,
        max_facts_per_result: int = 5,
        min_result_size: int = 500,
    ):
        self.extractor = extractor_llm_client
        self.prompt_template = prompt_template
        self.max_facts = max_facts_per_result
        self.min_result_size = min_result_size

    async def on_tool_result(self, tool_name, tool_result, ctx):
        text = str(tool_result)
        if len(text) < self.min_result_size:
            return
        prompt = self.prompt_template.format(tool=tool_name, result=text[:10000])
        facts = await self.extractor.extract(prompt)
        for fact in facts[: self.max_facts]:
            ctx.session_memory.add_evidence(
                EvidenceItem(tool_name=tool_name, turn=ctx.turn_number, summary=fact)
            )
```

#### `SummarizeOnCompactStrategy`（可选）

显式表达"激进压缩 + 强 summary"模式（LangGraph / Mastra 风格）。

```python
class SummarizeOnCompactStrategy(MemoryExtractionStrategy):
    """Context 压缩时把整个 summary 作为 memory anchor 存 session_memory。

    和 SummaryEvidenceStrategy 互补：SummaryEvidence 抽 summary 里的 ## Evidence 段；
    本 strategy 把整段 summary 作为"记忆锚点"保存。适合 LangGraph / Mastra
    "memory is summary" 思路。
    """

    name = "summarize_on_compact"

    async def on_compact(self, summary, up_to_turn, ctx):
        ctx.session_memory.add_memory_anchor(
            turn=up_to_turn,
            summary=summary,
            anchor_type="compact_summary",
        )
```

### 用户自定义 Strategy 示例：Vector Store 接入

Vector store 接入**不作为内置 strategy**提供。原因：
- 每个 vector store（Pinecone / Weaviate / Chroma / Qdrant / PGVector / ...）client 接口不同，难以统一 duck-typed 抽象
- Chunk 策略、embedding model、namespace schema 都与具体部署强耦合
- 框架提供空壳 strategy 反而让用户误以为开箱即用，实际仍需大量定制

参考实现（用户自己写在项目里）：

```python
from mem_deep_research_core.memory_extraction import MemoryExtractionStrategy

class MyVectorStoreStrategy(MemoryExtractionStrategy):
    """Offload 时把原内容同步到 vector store（用户项目内实现）。

    配套需要：
    - 用户的 vector store client（含 embedding）
    - 用户注册一个检索工具，LLM 通过它查询相关片段
    """

    name = "my_vector_store"

    def __init__(self, client, embedder, chunk_fn):
        super().__init__()
        self.client = client
        self.embedder = embedder
        self.chunk_fn = chunk_fn
        self._stored: set[str] = set()

    async def on_offload(self, ref, tool_name, original_content, ctx):
        if ref in self._stored:
            return
        chunks = self.chunk_fn(original_content)
        vectors = await self.embedder.embed_batch([c.text for c in chunks])
        await self.client.upsert(
            namespace=ctx.task_description,
            items=[...],  # 按自己 store 的 API
        )
        self._stored.add(ref)

    def snapshot(self) -> dict:
        return {"stored": list(self._stored)}

    def restore(self, state: dict) -> None:
        self._stored = set(state.get("stored", []))
```

然后在项目里传入：

```python
from mem_deep_research_core.memory_extraction import register_strategy

register_strategy(MyVectorStoreStrategy)

dr = DeepResearch(
    profile="deep_research",
    profile_config={
        "extraction_strategies_extra": [
            MyVectorStoreStrategy(client=my_pinecone, embedder=openai_emb, chunk_fn=my_chunker),
        ],
    },
)
```

## 配置入口

### 代码方式

```python
from mem_deep_research_core.memory_extraction import (
    EvidenceTagStrategy,
    FactExtractionStrategy,
    SummarizeOnCompactStrategy,
)

# 追加到 deep_research 默认集
dr = DeepResearch(
    profile="deep_research",
    profile_config={
        "extraction_strategies_extra": [  # 追加，不覆盖
            FactExtractionStrategy(extractor_llm_client=haiku_client),
        ],
    },
)

# 完全覆盖 standard 的默认
dr = DeepResearch(
    profile="standard",
    profile_config={
        "extraction_strategies": [  # 完全覆盖
            FactExtractionStrategy(extractor_llm_client=haiku_client),
            SummarizeOnCompactStrategy(),
        ],
    },
)
```

两个配置 key：
- `extraction_strategies`：完全覆盖 profile 的默认 list
- `extraction_strategies_extra`：追加到 profile 的默认 list 后面

### YAML 方式

```yaml
main_agent:
  profile: deep_research
  profile_config:
    extraction_strategies_extra:
      - name: fact_extraction
        config:
          extractor_model: claude-haiku
          max_facts_per_result: 3
      - name: summarize_on_compact
```

Vector store / RAG 接入：用户自定义 strategy 后通过 `register_strategy` 注册即可在 YAML 里按名字引用。

## 新增目录结构

```
mem_deep_research_core/
├── memory_extraction/
│   ├── __init__.py              # Strategy registry（name → class）
│   ├── base.py                  # MemoryExtractionStrategy ABC + ExtractionContext
│   ├── evidence_tag.py          # EvidenceTagStrategy
│   ├── offload_evidence.py      # OffloadEvidenceStrategy
│   ├── summary_evidence.py      # SummaryEvidenceStrategy
│   ├── fact_extraction.py       # FactExtractionStrategy（Phase 2b）
│   ├── summarize_on_compact.py  # SummarizeOnCompactStrategy（Phase 2b）
(vector store 等外部存储接入由用户自定义 strategy 实现，不作为内置)
```

## 对 Runtime 的改造

### main_loop.py — `on_llm_response` 触发点

```python
# 当前
cleaned = _extract_evidence_tags(assistant_response_text, turn_count, self.session_memory)
if cleaned != assistant_response_text:
    assistant_response_text = cleaned
    _strip_evidence_from_last_assistant(message_history)
...
cleaned = _extract_offload_evidence(assistant_response_text, self.context_manager)
if cleaned != assistant_response_text:
    assistant_response_text = cleaned

# 迁移后
ext_ctx = self._build_extraction_ctx(turn_count, effective_mode)
assistant_response_text = await self.profile.run_strategies_on_llm_response(
    assistant_response_text, ext_ctx,
)
# Runtime 统一做 tag 清理（输出卫生）
assistant_response_text = _strip_framework_tags(assistant_response_text)
_strip_framework_tags_from_last_assistant(message_history)
```

### context_manager.py — 新增 `on_offload` 触发点（Phase 2b）

在 `add_result_offload` 写文件后调用：

```python
async def add_result_offload(self, tool_name, content, ...):
    ref = ...  # 生成 ref + 写文件
    # 新增：触发 strategy
    if self._profile_adapter:  # 通过 orchestrator 注入
        await self._profile_adapter.run_strategies_on_offload(
            ref, tool_name, content, ext_ctx,
        )
    return ref
```

### window_strategy.py — `on_compact` 触发点（Phase 2a）

LLMSummarize 完成后调用 `strategy.on_compact(summary, cutoff_turn, ctx)`，不再硬编码 `_extract_evidence_from_summary`。

### tool_executor.py — `on_tool_result` 触发点（Phase 2b）

工具结果回来后、注入 message_history 前调用。并发工具场景：每个工具单独触发（与并发语义对齐）。

## Profile 集成

### Profile 基类加 `extraction_strategies`

```python
class Profile(ABC):
    # 子类通过类属性或 __init__ 设置默认
    default_extraction_strategies: list[MemoryExtractionStrategy] = []

    def __init__(self, config=None):
        config = config or {}

        # 配置逻辑：extraction_strategies 完全覆盖，extraction_strategies_extra 追加
        if "extraction_strategies" in config:
            self.extraction_strategies = list(config["extraction_strategies"])
        else:
            self.extraction_strategies = list(self.__class__.default_extraction_strategies)
            if "extraction_strategies_extra" in config:
                self.extraction_strategies.extend(config["extraction_strategies_extra"])

    # run_strategies_on_* 方法（见前）
```

### StandardProfile

```python
class StandardProfile(Profile):
    name = "standard"
    default_extraction_strategies = [
        OffloadEvidenceStrategy(),
        SummaryEvidenceStrategy(),
    ]
```

### DeepResearchProfile

```python
class DeepResearchProfile(Profile):
    name = "deep_research"
    default_extraction_strategies = [
        OffloadEvidenceStrategy(),
        SummaryEvidenceStrategy(),
        EvidenceTagStrategy(),  # 研究专属追加
    ]
```

## Snapshot / Restore 契约（HITL 集成）

Profile 的 `snapshot()` 内部递归包含 strategies 的 snapshot：

```python
class Profile(ABC):
    def snapshot(self) -> dict:
        return {
            "name": self.name,
            "strategies": {
                strat.name: strat.snapshot()
                for strat in self.extraction_strategies
            },
        }

    def restore(self, state: dict) -> None:
        # 按 name 匹配恢复，宽松处理不匹配项
        strategy_states = state.get("strategies", {})
        for strat in self.extraction_strategies:
            if strat.name in strategy_states:
                strat.restore(strategy_states[strat.name])
```

RuntimeSnapshot 只需要存 `profile.snapshot()`，不需要单独为 strategies 存字段。

- `OffloadEvidenceStrategy` / `SummaryEvidenceStrategy` / `EvidenceTagStrategy` 都无状态，snapshot 返回 `{}`
- `FactExtractionStrategy` 可能维护 "已抽取的 tool_result 哈希集合"，需要 snapshot
- 用户自定义的 vector store / 外部存储 strategy 如需 snapshot（例如 `stored_refs` 集合），在 strategy 内部实现 `snapshot()` / `restore()` 即可

## Strategy 异常处理

**不吞异常**。Strategy 抛错应暴露给上层，由 Runtime catch 并 fail-fast 或重试（按现有主循环异常处理规则）。

这和 Profile 钩子的异常处理保持一致 —— Strategy 是框架扩展机制，不是用户业务 hook，行为应该可预测。

## 用户自定义 Strategy

### 继承 + 传入

```python
class MyStrategy(MemoryExtractionStrategy):
    name = "my_strategy"

    async def on_llm_response(self, text, ctx):
        # 自定义逻辑
        return text

# 直接传入
dr = DeepResearch(
    profile="standard",
    profile_config={
        "extraction_strategies_extra": [MyStrategy()],
    },
)
```

### 注册到 registry（用于 YAML 配置）

```python
from mem_deep_research_core.memory_extraction import register_strategy

register_strategy(MyStrategy)

# YAML 里引用
# extraction_strategies_extra:
#   - name: my_strategy
```

## Phase 实施

### Phase 2a（首批）

- 建 `memory_extraction/` 目录
- 实现 `MemoryExtractionStrategy` ABC + `ExtractionContext`
- 实现 `OffloadEvidenceStrategy`（封装 `_extract_offload_evidence`）
- 实现 `SummaryEvidenceStrategy`（封装 `_extract_evidence_from_summary`）
- 实现 `EvidenceTagStrategy`（封装 `_extract_evidence_tags`）
- Profile 基类加 `extraction_strategies` + `run_strategies_on_*` 方法
- StandardProfile 默认 = `[OffloadEvidence, SummaryEvidence]`
- DeepResearchProfile（新建，空实现除了 default strategies）默认 = `[OffloadEvidence, SummaryEvidence, EvidenceTag]`
- Runtime 改为调 `profile.run_strategies_on_llm_response` 和 `on_compact`
- Runtime 统一 tag 清理 (`_strip_framework_tags`)
- 测试：原 evidence 相关 e2e 行为等价

**行为等价保证**：StandardProfile + DeepResearchProfile 运行时调用的 strategies 总集 = 原来硬编码执行的抽取逻辑，零行为差异。

### Phase 2b

- 为 runtime 加 `on_tool_result` 触发点（`tool_executor.py`）
- 为 runtime 加 `on_offload` 触发点（`context_manager.py`）
- 实现 `FactExtractionStrategy`（内置但默认不启用）
- 实现 `SummarizeOnCompactStrategy`（内置但默认不启用）
- 文档 + example project 展示 strategy 组合

### Phase 2c（可选）

- Vector store / RAG 接入：用户自定义 strategy（框架不内置，见 "用户自定义 Strategy 示例" 章节）
- 不做：webhook notification、encryption 等非 extraction 职责的 strategy

## 决策汇总

| # | 决策 | 结论 |
|---|------|------|
| 1 | Strategies 归属 | **全部在 Profile 层**。Runtime 不持有 strategies 列表，通过 `profile.run_strategies_on_*` 调用 |
| 2 | Strategy 调用顺序 | **按 list 顺序**，user 配置顺序 = 执行顺序 |
| 3 | Runtime tag 清理 vs Strategy 修改 text | **Strategy 链完成后 runtime 统一清理**，strategy 只读为主 |
| 4 | Strategy 访问 context_manager 的权限 | **暴露完整实例**，靠自律；不做窄 view |
| 5 | `on_tool_result` 并发时触发方式 | **每个工具单独触发**；需 batch 的 strategy 内部维护 buffer |
| 6 | Profile 和 Strategy 的 snapshot 交互 | Profile.snapshot() 递归包含 strategies；按 name 匹配恢复 |
| 7 | Phase 2a 是否同步迁移 prompt 模板 | **不迁**，保持当前 prompt 结构。Phase 2b 和其他 profile 专属逻辑一起迁 |
| 8 | StandardProfile 的 "细节保鲜" 能力 | StandardProfile 默认带 `[OffloadEvidence, SummaryEvidence]`，base profile 的实用基线 |
| 9 | 用户如何完全覆盖 vs 追加 | 两个 config key：`extraction_strategies`（覆盖）vs `extraction_strategies_extra`（追加） |
| 10 | Strategy 异常处理 | 不吞，抛给 runtime 处理 |

## 关键风险

| 风险 | 缓解 |
|------|------|
| 多 strategy 修改 assistant_text 相互干扰 | 接口约定 strategy 原则上只读；文档说清；list 顺序明确 |
| Strategy 组合下性能退化（N 个 strategy × 每轮调用） | 无状态 strategy 执行快；需要 LLM 的 strategy（FactExtraction）用户显式启用 |
| Strategy 内部状态在 resume 后漂移 | snapshot/restore 契约；golden test |
| 用户自定义 strategy 抛异常影响主流程 | 不吞异常，文档强调 strategy 应自身处理容错 |
| 用户误用 `extraction_strategies` 完全覆盖导致细节保鲜失效 | 文档强调两个 config key 语义差异；默认推荐 `_extra` |

## 不做

- 不做 strategy 优先级 / 依赖管理（按 list 顺序即可）
- 不做 strategy 条件性跳过（用户不想要就不传这个 strategy）
- 不做 strategy 级 Hook（strategy 本身就是扩展点）
- 不做 strategy 的线程级 / 进程级并行执行（每个 strategy 串行调用，避免状态竞争）
