# Industry Framework Analysis and Optimization Directions

> 状态基线：2026-04-20
> 文档目标：从更广的 Agent Framework 产业格局出发，分析 Mem Deep Research 的后续优化方向，而不是只围绕 deep research 单点继续演进。

## 一句话结论

业界主流 Agent 框架的竞争焦点，已经不再是“谁能做一个会搜索和写总结的研究 Agent”，而是在向下面几类能力收敛：

- `workflow + agent` 混合编排
- durable execution / suspend / resume
- first-class tracing / eval / observability
- tool protocol standardization
- guardrails 与安全边界前移
- 从本地框架走向托管 runtime

从这个视角看，Mem Deep Research 的下一阶段不应该只强化 deep research 本身，而应该逐步升级为一个更通用的 Agent Runtime。

## 产业格局：主流框架在往哪里走

### 1. 低层 orchestration runtime

这一类框架强调状态、持久化、复杂控制流和长任务执行。

代表：

- LangGraph：主打 durable execution、human-in-the-loop、memory、deployment，明确把自己定位为低层 agent orchestration runtime。
- Google ADK：强调 modular、model-agnostic、deployment-agnostic，并和 Vertex AI Agent Engine 强绑定。
- AutoGen Core：强调 event-driven、distributed、scalable、resilient multi-agent systems。
- Semantic Kernel Process Framework：强调 event-driven process orchestration、可复用步骤和审计能力。

共同趋势：

- 不再把 agent loop 当成唯一抽象，而是把 agent 放进更大的 workflow / process / graph 里。
- 状态恢复、暂停、审计、分布式执行正在变成框架级能力，而不是业务层补丁。

## 2. 应用层 agent harness

这一类框架更关注开发者体验、默认能力和快速落地。

代表：

- OpenAI Agents SDK：主打 handoffs、tracing、guardrails、tool use、sandbox execution。
- OpenAI Agent Builder：主打可视化 workflow、typed inputs/outputs、node-based orchestration、版本化发布。
- CrewAI：主打 crews + flows，把多 Agent 协作和事件驱动工作流打包成统一体验。
- Mastra：主打 agents、workflows、memory、MCP、streaming、evals、tracing 的一体化开发体验。
- PydanticAI：主打类型安全、依赖注入、可观测性，以及对 durable execution 方案的官方集成。

共同趋势：

- 单一“聊天循环”正在被更高层的 application harness 取代。
- 框架不只提供模型调用和工具调度，还提供 guardrails、tracing、evals、memory、deployment 接口。
- 连以模型 API 为核心的平台方，也开始把“workflow 编排 + 版本化发布 + 可视化调试”当成一等能力。

## 3. 协议与生态层

这一层不是单个框架，而是整个行业共同采用的能力接口。

代表：

- MCP（Model Context Protocol）：正在成为工具和外部系统接入的通用层。
- OpenAI Responses API / built-in tools：把 web search、file search、computer use、remote MCP 等能力做成模型原生接口。
- OpenAI Agent Builder / node system：把 agent workflow 的节点化编排进一步产品化。

共同趋势：

- 工具接入从“每家框架各写各的 adapter”转向“统一协议 + 统一工具层”。
- 一旦工具协议标准化，竞争重心就会从“能不能接工具”变成“能不能安全、稳定、可观测地接工具”。

## 主流框架的真实分野

如果把这些框架放到同一张图里，它们的差异主要不在“能不能做多 Agent”，而在下面几个维度：

| 维度 | 行业主流做法 | 代表框架 |
|------|-------------|---------|
| 编排抽象 | graph / process / flow / event-driven runtime | LangGraph, ADK, Semantic Kernel, CrewAI, Mastra |
| 运行时持久化 | sessions, snapshots, durable execution, workflow state | LangGraph, ADK, Mastra, PydanticAI |
| 多 Agent 协作 | handoff, delegation, crews, actor messaging | OpenAI Agents SDK, CrewAI, AutoGen |
| 可观测性 | tracing 默认开启，span 级别记录 LLM / tool / handoff | OpenAI Agents SDK, LangGraph ecosystem, Mastra |
| 安全控制 | input/output/tool guardrails, HITL, policy boundary | OpenAI Agents SDK, LangGraph, Mastra |
| 托管运行时 | cloud runtime / managed deployment / agent engine | OpenAI, Google, CrewAI Enterprise, LangGraph Platform |

这说明一个很重要的判断：

**未来框架竞争不是单点能力竞争，而是 runtime 完整性竞争。**

## 对 Mem Deep Research 的启发

当前项目的优势，是已经有了一个很扎实的研究型执行内核：

- 主循环
- context 管理
- tool 执行
- sub-agent
- memory / todo / transcript
- offload / resume

但如果和主流框架对比，当前还更像：

**一个强研究内核 + 一些框架能力**

而不是：

**一个通用 Agent Runtime，研究只是其中一种 profile**

这会带来三个问题：

1. 继续优化 deep research，本质上是在强化一个“场景模板”，而不是扩大 runtime 的通用性。
2. 主循环承担了太多“研究增强逻辑”，未来想扩展到 automation、ops、coding、enterprise workflow 时会越来越重。
3. 对外叙事还容易被理解成“一个 research agent 项目”，而不是“一个可扩展的 Agent Framework”。

## 后续优化方向

下面这些方向，是结合业界框架演进路线后，更值得投入的主线。

## 方向 1：从 Research Runtime 升级为 General Agent Runtime

这是最核心的方向。

建议把 deep research 定位从“框架本体”调整为“框架上的高级执行 profile”。

对应动作：

- 保留 `quick / standard / deep`，但把 deep 明确为一种 profile
- 新增更通用的 workflow / automation / coding profile 概念
- 把研究专属逻辑逐步从主循环抽离到 policy / profile 层

如果这一步不做，后续任何新场景都会继续往 `MainLoopRunner` 里堆条件分支。

## 方向 2：补 Workflow Layer，而不是只强化 Agent Loop

业界已经很清楚地分成了两层：

- workflow：开发者定义结构化控制流
- agent：在局部节点里提供开放式推理和工具调用能力

Mem Deep Research 现在 agent loop 很强，但缺一个更正式的 workflow layer。

建议新增：

- `Flow` / `Process` / `TaskGraph` 一类抽象
- 显式支持顺序、并行、路由、等待、人工确认、子流程
- 允许 Agent 作为 workflow 的节点，而不是把所有复杂性都塞进单个 agent loop

这会让框架从“研究任务执行器”更自然地升级为“复杂任务编排器”。

## 方向 3：把 Durable Execution 做成正式 Contract

业界成熟框架都在强化这一点：

- LangGraph 强调 durable execution
- Google 强调 sessions / memory / agent engine
- Mastra 强调 suspend / resume / snapshots
- PydanticAI 明确提供 durable execution 集成

你这个项目其实已经有雏形了：

- resume
- offload
- transcript
- checkpoints

但它们还没有彻底收口成统一的 runtime contract。

建议继续演进为：

- 显式 `RuntimeSnapshot`
- 可恢复的 turn state / tool state / dedup state / memory state
- suspend / resume / replay 的统一语义
- 可扩展到外部 durability backend 的接口

这一步完成后，项目的定位会明显向“runtime”而不是“agent demo”靠拢。

## 方向 4：把 Observability 从“有日志”升级为“有运行时控制面”

业界领先框架几乎都把 tracing 放到了核心位置：

- OpenAI Agents SDK 默认 tracing
- LangGraph 强调 debugging / deployment / state visibility
- Mastra 提供 tracing 和 evals
- Semantic Kernel 强调 OpenTelemetry 和审计

当前项目已有：

- transcript
- task_tracer
- perf metrics

但还缺更强的统一运行时视角。

建议补：

- 标准化 span / event schema
- tool、sub-agent、compact、resume、reflection 的统一 trace 事件
- 失败原因分类和可视化字段
- benchmark / eval 与 trace 联动

后面如果想接 Langfuse、OpenTelemetry、LangSmith、Logfire 一类外部观测体系，也会更顺。

## 方向 5：把 Guardrails 前移到 Tool Boundary

行业趋势非常明确：安全控制正在从“最终回答审查”前移到“输入、工具、输出、handoff”每个边界。

OpenAI Agents SDK 已经把 guardrails 分成：

- input guardrails
- output guardrails
- tool guardrails

这对当前项目很有启发。

你现在有：

- hook
- secure context
- message interception

但还缺一个更明确的 policy 层。

建议补：

- tool input / output policy
- agent handoff / sub-agent spawn policy
- context injection allowlist
- MCP server 信任级别和权限边界
- 对高风险工具的审批 / tripwire / fallback 机制

这个方向的重要性已经不只是“安全增强”，而是未来通用 Agent Framework 的基础能力。

## 方向 6：把 MCP 支持从“接工具”升级为“接生态”

MCP 正在成为行业工具接入层，这对项目是机会。

当前项目基于 MCP 做工具系统，本身方向是对的。下一阶段更值得做的不是“再加几个内置工具”，而是把 MCP 能力做深：

- 更清晰的 client / server capability model
- server session 生命周期治理
- 认证、权限、隔离和审计
- tool schema 缓存与版本控制
- remote MCP / local MCP / hosted MCP 的统一抽象

换句话说，MCP 不只是 transport 问题，而应该成为框架的外部能力总线。

## 方向 7：从 Memory Feature 升级为 Stateful Runtime

业界现在对 memory 的理解已经不只是“给模型塞一点历史”。

更成熟的做法是分层：

- working memory
- session state
- cross-session memory
- resource-linked memory
- workflow state

当前项目有 `SessionMemory` 和 `LongTermMemory`，这是不错的起点。

后续可以继续往下拆：

- memory 和 runtime state 分离
- memory 和 transcript / task state / offload registry 解耦
- 引入资源级 memory，例如围绕 task / user / project / artifact 的关联

这样 memory 就不只是研究 Agent 的辅助件，而会变成通用运行时能力。

## 方向 8：把评测体系产品化

Mastra、LangGraph 生态、OpenAI 的 tracing / eval 方向都说明了一件事：

框架最终会进入“可评测、可比较、可回归”的竞争。

对这个仓库来说，后续非常值得做的是：

- benchmark 分层
- 任务级回放和重评分
- 结果质量、工具效率、成本、时延、恢复成功率统一度量
- 针对 mode / provider / tool transport 的回归集

研究型框架如果没有评测体系，后面很容易退化成“改了很多，但不知道是不是更好”。

## 方向 9：给 Hosted / Managed Runtime 留接口

业界几乎都在往托管 runtime 走：

- OpenAI 在强化 Agents SDK + sandbox + hosted tooling
- Google 把 ADK 和 Agent Engine 绑定
- LangGraph 明确有 deployment/runtime 叙事

你不一定要马上做云平台，但架构上最好给这条路留口：

- runtime state 外部化
- tool session 生命周期可托管
- trace / eval / checkpoint 可持久化
- 支持 worker model 或 queue-based execution

这会决定项目后面能不能自然走向服务化，而不是永远停留在单机本地框架。

## 不建议继续作为主线加码的方向

结合业界框架的演进节奏，下面几件事不建议继续作为主线投入：

- 只继续优化 deep research prompt 和 reflection 细节
- 持续往主循环里塞更多研究专属 heuristics
- 先堆更多 builtin tools，再考虑 runtime 边界
- 在 workflow 抽象缺位时，过早强化多 Agent 花样

原因不是这些方向没价值，而是它们更像“能力插件”，不是当前阶段最稀缺的“框架底座”。

## 建议的优先级顺序

如果按未来两个大版本来排，我会建议这样推进：

### P0：Runtime Contract 收敛

- 收口 `resume + offload + read_result + checkpoint`
- 统一 tool / message / state 生命周期
- 强化 MCP session、policy、权限边界

### P1：Workflow Layer + Mode/Profile 重构

- 引入 flow/process/task graph 抽象
- 把 deep research 从主循环逻辑降为 profile
- 把多场景能力挂到统一 execution profile 层

### P2：Observability + Eval 平台化

- trace schema
- replay / benchmark
- 跨 provider / mode 的质量和成本比较

### P3：Managed Runtime 准备

- 外部持久化接口
- worker / queue / hosted execution 预留
- 面向服务化的 runtime API

## 最终判断

如果只从 deep research 继续优化，这个项目会变成一个很强的研究 Agent 框架。

如果顺着业界主流路线往前走，它有机会变成：

**一个以 research 为强项、但不被 research 场景绑定的通用 Agent Runtime。**

这两条路的上限完全不一样。

## 参考资料

- LangGraph overview: [docs.langchain.com/oss/python/langgraph/overview](https://docs.langchain.com/oss/python/langgraph/overview)
- LangGraph workflows and agents: [docs.langchain.com/oss/python/langgraph/workflows-agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents)
- OpenAI Responses API: [platform.openai.com/docs/guides/responses-vs-chat-completions](https://platform.openai.com/docs/guides/responses-vs-chat-completions)
- OpenAI Agents SDK: [platform.openai.com/docs/guides/agents-sdk](https://platform.openai.com/docs/guides/agents-sdk/)
- OpenAI Agent Builder: [platform.openai.com/docs/guides/agent-builder](https://platform.openai.com/docs/guides/agent-builder)
- OpenAI node reference: [platform.openai.com/docs/guides/node-reference](https://platform.openai.com/docs/guides/node-reference)
- OpenAI agent evals: [platform.openai.com/docs/guides/agent-evals](https://platform.openai.com/docs/guides/agent-evals)
- OpenAI Agents SDK handoffs: [openai.github.io/openai-agents-python/handoffs](https://openai.github.io/openai-agents-python/handoffs/)
- OpenAI Agents SDK tracing: [openai.github.io/openai-agents-python/tracing](https://openai.github.io/openai-agents-python/tracing/)
- OpenAI Agents SDK guardrails: [openai.github.io/openai-agents-python/guardrails](https://openai.github.io/openai-agents-python/guardrails/)
- OpenAI Responses API built-in tools update: [openai.com/index/new-tools-and-features-in-the-responses-api](https://openai.com/index/new-tools-and-features-in-the-responses-api/)
- OpenAI Agents SDK sandbox update: [openai.com/index/the-next-evolution-of-the-agents-sdk](https://openai.com/index/the-next-evolution-of-the-agents-sdk)
- Google ADK overview: [docs.cloud.google.com/agent-builder/agent-development-kit/overview](https://docs.cloud.google.com/agent-builder/agent-development-kit/overview)
- Vertex AI Agent Engine Sessions: [docs.cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/sessions/overview](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/sessions/overview)
- Semantic Kernel Agent Framework: [learn.microsoft.com/en-us/semantic-kernel/frameworks/agent](https://learn.microsoft.com/en-us/semantic-kernel/frameworks/agent/)
- Semantic Kernel Process Framework: [learn.microsoft.com/en-us/semantic-kernel/frameworks/process/process-framework](https://learn.microsoft.com/en-us/semantic-kernel/frameworks/process/process-framework)
- AutoGen stable docs: [microsoft.github.io/autogen/stable](https://microsoft.github.io/autogen/stable/index.html)
- AutoGen Core: [microsoft.github.io/autogen/stable/user-guide/core-user-guide](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/index.html)
- CrewAI introduction: [docs.crewai.com/en/introduction](https://docs.crewai.com/en/introduction)
- CrewAI flows: [docs.crewai.com/en/concepts/flows](https://docs.crewai.com/en/concepts/flows)
- Mastra workflows: [mastra.ai/workflows](https://mastra.ai/workflows)
- Mastra docs overview pages: [mastra.ai/en/docs](https://mastra.ai/en/docs)
- PydanticAI overview: [pydantic.dev/docs/ai/overview](https://pydantic.dev/docs/ai/overview/)
- PydanticAI durable execution: [pydantic.dev/docs/ai/integrations/durable_execution/overview](https://pydantic.dev/docs/ai/integrations/durable_execution/overview/)
- MCP introduction: [modelcontextprotocol.io/schema/v1](https://modelcontextprotocol.io/schema/v1)
- Anthropic MCP docs: [docs.anthropic.com/en/docs/mcp](https://docs.anthropic.com/en/docs/mcp)
