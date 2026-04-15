# Dual-Mode Execution Status

本页不再保留早期的 refactor 提案细节，只记录当前“双模执行”已经落地到什么程度，以及还缺什么。

## 当前状态

Dual-mode 能力已经落地，但仍然建立在同一个主循环上。

换句话说，当前状态是：

- `quick / standard / deep / auto` 已可用
- `auto` 已由 `LLMRouter` 驱动
- quick 已有轻量 fast path
- deep 已有研究增强能力
- 但三种模式还没有拆成独立 runner

## 已经落地的内容

### `auto` 路由已接通

- 支持 hook 分类
- 支持结构信号
- 支持 `router_model`
- 未配置 `router_model` 时回退主模型分类

### quick 已经不只是“少跑几轮”

- 动态裁剪重型内置工具
- 注入 quick preset
- 不走 reflection / verify

### deep 已具备研究增强

- reflection checkpoint
- sub-agent
- verify checkpoint
- final summary 强制收尾

## 仍未完成的部分

### 1. 还没有独立 runner

当前仍是：

```text
一个 MainLoopRunner
  + 多个 mode 分支
```

目标仍建议演进为：

```text
ModeResolver
  -> QuickRunner
  -> StandardRunner
  -> DeepRunner
```

### 2. mode profile 还没有配置化

目前 quick / standard / deep 的差异主要体现在代码分支里，尚未完全收敛成清晰的 mode profile。

### 3. benchmark 还偏 smoke

模式已经可以跑，但还缺：

- 更稳定的任务集
- 更完整的质量指标
- 明确的 acceptance standard

## 验收标准

当下面几点满足时，可以认为 dual-mode 基本完成：

- quick 明显比 standard / deep 更轻更快
- deep 在研究任务上稳定优于 standard
- `auto` 路由结果可解释、可调试、可复现
- mode 行为差异主要由独立 runner 或 mode profile 表达，而不是散落条件分支

## 推荐下一步

1. 先拆 runner
2. 再收 mode profile
3. 最后补 benchmark / eval

## 相关文档

- [13-execution-modes](./13-execution-modes.md)
- [15-technical-roadmap](./15-technical-roadmap.md)
- [17-repo-architecture-review](./17-repo-architecture-review.md)
