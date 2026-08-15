# backtest_executor 文档与下一步计划

- 更新日期：2026-08-15
- 状态：策略研究自动化流水线是下一阶段唯一现行计划

## 下一步

从 [`../../docs/planning/strategy_research_automation_pipeline_design.md`](../../docs/planning/strategy_research_automation_pipeline_design.md) 开始。该设计覆盖 JQ 社区候选发现、源码留存与审计、参数化、短区间冒烟回测、自适应参数搜索、完整结果缓存和研究文档生成。

实施时按文档中的阶段门推进，先修复实验身份、锚点策略 ID、mapper 安全和模拟指标等执行风险，再扩展自动化范围。

## 其余文档

- [`../../docs/reports/forum/articles/`](../../docs/reports/forum/articles/)：策略比较和参数研究文章，属于研究记录，不是当前实施清单。
- [`../README.md`](../README.md)：当前框架使用说明。
- [`../archive/README.md`](../archive/README.md)：旧回测程序、已完成的旧设计、旧配置和仍可迁移的能力。

当前代码入口位于 `backtest_executor/` 包根目录。已完成或失效的计划不再留在 `docs/planning/`。
