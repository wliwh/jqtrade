# 项目文档

## 当前入口

- [工程总览](../README.md)
- [backtest_executor 使用说明](../backtest_executor/README.md)
- [下一阶段计划](planning/strategy_research_automation_pipeline_design.md)
- [ETF 现役策略](../strategies/etf_rotation/README.md)

## JQ 参考

- [`reference/joinquant/JQ_backtest_API.md`](reference/joinquant/JQ_backtest_API.md)：项目整理过的 `create_backtest()`、`get_backtest()` 研究接口说明。
- [`reference/joinquant/joinquant-api-docs.md`](reference/joinquant/joinquant-api-docs.md)：较完整的 JQ API 文档快照，适合全文搜索；具体行为仍以实际 JQ 环境为准。

## 历史资料

- [`reports/forum/articles/`](reports/forum/articles/)：策略比较和回测研究文章；相邻 `assets/` 保存其配图。
- [`archive/`](archive/)：不再代表当前工程的旧进度与外部 TDX 数据工程快照。
- [`../backtest_executor/archive/`](../backtest_executor/archive/)：旧回测系统、旧设计和旧配置。

新文档应先判断受众：项目入口写入 README，现行设计写入对应 `docs/`，已完成或失效的计划移入 archive。不要在当前文档中保留不存在的代码路径或另一工程的运行状态。
