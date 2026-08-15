# 项目文档

## 当前入口

- [工程总览](../README.md)
- [文档检索索引](INDEX.md)
- [backtest_executor 使用说明](../backtest_executor/README.md)
- [下一阶段计划](planning/strategy_research_automation_pipeline_design.md)
- [ETF 现役策略](../strategies/etf_rotation/README.md)
- [指数顶底信号研究](../research/index_turning_points/README.md)

## JQ 参考

- [`reference/joinquant/`](reference/joinquant/README.md)：**策略开发首选**。包含聚宽官网完整 API 快照、原始 HTML 和按主题划分的全文检索 Markdown。

## 历史资料

- [`reports/forum/articles/`](reports/forum/articles/)：策略比较和回测研究文章；相邻 `assets/` 保存其配图。
- [`archive/`](archive/)：不再代表当前工程的旧进度与外部 TDX 数据工程快照。
- [`../backtest_executor/archive/`](../backtest_executor/archive/)：旧回测系统、旧设计和旧配置。

新文档应先判断受众：项目入口写入 README，现行设计写入对应 `docs/`，已完成或失效的计划移入 archive。不要在当前文档中保留不存在的代码路径或另一工程的运行状态。
