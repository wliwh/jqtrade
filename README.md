# jqtrade

`jqtrade` 是一组围绕聚宽（JQ）的 ETF 轮动策略、资产池研究和参数回测工具。策略文件主要在 JQ 回测或研究环境中运行；本地代码负责参数生成、结果分析、数据导出和研究辅助，不构成统一的本地交易应用。

## 当前重点

- 策略工程：[研究自动化流水线](docs/planning/strategy_research_automation_pipeline_design.md)；执行入口见 [`backtest_executor/`](backtest_executor/README.md)。
- 独立研究：[指数顶底区域与点时信号](research/index_turning_points/README.md)，不纳入策略开发流程。

## 目录

| 目录 | 用途 | 状态 |
| --- | --- | --- |
| [`strategies/etf_rotation/`](strategies/etf_rotation/README.md) | 4 个现役 JQ ETF 策略、辅助工具和旧策略归档 | 活跃 |
| [`backtest_executor/`](backtest_executor/README.md) | YAML 驱动的 JQ 参数回测与结果分析 | 活跃 |
| [`research/pools/`](research/pools/README.md) | ETF 池过滤、聚类、族谱和 PCA 研究 | 研究中 |
| [`research/momentum_signal_validation/`](research/momentum_signal_validation/README.md) | JQ 原始数据导出与本地横截面动量研究 | 研究中 |
| [`research/micro/`](research/micro/README.md) | 市场宽度和一致性研究 | 探索中 |
| [`research/index_turning_points/`](research/index_turning_points/README.md) | 事后顶底区域、点时信号与两套离线评测，不开发交易策略 | 研究中 |
| [`tools/jq_data_export/`](tools/jq_data_export/README.md) | JQ 数据导出和批次续传工具 | 辅助工具 |
| [`docs/`](docs/INDEX.md) | 计划、JQ 参考、研究报告与历史归档 | 文档 |
| `tests/` | 回测辅助、数据导出和研究模块的本地确定性测试 | 测试 |

## 快速验证

本地测试不执行真实 JQ 回测：

```bash
pytest -q
```

真实策略运行和参数回测需要 JQ 研究环境提供 `jqdata`、`create_backtest()` 和 `get_backtest()`；具体用法见 [backtest_executor/README.md](backtest_executor/README.md)。

任何准备复制或注入 JQ 投资研究、回测或模拟盘的程序，先遵守 [JQ 目标程序运行时兼容性](docs/reference/joinquant/jq_research_compatibility.md)，再按具体 API 查询官方快照。

文档检索从 [`docs/INDEX.md`](docs/INDEX.md) 开始。现役策略清单和“最新”的含义以 [`strategies/etf_rotation/README.md`](strategies/etf_rotation/README.md) 为准；归档和历史文章只用于追溯，策略表现必须以明确区间、成本口径和实际 JQ 回测 ID 为依据。
